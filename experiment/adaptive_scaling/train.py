from typing import Tuple, Mapping, Optional, List
from enum import Enum, unique
import logging
import statistics
import shutil

import attrs
import iolite as io
import torch
from torch.utils.data import DataLoader

from vkit.utility import dyn_structure
from vkit_open_model.model import (
    AdaptiveScaling,
    AdaptiveScalingSize,
)
from vkit_open_model.dataset import (
    adaptive_scaling_dataset_collate_fn,
    AdaptiveScalingIterableDataset,
)
from vkit_open_model.loss_function import AdaptiveScalingLossFunction
from vkit_open_model.train import (
    batch_to_device,
    device_is_cuda,
    enable_cudnn_benchmark,
    enable_cudnn_deterministic,
    setup_seeds,
    calculate_iterable_dataset_num_samples,
    generate_iterable_dataset_rng_seeds,
    Metrics,
)

logger = logging.getLogger(__name__)


@attrs.define
class EpochConfig:
    num_epochs: int = 98
    train_num_batches: int = 720
    train_batch_size: int = 3
    train_prefetch_factor: int = 4
    dev_num_batches: int = 90
    dev_batch_size: int = 24
    dev_rng_seed: int = 13
    dev_prefetch_factor: int = 4
    num_workers: int = 8
    avg_num_batches: int = 50


@attrs.define
class OptimizerConfig:
    adamw_lr: float = 2E-3
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    adamw_weight_decay: float = 0.01
    cosine_annealing_warm_restarts_t0: int = 14
    cosine_annealing_warm_restarts_tmulti: int = 2
    cosine_annealing_warm_restarts_eta_min: float = 2E-5


@attrs.define
class LossConfig:
    negative_ratio: float = 3.0
    bce_factor: float = 2.0
    dice_factor: float = 1.0
    l1_factor: float = 1.0


@unique
class MetricsTag(Enum):
    TRAIN_LOSS = 'train_loss'
    DEV_LOSS = 'dev_loss'


@attrs.define
class RestoreState:
    epoch_idx: int
    model_jit_state_dict: Mapping[str, torch.Tensor]
    optimizer_state_dict: Mapping[str, torch.Tensor]
    optimizer_scheduler_state_dict: Mapping[str, torch.Tensor]


def train(
    adaptive_scaling_dataset_steps_json: str,
    output_folder: str,
    device_value: str = 'cuda',
    adaptive_scaling_size: str = 'small',
    epoch_config_json: Optional[str] = None,
    optimizer_config_json: Optional[str] = None,
    loss_config_json: Optional[str] = None,
    restore_state_dict_path: Optional[str] = None,
    reset_epoch_idx: bool = False,
):
    out_fd = io.folder(output_folder)
    assert not out_fd.exists()
    out_fd.mkdir(parents=True)
    logger.info(f'out_fd = {out_fd}')

    # Logging to file.
    logger.addHandler(logging.FileHandler(out_fd / 'log.txt'))

    # Config.
    epoch_config = dyn_structure(
        epoch_config_json,
        EpochConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('epoch_config:')
    logger.info(attrs.asdict(epoch_config))
    io.write_json(out_fd / 'epoch_config.json', attrs.asdict(epoch_config), indent=2)

    optimizer_config = dyn_structure(
        optimizer_config_json,
        OptimizerConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('optimizer_config:')
    logger.info(attrs.asdict(optimizer_config))
    io.write_json(out_fd / 'optimizer_config.json', attrs.asdict(optimizer_config), indent=2)

    loss_config = dyn_structure(
        loss_config_json,
        LossConfig,
        support_path_type=True,
        support_none_type=True,
    )
    logger.info('loss_config:')
    logger.info(attrs.asdict(loss_config))
    io.write_json(out_fd / 'loss_config.json', attrs.asdict(loss_config), indent=2)

    device = torch.device(device_value)
    logger.info(f'device = {device}')

    # Init.
    setup_seeds()
    enable_cudnn_benchmark(device)
    enable_cudnn_deterministic(device)

    # Dataset.
    logger.info('dataset')
    train_num_samples = calculate_iterable_dataset_num_samples(
        num_workers=epoch_config.num_workers,
        batch_size=epoch_config.train_batch_size,
        num_batches=epoch_config.train_num_batches,
    )
    dev_num_samples = calculate_iterable_dataset_num_samples(
        num_workers=epoch_config.num_workers,
        batch_size=epoch_config.dev_batch_size,
        num_batches=epoch_config.dev_num_batches,
    )
    logger.info(f'num_workers={epoch_config.num_workers}')
    logger.info(f'train_num_samples = {train_num_samples}, dev_num_samples={dev_num_samples}')

    shutil.copyfile(
        adaptive_scaling_dataset_steps_json, out_fd / 'adaptive_scaling_dataset_steps.json'
    )
    train_adaptive_scaling_dataset = AdaptiveScalingIterableDataset(
        steps_json=adaptive_scaling_dataset_steps_json,
        num_samples=train_num_samples,
    )
    dev_adaptive_scaling_dataset = AdaptiveScalingIterableDataset(
        steps_json=adaptive_scaling_dataset_steps_json,
        num_samples=dev_num_samples,
        rng_seed=generate_iterable_dataset_rng_seeds(
            num_samples=dev_num_samples,
            rng_seed=epoch_config.dev_rng_seed,
        )
    )

    # Model.
    model = AdaptiveScaling(size=AdaptiveScalingSize(adaptive_scaling_size))
    model_jit: torch.jit.ScriptModule = torch.jit.script(model)  # type: ignore
    model_jit = model_jit.to(device)
    del model

    # Loss.
    loss_function = AdaptiveScalingLossFunction(
        negative_ratio=loss_config.negative_ratio,
        bce_factor=loss_config.bce_factor,
        dice_factor=loss_config.dice_factor,
        l1_factor=loss_config.l1_factor,
    )

    # Optimizer.
    optimizer = torch.optim.AdamW(
        params=model_jit.parameters(),
        lr=optimizer_config.adamw_lr,
        betas=optimizer_config.adamw_betas,
        weight_decay=optimizer_config.adamw_weight_decay,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=optimizer_config.cosine_annealing_warm_restarts_t0,
        T_mult=optimizer_config.cosine_annealing_warm_restarts_tmulti,
        eta_min=optimizer_config.cosine_annealing_warm_restarts_eta_min,
    )

    # Metircs.
    metrics = Metrics(MetricsTag, avg_num_batches=epoch_config.avg_num_batches)

    # Epoch.
    epoch_idx = 0

    # Restore.
    if restore_state_dict_path:
        train_state = dyn_structure(
            torch.load(restore_state_dict_path, device=device),
            RestoreState,
        )
        if not reset_epoch_idx:
            epoch_idx = train_state.epoch_idx
        train_adaptive_scaling_dataset.epoch_idx = train_state.epoch_idx
        model_jit.load_state_dict(train_state.model_jit_state_dict)
        optimizer.load_state_dict(train_state.optimizer_state_dict)  # type: ignore
        optimizer_scheduler.load_state_dict(
            train_state.optimizer_scheduler_state_dict  # type: ignore
        )

    # Dataloader.
    train_data_loader = DataLoader(
        train_adaptive_scaling_dataset,
        collate_fn=adaptive_scaling_dataset_collate_fn,
        batch_size=epoch_config.train_batch_size,
        num_workers=epoch_config.num_workers,
        prefetch_factor=epoch_config.train_prefetch_factor,
        pin_memory=device_is_cuda(device),
        pin_memory_device=str(device) if device_is_cuda(device) else '',
        persistent_workers=True,
    )
    dev_data_loader = DataLoader(
        dev_adaptive_scaling_dataset,
        collate_fn=adaptive_scaling_dataset_collate_fn,
        batch_size=epoch_config.dev_batch_size,
        num_workers=epoch_config.num_workers,
        prefetch_factor=epoch_config.dev_prefetch_factor,
        pin_memory=device_is_cuda(device),
        pin_memory_device=str(device) if device_is_cuda(device) else '',
    )

    best_dev_loss = float('inf')

    while epoch_idx < epoch_config.num_epochs:
        logger.info('Training...')
        model_jit.train()
        torch.set_grad_enabled(True)

        for batch_idx, batch in enumerate(train_data_loader, start=1):
            batch = batch_to_device(batch, device)
            mask_feature, scale_feature = model_jit(batch['image'])

            loss = loss_function(
                mask_feature=mask_feature,
                scale_feature=scale_feature,
                downsampled_mask=batch['downsampled_mask'],
                downsampled_score_map=batch['downsampled_score_map'],
                downsampled_shape=batch['downsampled_shape'],
                downsampled_core_box=batch['downsampled_core_box'],
            )
            avg_loss = metrics.update(MetricsTag.TRAIN_LOSS, float(loss))
            loss.backward()

            optimizer.step()
            optimizer_scheduler.step(
                epoch_idx + (batch_idx - 1) / epoch_config.train_num_batches  # type: ignore
            )
            optimizer.zero_grad()

            if batch_idx % 4 == 0 or batch_idx == epoch_config.train_num_batches:
                logger.info(
                    f'E={epoch_idx}, '
                    f'B={batch_idx}/{epoch_config.train_num_batches}, '
                    f'L={avg_loss:.5f}, '
                    f'LR={optimizer_scheduler.get_last_lr()[-1]:.6f}'
                )

        logger.info('Evaluating...')
        model_jit.eval()
        torch.set_grad_enabled(False)

        dev_losses: List[float] = []
        for batch_idx, batch in enumerate(dev_data_loader, start=1):
            batch = batch_to_device(batch, device)
            mask_feature, scale_feature = model_jit(batch['image'])

            loss = loss_function(
                mask_feature=mask_feature,
                scale_feature=scale_feature,
                downsampled_mask=batch['downsampled_mask'],
                downsampled_score_map=batch['downsampled_score_map'],
                downsampled_shape=batch['downsampled_shape'],
                downsampled_core_box=batch['downsampled_core_box'],
            )
            loss = float(loss)
            dev_losses.append(loss)

            avg_loss = metrics.update(MetricsTag.TRAIN_LOSS, loss)
            if batch_idx % 4 == 0 or batch_idx == epoch_config.dev_num_batches:
                logger.info(
                    f'E={epoch_idx}, '
                    f'B={batch_idx}/{epoch_config.dev_num_batches}, '
                    f'L={avg_loss:.5f}'
                )

        dev_loss = statistics.mean(dev_losses)
        logger.info(f'E={epoch_idx}, dev_loss = {dev_loss}')
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            state_dict_path = out_fd / f'state_dict_{epoch_idx}.pt'
            logger.info(f'E={epoch_idx}, FOR NOW THE BEST, SAVING TO {state_dict_path}')

            restore_state = RestoreState(
                epoch_idx=epoch_idx,
                model_jit_state_dict=model_jit.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                optimizer_scheduler_state_dict=optimizer_scheduler.state_dict(),
            )
            torch.save(attrs.asdict(restore_state), state_dict_path)

        epoch_idx += 1
