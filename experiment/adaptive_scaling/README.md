# env

```bash
# Once.
docker pull swr.cn-east-3.myhuaweicloud.com/wden/wden:devel-cpu-ubuntu20.04-python3.8 && docker tag swr.cn-east-3.myhuaweicloud.com/wden/wden:devel-cpu-ubuntu20.04-python3.8 wden/wden:devel-cpu-ubuntu20.04-python3.8

docker pull swr.cn-east-3.myhuaweicloud.com/wden/wden:devel-cuda11.3.1-cudnn8-ubuntu20.04-python3.8 && docker tag swr.cn-east-3.myhuaweicloud.com/wden/wden:devel-cuda11.3.1-cudnn8-ubuntu20.04-python3.8 wden/wden:devel-cuda11.3.1-cudnn8-ubuntu20.04-python3.8

mkdir -p "$HOME"/container/vkit-open-model
touch "$HOME"/container/vkit-open-model/bash_history
touch "$HOME"/container/vkit-open-model/screen_daemon.log

CUSTOMIZED_INIT_SH=$(
cat << 'EOF'

cd


cd cache

if [ ! -f 'torch-1.12.0+cpu-cp38-cp38-linux_x86_64.whl' ]; then
    wget https://download.pytorch.org/whl/cpu/torch-1.12.0%2Bcpu-cp38-cp38-linux_x86_64.whl
fi
if [ ! -f 'torch-1.12.0+cu113-cp38-cp38-linux_x86_64.whl' ]; then
    wget https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
fi

if [ ! -f 'torchvision-0.13.0+cpu-cp38-cp38-linux_x86_64.whl' ]; then
    wget https://download.pytorch.org/whl/cpu/torchvision-0.13.0%2Bcpu-cp38-cp38-linux_x86_64.whl
fi
if [ ! -f 'torchvision-0.13.0+cu113-cp38-cp38-linux_x86_64.whl' ]; then
    wget https://download.pytorch.org/whl/cu113/torchvision-0.13.0%2Bcu113-cp38-cp38-linux_x86_64.whl
fi

pip install 'torch-1.12.0+cu113-cp38-cp38-linux_x86_64.whl' 'torchvision-0.13.0+cu113-cp38-cp38-linux_x86_64.whl'


cd

if [ ! -d vkit ]; then
    git clone git@github.com:vkit-x/vkit.git
fi
cd vkit
git pull


cd

if [ ! -d vkit-open-model ]; then
    git clone git@github.com:vkit-x/vkit-open-model.git
fi
cd vkit-open-model

echo 'export VKIT_ARTIFACT_PACK="/dev/shm/vkit_artifact_pack"' >> .envrc.private
direnv allow

git pull
pip install -e ../vkit
pip install -e .'[dev]'

cd


cp -r /vkit-x/vkit-private-data/vkit_artifact_pack /dev/shm/


EOF
)
echo "$CUSTOMIZED_INIT_SH" | tee "$HOME"/container/vkit-open-model/customized_init.sh > /dev/null


nvidia-docker run \
  --name vkit-open-model \
  -d --rm -it \
  --shm-size='13g' \
  --network host \
  --user "$(id -u):$(id -g)" \
  -v "$HOME"/vkit-x:/vkit-x \
  -e CD_DEFAULT_FOLDER=/vkit-x \
  -v "$HOME"/.gitconfig:/etc/gitconfig:ro \
  -v "$SSH_AUTH_SOCK":/run/ssh-auth.sock:shared \
  -e SSH_AUTH_SOCK="/run/ssh-auth.sock" \
  -e SSHD_PORT='2222' \
  -e SSH_SOCKS5_PROXY="0.0.0.0:1081" \
  -e APT_SET_MIRROR_TENCENT=1 \
  -e PIP_SET_INDEX_TENCENT=1 \
  -v "$HOME"/container/vkit-open-model/bash_history:/run/.bash_history \
  -e HISTFILE='/run/.bash_history' \
  -v "$HOME"/container/vkit-open-model/screen_daemon.log:/run/screen_daemon.log \
  -e SCREEN_DAEMON_LOG='/run/screen_daemon.log' \
  -v "$HOME"/container/vkit-open-model/customized_init.sh:/run/customized_init.sh \
  -e CUSTOMIZED_INIT_SH='/run/customized_init.sh'\
  wden/wden:devel-cuda11.3.1-cudnn8-ubuntu20.04-python3.8
```

## dataset performance profiling

```bash
# Single process.
fib tests/test_adaptive_scaling.py:profile_adaptive_scaling_dataset \
    --num_workers="0" \
    --batch_size="10" \
    --epoch_size="320"

OUTPUT <<
total: 391
per_batch: 39.1
per_batch std: 0.8774018374581855
OUTPUT

# All processes.
fib tests/test_adaptive_scaling.py:profile_adaptive_scaling_dataset \
    --num_workers="32" \
    --batch_size="10" \
    --epoch_size="320"

OUTPUT <<
total: 38
per_batch: 3.8
per_batch std: 0.5240691151937882
OUTPUT
```


```bash
fib tests/test_adaptive_scaling.py:sample_adaptive_scaling_dataset \
    --num_workers="0" \
    --batch_size="20" \
    --epoch_size="5"  \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/sample_adaptive_scaling_dataset"
```


## training

Sync pipeline conifg.

```bash
rsync -r "$VKIT_PRIVATE_DATA/vkit_artifact_pack/pipeline/text_detection/" "${VKIT_ARTIFACT_PACK}/pipeline/text_detection"
```

Train.

```bash
fib experiment/adaptive_scaling/train.py:train \
    --adaptive_scaling_dataset_steps_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210711" \
    --reset_output_folder
```

```bash
fib experiment/adaptive_scaling/train.py:train \
    --adaptive_scaling_dataset_steps_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210713"
```

```bash
fib experiment/adaptive_scaling/train.py:train \
    --adaptive_scaling_dataset_steps_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210714"
```

```bash
fib experiment/adaptive_scaling/train.py:train \
    --adaptive_scaling_dataset_steps_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210715"
```

```bash
fib experiment/adaptive_scaling/train.py:train \
    --adaptive_scaling_dataset_steps_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling.json" \
    --epoch_config_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/overfit_epoch_config.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210715-overfit"
```

```bash
fib experiment/adaptive_scaling/train.py:train \
    --adaptive_scaling_dataset_steps_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210718"
```

```bash
echo '{"neck_head_type": "upernext"}' > "${VKIT_ARTIFACT_PACK}/pipeline/text_detection/upernext_model_config.json"

fib experiment/adaptive_scaling/train.py:train \
    --adaptive_scaling_dataset_steps_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling.json" \
    --model_config_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/upernext_model_config.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210719"
```

NOTE: 20220720, train cli is changed and invalidates the previous notes.

```bash
fib experiment/adaptive_scaling/train.py:train \
    --dataset_config_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling_dataset_config.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210720"
```

```bash
fib experiment/adaptive_scaling/train.py:train \
    --dataset_config_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling_dataset_config.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210721" \
    --restore_state_dict_path="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210720/state_dict_41.pt"
```

```bash
fib experiment/adaptive_scaling/train.py:train \
    --dataset_config_json="${VKIT_ARTIFACT_PACK}/pipeline/text_detection/adaptive_scaling_dataset_config.json" \
    --output_folder="${VKIT_OPEN_MODEL_DATA}/adaptive_scaling_default/20210728"
```

## demo

```bash
fib experiment/adaptive_scaling/demo.py:infer \
    --model_jit_path="$VKIT_OPEN_MODEL_DATA/adaptive_scaling_default/20210711/model.jit" \
    --image_file="$VKIT_OPEN_MODEL_DATA/adaptive_scaling_default/20210711/image.png" \
    --output_folder="$VKIT_OPEN_MODEL_DATA/adaptive_scaling_default/20210711/image-demo"
```
