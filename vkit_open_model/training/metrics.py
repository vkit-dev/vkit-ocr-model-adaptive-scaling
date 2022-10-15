# This project (vkit-x/vkit-open-model) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from typing import Type, TypeVar, Generic, Dict, Optional, Sequence
from enum import Enum
from queue import Queue

_T = TypeVar('_T', bound=Enum)


class Metrics(Generic[_T]):

    def __init__(self, tag_enum_cls: Type[_T], avg_num_batches: int):
        self.tag_enum_cls = tag_enum_cls
        self.avg_num_batches = avg_num_batches

        self.tag_to_queue: Dict[_T, Queue[float]] = {}
        self.tag_to_avg_value: Dict[_T, Optional[float]] = {}
        self.reset()

    def reset(self, tags: Optional[Sequence[_T]] = None):
        if tags is None:
            tags = tuple(self.tag_enum_cls)
        for tag in tags:
            self.tag_to_queue[tag] = Queue(self.avg_num_batches)
            self.tag_to_avg_value[tag] = None

    def update(self, tag: _T, value: float):
        queue = self.tag_to_queue[tag]
        avg_value = self.tag_to_avg_value[tag]

        queue_size = queue.qsize()
        if queue.empty():
            new_avg_value = value
        else:
            assert avg_value is not None
            if not queue.full():
                new_avg_value = (avg_value * queue_size + value) / (queue_size + 1)
            else:
                assert queue_size == self.avg_num_batches
                popped_value = queue.get()
                new_avg_value = avg_value + (value - popped_value) / queue_size

        queue.put(value)
        self.tag_to_avg_value[tag] = new_avg_value
        return new_avg_value
