from typing import Tuple

from ....CTDG.link_pred.memory_node.trainer import MemoryNodeTrainer
from .....dataset.DTDG.graph_generation.ordered_long_range import OrderedLongRange


class OrderedLongRangeTrainer(MemoryNodeTrainer):
    def get_dataset_regex_pattern(self) -> Tuple[str, str]:
        return OrderedLongRange.REGEX, self.args.data.split("/")[0]
