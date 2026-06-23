import re
from typing import Any, Dict, List, Tuple

import torch

from ....CTDG.link_pred.memory_node.trainer import MemoryNodeTrainer
from .....dataset.DTDG.graph_generation.associative_recall import AssociativeRecall


class AssociativeRecallTrainer(MemoryNodeTrainer):
    def __init__(self):
        super().__init__()
        data_pattern = self.args.data.split("/")[0]
        match = re.fullmatch(AssociativeRecall.REGEX, data_pattern)
        if not match:
            raise ValueError(f"Invalid associative_recall data pattern: {data_pattern}")
        self.lag = int(match.group(1))
        self.num_pairs = int(match.group(2))
        self.K = 0
        self.cycle_len = self.num_pairs + self.lag + 2
        self.target_offset = self.cycle_len - 1

    def get_dataset_regex_pattern(self) -> Tuple[str, str]:
        return AssociativeRecall.REGEX, self.args.data.split("/")[0]

    def update_metrics(
        self,
        curr_snapshot: torch.Tensor,
        snapshot_t: int,
        snapshot_idx: int,
        metrics_list: Dict[str, List[Any]],
        split_mode: str,
        dataset,
    ):
        if int(snapshot_t) % self.cycle_len != self.target_offset:
            self._advance_state_with_full_snapshot(curr_snapshot, snapshot_t, dataset)
            return

        super().update_metrics(
            curr_snapshot,
            snapshot_t,
            snapshot_idx,
            metrics_list,
            split_mode,
            dataset,
        )
