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
        self.num_write_steps = int(match.group(2))
        self.K = 0
        self.cycle_len = self.num_write_steps + self.lag + 2
        self.query_offset = self.cycle_len - 2
        self.target_offset = self.cycle_len - 1

    def get_dataset_regex_pattern(self) -> Tuple[str, str]:
        return AssociativeRecall.REGEX, self.args.data.split("/")[0]

    def is_query_snapshot(self, snapshot_t: int) -> bool:
        return int(snapshot_t) % self.cycle_len == self.query_offset

    def is_target_snapshot(self, snapshot_t: int) -> bool:
        return int(snapshot_t) % self.cycle_len == self.target_offset

    def should_train_snapshot(self, snapshot_t: int) -> bool:
        return not self.is_query_snapshot(snapshot_t)

    def get_training_neg_link(
        self,
        pos_src: torch.Tensor,
        pos_dst: torch.Tensor,
        pos_t: torch.Tensor,
        snapshot_t: int,
    ):
        if not self.is_target_snapshot(snapshot_t):
            return super().get_training_neg_link(pos_src, pos_dst, pos_t, snapshot_t)

        num_nodes = self.train_loader.dataset.num_nodes
        node_ids = torch.arange(1, num_nodes, dtype=torch.long, device=self.device)
        cand_src = torch.cat([
            torch.zeros_like(node_ids),
            node_ids,
        ])
        cand_dst = torch.cat([
            node_ids,
            torch.zeros_like(node_ids),
        ])

        pos_pairs = set((int(src), int(dst)) for src, dst in zip(pos_src.tolist(), pos_dst.tolist()))
        keep = torch.tensor(
            [(int(src), int(dst)) not in pos_pairs for src, dst in zip(cand_src.tolist(), cand_dst.tolist())],
            dtype=torch.bool,
            device=self.device,
        )
        return cand_src[keep], cand_dst[keep]

    def compute_snapshot_training_loss(
        self,
        pos_src: torch.Tensor,
        pos_dst: torch.Tensor,
        neg_src: torch.Tensor,
        neg_dst: torch.Tensor,
        pos_pred: torch.Tensor,
        neg_pred: torch.Tensor,
        snapshot_t: int,
    ):
        if not self.is_target_snapshot(snapshot_t):
            return super().compute_snapshot_training_loss(
                pos_src,
                pos_dst,
                neg_src,
                neg_dst,
                pos_pred,
                neg_pred,
                snapshot_t,
            )

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_pred),
            torch.zeros_like(neg_pred),
        ], dim=0)
        loss = self.criterion(pred, labels)
        nan_loss = loss.new_tensor(float("nan"))
        return loss, loss, nan_loss

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
