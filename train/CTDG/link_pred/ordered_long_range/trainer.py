from typing import Tuple

import torch

from ....CTDG.link_pred.memory_node.trainer import MemoryNodeTrainer
from .....dataset.DTDG.graph_generation.ordered_long_range import OrderedLongRange


class OrderedLongRangeTrainer(MemoryNodeTrainer):
    def get_dataset_regex_pattern(self) -> Tuple[str, str]:
        return OrderedLongRange.REGEX, self.args.data.split("/")[0]

    @staticmethod
    def _has_memory_node_edges(src: torch.Tensor, dst: torch.Tensor) -> bool:
        return bool(torch.any(torch.logical_or(src == 0, dst == 0)).item())

    def get_training_neg_link(
        self,
        pos_src: torch.Tensor,
        pos_dst: torch.Tensor,
        pos_t: torch.Tensor,
        snapshot_t: int,
    ):
        sampled_neg_src, sampled_neg_dst = super().get_training_neg_link(
            pos_src,
            pos_dst,
            pos_t,
            snapshot_t,
        )

        if not self._has_memory_node_edges(pos_src, pos_dst):
            return sampled_neg_src, sampled_neg_dst

        sampled_non_mem_mask = torch.logical_and(sampled_neg_src != 0, sampled_neg_dst != 0)
        sampled_neg_src = sampled_neg_src[sampled_non_mem_mask]
        sampled_neg_dst = sampled_neg_dst[sampled_non_mem_mask]

        num_nodes = self.train_loader.dataset.num_nodes
        node_ids = torch.arange(1, num_nodes, dtype=torch.long, device=self.device)
        mem_neg_src = torch.cat([
            torch.zeros_like(node_ids),
            node_ids,
        ])
        mem_neg_dst = torch.cat([
            node_ids,
            torch.zeros_like(node_ids),
        ])

        pos_pairs = set((int(src), int(dst)) for src, dst in zip(pos_src.tolist(), pos_dst.tolist()))
        keep = torch.tensor(
            [(int(src), int(dst)) not in pos_pairs for src, dst in zip(mem_neg_src.tolist(), mem_neg_dst.tolist())],
            dtype=torch.bool,
            device=self.device,
        )
        mem_neg_src = mem_neg_src[keep]
        mem_neg_dst = mem_neg_dst[keep]

        return (
            torch.cat([sampled_neg_src, mem_neg_src], dim=0),
            torch.cat([sampled_neg_dst, mem_neg_dst], dim=0),
        )

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
        if not self._has_memory_node_edges(pos_src, pos_dst):
            return super().compute_snapshot_training_loss(
                pos_src,
                pos_dst,
                neg_src,
                neg_dst,
                pos_pred,
                neg_pred,
                snapshot_t,
            )

        pos_memnode_mask = torch.logical_or(pos_src == 0, pos_dst == 0)
        neg_memnode_mask = torch.logical_or(neg_src == 0, neg_dst == 0)
        pos_non_memnode_mask = ~pos_memnode_mask
        neg_non_memnode_mask = ~neg_memnode_mask

        loss_terms = []
        nan_loss = pos_pred.new_tensor(float("nan"))
        loss_memnode = nan_loss
        loss_non_memnode = nan_loss

        if torch.any(pos_memnode_mask) and torch.any(neg_memnode_mask):
            mem_pred = torch.cat([
                pos_pred[pos_memnode_mask],
                neg_pred[neg_memnode_mask],
            ], dim=0)
            mem_labels = torch.cat([
                torch.ones_like(pos_pred[pos_memnode_mask]),
                torch.zeros_like(neg_pred[neg_memnode_mask]),
            ], dim=0)
            loss_memnode = self.criterion(mem_pred, mem_labels)
            loss_terms.append(loss_memnode)

        if torch.any(pos_non_memnode_mask) and torch.any(neg_non_memnode_mask):
            non_mem_pred = torch.cat([
                pos_pred[pos_non_memnode_mask],
                neg_pred[neg_non_memnode_mask],
            ], dim=0)
            non_mem_labels = torch.cat([
                torch.ones_like(pos_pred[pos_non_memnode_mask]),
                torch.zeros_like(neg_pred[neg_non_memnode_mask]),
            ], dim=0)
            loss_non_memnode = self.criterion(non_mem_pred, non_mem_labels)
            loss_terms.append(loss_non_memnode)

        if len(loss_terms) == 0:
            return None, loss_memnode, loss_non_memnode

        return sum(loss_terms), loss_memnode, loss_non_memnode
