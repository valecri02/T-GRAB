"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch_geometric.utils import scatter
from torch_scatter import scatter_max


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")


class SequentialAggregator(torch.nn.Module):
    @staticmethod
    def _empty_selection(msg: Tensor, index: Tensor, t: Tensor):
        empty_index = index.new_empty((0,))
        empty_msg = msg.new_empty((0, msg.size(-1)))
        empty_t = t.new_empty((0,))
        return empty_index, empty_msg, empty_t

    @staticmethod
    def _order_by_node_time(index: Tensor, t: Tensor) -> Tensor:
        order = torch.arange(index.size(0), device=index.device)
        order = order[torch.argsort(t[order], stable=True)]
        order = order[torch.argsort(index[order], stable=True)]
        return order

    def select_next(self, msg: Tensor, index: Tensor, t: Tensor):
        if msg.numel() == 0:
            empty_index, empty_msg, empty_t = self._empty_selection(msg, index, t)
            return empty_index, empty_msg, empty_t, empty_msg, empty_index, empty_t

        order = self._order_by_node_time(index, t)
        sorted_index = index[order]
        _, counts = sorted_index.unique_consecutive(return_counts=True)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        selected = order[offsets]
        keep = torch.ones(msg.size(0), dtype=torch.bool, device=msg.device)
        keep[selected] = False

        return (
            index[selected],
            msg[selected],
            t[selected],
            msg[keep],
            index[keep],
            t[keep],
        )

    def iter_by_timestamp(self, msg: Tensor, index: Tensor, t: Tensor):
        if msg.numel() == 0:
            return

        order = self._order_by_node_time(index, t)
        sorted_index = index[order]
        active_nodes, counts = sorted_index.unique_consecutive(return_counts=True)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])

        for rank in range(int(counts.max().item())):
            active_mask = counts > rank
            selected = order[offsets[active_mask] + rank]
            yield active_nodes[active_mask], msg[selected], t[selected]
