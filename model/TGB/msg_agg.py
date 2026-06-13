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
    def select_next(self, msg: Tensor, index: Tensor, t: Tensor):
        if msg.numel() == 0:
            empty_index = index.new_empty((0,))
            empty_msg = msg.new_empty((0, msg.size(-1)))
            empty_t = t.new_empty((0,))
            return empty_index, empty_msg, empty_t, empty_msg, empty_index, empty_t

        selected = []
        for node_idx in index.unique(sorted=True).tolist():
            msg_idx = (index == node_idx).nonzero(as_tuple=False).view(-1)
            selected.append(msg_idx[torch.argmin(t[msg_idx])])

        selected = torch.stack(selected)
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
