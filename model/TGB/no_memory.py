from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.nn.inits import zeros

from .time_encoder import TimeEncoder


class TGNNoMemory(torch.nn.Module):
    """TGN-compatible state adapter without recurrent node memory.

    The adapter keeps zero-valued memory vectors so the downstream GNN has the
    same input dimensionality as TGN-ProvIDS. Only the last interaction time is
    updated; no messages are stored and no recurrent state update is applied.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        time_dim: int,
        node_raw_features: Tensor,
        init_time: int = 0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.node_raw_features = node_raw_features
        self.time_enc = TimeEncoder(time_dim)

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer(
            "last_update",
            torch.full((num_nodes,), init_time, dtype=torch.long),
        )

    @property
    def device(self) -> torch.device:
        return self.memory.device

    def reset_state(self) -> None:
        zeros(self.memory)
        zeros(self.last_update)

    def detach(self) -> None:
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        return self.memory[n_id], self.last_update[n_id]

    @torch.no_grad()
    def update_state(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
    ) -> None:
        del raw_msg
        n_id = torch.cat([src, dst]).unique()
        if n_id.numel() > 0:
            self.last_update[n_id] = t.max()
