import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import TransformerConv

from ..node_emb import NodeEmbeddingModel


class IdentityMessageProvIDS(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, edge_encoder=None):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim
        self.edge_encoder = edge_encoder

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        if self.edge_encoder is not None:
            raw_msg = self.edge_encoder(raw_msg)
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class GraphAttentionEmbeddingProvIDS(NodeEmbeddingModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        msg_dim,
        time_enc,
        mean_delta_t=0.,
        std_delta_t=1.,
        encode_edge=False,
        edge_encoder=None,
    ):
        super().__init__()
        self.time_enc = time_enc
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.edge_encoder = edge_encoder if edge_encoder is not None else (
            torch.nn.Linear(msg_dim, msg_dim) if encode_edge else None
        )
        edge_dim = msg_dim + time_enc.out_channels
        num_heads = 2
        self.out_channels = out_channels + (out_channels % num_heads)

        self.conv = TransformerConv(
            in_channels,
            self.out_channels // num_heads,
            heads=num_heads,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    @property
    def out_dimension(self) -> int:
        return self.out_channels

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = t - last_update[edge_index[0]]
        rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        if self.edge_encoder is not None:
            msg = self.edge_encoder(msg)
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


def compute_provids_delta_t_stats(train_dataset, init_time):
    last_timestamp = {}
    all_timediffs = []

    for src, dst, t in zip(train_dataset.src, train_dataset.dst, train_dataset.t):
        src = int(src.item())
        dst = int(dst.item())
        t = t.item()

        all_timediffs.append(t - last_timestamp.get(src, init_time))
        all_timediffs.append(t - last_timestamp.get(dst, init_time))

        last_timestamp[src] = t
        last_timestamp[dst] = t

    all_timediffs = np.asarray(all_timediffs)
    if all_timediffs.size == 0:
        raise ValueError("Cannot compute delta_t stats from an empty train split.")
    mean_delta_t = float(np.mean(all_timediffs))
    std_delta_t = float(np.std(all_timediffs))
    if std_delta_t == 0:
        std_delta_t = 1.0

    print(f"avg delta_t(all): {mean_delta_t} +/- {std_delta_t}", flush=True)
    return mean_delta_t, std_delta_t
