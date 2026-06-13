import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import scatter

from ..node_emb import NodeEmbeddingModel
from .msg_agg import SequentialAggregator
from .tgn_memory import TGNMemory


class TGNMemoryProvIDS(TGNMemory):
    def __init__(self, *args, **kwargs):
        self._last_updated_n_id = None
        super().__init__(*args, **kwargs)
        self._store_mode = "base"
        self._reset_message_store_z()

    def _reset_message_store_z(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        z = self.memory.new_empty((0, self.memory_dim), device=self.device)
        self.msg_s_store_z = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}
        self.msg_d_store_z = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}

    def reset_state(self):
        super().reset_state()
        self._store_mode = "base"
        self._last_updated_n_id = None
        self._reset_message_store_z()

    def _update_msg_store_z(self, src: Tensor, dst: Tensor, t: Tensor,
                            raw_msg: Tensor, z_src: Tensor, z_dst: Tensor,
                            msg_store):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx], z_src[idx], z_dst[idx])

    def _compute_msg_z(self, n_id: Tensor, msg_store, msg_module):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg, z_src, z_dst = list(zip(*data))

        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        z_src = torch.cat(z_src, dim=0)
        z_dst = torch.cat(z_dst, dim=0)

        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = msg_module(z_src, z_dst, raw_msg, t_enc)
        return msg, t, src, dst

    def _get_updated_memory(self, n_id: Tensor):
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        if self._store_mode == "z":
            msg_s, t_s, src_s, dst_s = self._compute_msg_z(n_id, self.msg_s_store_z, self.msg_s_module)
            msg_d, t_d, src_d, dst_d = self._compute_msg_z(n_id, self.msg_d_store_z, self.msg_d_module)
        else:
            msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store, self.msg_s_module)
            msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store, self.msg_d_module)

        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)

        if isinstance(self.aggr_module, SequentialAggregator):
            memory = self.memory[n_id]
            last_update = self.last_update[n_id].clone()
            local_idx = self._assoc[idx]

            while msg.numel() > 0:
                active_nodes, active_msg, active_t, msg, local_idx, t = self.aggr_module.select_next(
                    msg, local_idx, t
                )
                if active_nodes.numel() == 0:
                    break
                memory = memory.index_copy(
                    0,
                    active_nodes,
                    self.memory_updater(active_msg, memory[active_nodes]),
                )
                last_update[active_nodes] = active_t

            return memory, last_update

        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        memory = self.memory_updater(aggr, self.memory[n_id])

        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
        return memory, last_update

    def update_state_with_z(self, src: Tensor, dst: Tensor, t: Tensor,
                            raw_msg: Tensor, z_src: Tensor, z_dst: Tensor):
        if z_src.size(-1) != self.memory_dim or z_dst.size(-1) != self.memory_dim:
            raise ValueError(
                f"update_state_with_z expected z dim={self.memory_dim}; "
                f"got z_src={z_src.size(-1)} and z_dst={z_dst.size(-1)}."
            )

        self._store_mode = "z"
        self._last_updated_n_id = torch.cat([src, dst]).unique()
        n_id = self._last_updated_n_id

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store_z(src, dst, t, raw_msg, z_src, z_dst, self.msg_s_store_z)
            self._update_msg_store_z(dst, src, t, raw_msg, z_dst, z_src, self.msg_d_store_z)
        else:
            self._update_msg_store_z(src, dst, t, raw_msg, z_src, z_dst, self.msg_s_store_z)
            self._update_msg_store_z(dst, src, t, raw_msg, z_dst, z_src, self.msg_d_store_z)
            self._update_memory(n_id)

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        self._store_mode = "base"
        self._last_updated_n_id = torch.cat([src, dst]).unique()
        return super().update_state(src, dst, t, raw_msg)

    def train(self, mode: bool = True):
        if self.training and not mode:
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
            self._reset_message_store_z()
        super(TGNMemory, self).train(mode)

    def detach(self):
        super().detach()
        n_id = self._last_updated_n_id
        if n_id is None:
            return

        for store_name in ("msg_s_store", "msg_d_store"):
            store = getattr(self, store_name, None)
            if store is None:
                continue
            for i in n_id.tolist():
                src_i, dst_i, t_i, raw_msg_i = store[i]
                store[i] = (src_i, dst_i, t_i, raw_msg_i.detach())

        for store_name in ("msg_s_store_z", "msg_d_store_z"):
            store = getattr(self, store_name, None)
            if store is None:
                continue
            for i in n_id.tolist():
                src_i, dst_i, t_i, raw_msg_i, z_src_i, z_dst_i = store[i]
                store[i] = (
                    src_i,
                    dst_i,
                    t_i,
                    raw_msg_i.detach(),
                    z_src_i.detach(),
                    z_dst_i.detach(),
                )


class IdentityMessageProvIDS(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, edge_encoder=None):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim
        self.edge_encoder = edge_encoder

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        if self.edge_encoder is not None:
            raw_msg = raw_msg.to(dtype=self.edge_encoder.weight.dtype)
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
            msg = msg.to(dtype=self.edge_encoder.weight.dtype)
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
