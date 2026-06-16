from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import scatter

from .provids import (
    GraphAttentionEmbeddingProvIDS,
    IdentityMessageProvIDS,
    compute_provids_delta_t_stats,
)
from .msg_agg import SequentialAggregator
from .tgn_memory import TGNMemory
from .mlstm import MLSTMStateDictType, mLSTMMemoryAdapter


class TGNMemoryProvIDSMLSTM(TGNMemory):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        node_raw_features: torch.Tensor,
        message_module,
        aggregator_module,
        mlstm_num_heads: int = 4,
        mlstm_state_max_nodes: Optional[int] = None,
        mlstm_state_storage_dtype: Optional[torch.dtype] = None,
    ):
        self.mlstm_num_heads = mlstm_num_heads
        self.mlstm_state_max_nodes = mlstm_state_max_nodes
        self.mlstm_state_storage_dtype = mlstm_state_storage_dtype
        self._mlstm_state_dict = {}
        self._pending_mlstm_node_ids = set()
        self._last_updated_n_id = None
        self._store_mode = "base"

        super().__init__(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            node_raw_features=node_raw_features,
            message_module=message_module,
            aggregator_module=aggregator_module,
            memory_updater_cell="gru",
        )

        self.memory_updater = mLSTMMemoryAdapter(
            message_dim=message_module.out_channels,
            memory_dim=memory_dim,
            num_heads=mlstm_num_heads,
            context_length=64,
            conv1d_kernel_size=4,
        )
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        self._mlstm_state_dict = {}
        self._pending_mlstm_node_ids = set()
        self._last_updated_n_id = None
        self._store_mode = "base"
        self._reset_message_store_z()

    def _reset_message_store_z(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        z = self.memory.new_empty((0, self.memory_dim), device=self.device)
        self.msg_s_store_z = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}
        self.msg_d_store_z = {j: (i, i, i, msg, z, z) for j in range(self.num_nodes)}

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

    def _empty_mlstm_state(self, batch_size: int) -> MLSTMStateDictType:
        mlstm_config = self.memory_updater.mlstm_layer.config
        num_heads = mlstm_config.num_heads
        inner_dim = mlstm_config._inner_embedding_dim
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"mLSTM inner embedding dim={inner_dim} must be divisible "
                f"by mlstm_num_heads={num_heads}."
            )

        head_dim = inner_dim // num_heads
        device = self.memory.device
        dtype = self.memory.dtype
        mlstm_state = (
            torch.zeros(batch_size, num_heads, head_dim, head_dim, device=device, dtype=dtype),
            torch.zeros(batch_size, num_heads, head_dim, 1, device=device, dtype=dtype),
            torch.zeros(batch_size, num_heads, 1, 1, device=device, dtype=dtype),
        )

        conv_state = None
        kernel_size = self.memory_updater.mlstm_layer.conv1d.config.kernel_size
        if kernel_size > 0:
            conv_state = (
                torch.zeros(batch_size, kernel_size, inner_dim, device=device, dtype=dtype),
            )

        return {"mlstm_state": mlstm_state, "conv_state": conv_state}

    def _get_mlstm_state(self, n_id: Tensor) -> MLSTMStateDictType:
        device = self.memory.device
        dtype = self.memory.dtype
        state = self._empty_mlstm_state(n_id.size(0))
        c_state, n_state, m_state = state["mlstm_state"]
        conv_state = state["conv_state"]

        for i, node in enumerate(n_id.tolist()):
            stored = self._mlstm_state_dict.get(node)
            if stored is None:
                continue

            c_i, n_i, m_i = stored["mlstm_state"]
            c_state[i] = c_i.to(device=device, dtype=dtype)
            n_state[i] = n_i.to(device=device, dtype=dtype)
            m_state[i] = m_i.to(device=device, dtype=dtype)

            stored_conv_state = stored.get("conv_state")
            if conv_state is not None and stored_conv_state is not None:
                conv_state[0][i] = stored_conv_state[0].to(device=device, dtype=dtype)

        return state

    def _store_mlstm_tensor_cpu(self, tensor: Tensor) -> Tensor:
        tensor = tensor.detach().cpu().clone()
        if self.mlstm_state_storage_dtype is not None:
            tensor = tensor.to(dtype=self.mlstm_state_storage_dtype)
        return tensor

    def _set_mlstm_state(self, n_id: Tensor, state: MLSTMStateDictType):
        new_nodes = set(n_id.tolist()).difference(self._mlstm_state_dict.keys())
        if (
            self.mlstm_state_max_nodes is not None
            and len(self._mlstm_state_dict) + len(new_nodes) > self.mlstm_state_max_nodes
        ):
            raise RuntimeError(
                f"mLSTM state cache would exceed mlstm_state_max_nodes="
                f"{self.mlstm_state_max_nodes}."
            )

        c_state, n_state, m_state = state["mlstm_state"]
        conv_state = state.get("conv_state")
        for i, node in enumerate(n_id.tolist()):
            node_conv_state = None
            if conv_state is not None:
                node_conv_state = (self._store_mlstm_tensor_cpu(conv_state[0][i]),)

            self._mlstm_state_dict[node] = {
                "mlstm_state": (
                    self._store_mlstm_tensor_cpu(c_state[i]),
                    self._store_mlstm_tensor_cpu(n_state[i]),
                    self._store_mlstm_tensor_cpu(m_state[i]),
                ),
                "conv_state": node_conv_state,
            }

    def _add_pending_mlstm_nodes(self, n_id: Tensor):
        self._pending_mlstm_node_ids.update(int(i) for i in n_id.tolist())

    def _discard_pending_mlstm_nodes(self, n_id: Tensor):
        self._pending_mlstm_node_ids.difference_update(int(i) for i in n_id.tolist())

    def _pending_mlstm_nodes_tensor(self) -> Tensor:
        if len(self._pending_mlstm_node_ids) == 0:
            return torch.empty(0, dtype=torch.long, device=self.memory.device)
        return torch.tensor(
            sorted(self._pending_mlstm_node_ids),
            dtype=torch.long,
            device=self.memory.device,
        )

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id, commit_mlstm_state=True)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor, commit_mlstm_state: bool = False):
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
            state = self._get_mlstm_state(n_id)
            local_idx = self._assoc[idx]

            for active_nodes, active_msg, active_t in self.aggr_module.iter_by_timestamp(
                msg, local_idx, t
            ):
                memory_new, state_new = self.memory_updater(
                    active_msg,
                    memory[active_nodes],
                    self._index_mlstm_state(state, active_nodes),
                )
                memory = memory.index_copy(0, active_nodes, memory_new)
                state = self._index_copy_mlstm_state(state, active_nodes, state_new)
                last_update[active_nodes] = active_t

            if commit_mlstm_state:
                self._set_mlstm_state(n_id, state)

            return memory, last_update

        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        state = self._get_mlstm_state(n_id)
        memory, state_new = self.memory_updater(aggr, self.memory[n_id], state)
        if commit_mlstm_state:
            self._set_mlstm_state(n_id, state_new)

        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
        return memory, last_update

    def _index_mlstm_state(self, state: MLSTMStateDictType, idx: Tensor):
        c_state, n_state, m_state = state["mlstm_state"]
        conv_state = state.get("conv_state")
        return {
            "mlstm_state": (
                c_state[idx],
                n_state[idx],
                m_state[idx],
            ),
            "conv_state": None if conv_state is None else (conv_state[0][idx],),
        }

    def _index_copy_mlstm_state(
        self,
        state: MLSTMStateDictType,
        idx: Tensor,
        update: MLSTMStateDictType,
    ) -> MLSTMStateDictType:
        c_state, n_state, m_state = state["mlstm_state"]
        c_new, n_new, m_new = update["mlstm_state"]
        conv_state = state.get("conv_state")
        conv_new = update.get("conv_state")

        conv_state_new = None
        if conv_state is not None:
            conv_state_new = conv_state
            if conv_new is not None:
                conv_state_new = (conv_state[0].index_copy(0, idx, conv_new[0]),)

        return {
            "mlstm_state": (
                c_state.index_copy(0, idx, c_new),
                n_state.index_copy(0, idx, n_new),
                m_state.index_copy(0, idx, m_new),
            ),
            "conv_state": conv_state_new,
        }

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        self._store_mode = "base"
        self._last_updated_n_id = torch.cat([src, dst]).unique()
        n_id = self._last_updated_n_id

        if self.training:
            self._update_memory(n_id)
            self._discard_pending_mlstm_nodes(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._add_pending_mlstm_nodes(n_id)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

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
            self._discard_pending_mlstm_nodes(n_id)
            self._update_msg_store_z(src, dst, t, raw_msg, z_src, z_dst, self.msg_s_store_z)
            self._update_msg_store_z(dst, src, t, raw_msg, z_dst, z_src, self.msg_d_store_z)
            self._add_pending_mlstm_nodes(n_id)
        else:
            self._update_msg_store_z(src, dst, t, raw_msg, z_src, z_dst, self.msg_s_store_z)
            self._update_msg_store_z(dst, src, t, raw_msg, z_dst, z_src, self.msg_d_store_z)
            self._update_memory(n_id)

    def train(self, mode: bool = True):
        if self.training and not mode:
            n_id = self._pending_mlstm_nodes_tensor()
            flush_batch = 32
            for i in range(0, n_id.size(0), flush_batch):
                self._update_memory(n_id[i:i + flush_batch])
            self._pending_mlstm_node_ids.clear()
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


__all__ = [
    "GraphAttentionEmbeddingProvIDS",
    "IdentityMessageProvIDS",
    "TGNMemoryProvIDSMLSTM",
    "compute_provids_delta_t_stats",
]
