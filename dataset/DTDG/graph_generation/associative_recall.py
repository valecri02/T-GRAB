import os
import pickle
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from .graph_generator import GraphGenerator, nx_undirected_graph_to_sparse


class AssociativeRecall(GraphGenerator):
    """Delayed query-value recall over ER-derived key/value bindings.

    In each episode, multiple write timesteps produce key/value associations.
    Edges are sampled from ER graphs over active non-memory nodes; each edge
    (u, v) defines key=min(u, v), value=max(u, v). After a lag, the memory node
    is connected to all keys. One timestep later, the target memory-node edges
    are all values associated with those queried keys.
    """

    _pattern = r"^\((\d+), (\d+)\)$"
    REGEX = f"{_pattern}"

    def __init__(self, args) -> None:
        super(AssociativeRecall, self).__init__(
            args.num_nodes,
            args.dataset_name,
            args.seed,
        )

        lag_and_write_steps = self.dataset_name.split("/")[0]
        if re.fullmatch(AssociativeRecall._pattern, lag_and_write_steps):
            match = re.fullmatch(AssociativeRecall._pattern, lag_and_write_steps)
            self.lag = int(match.group(1))
            self.num_write_steps = int(match.group(2))
        else:
            raise NotImplementedError()

        self.args = args
        if self.lag < 0:
            raise ValueError("associative_recall requires lag >= 0.")
        if self.num_write_steps <= 0:
            raise ValueError("associative_recall requires num_write_steps > 0.")
        if self.args.active_nodes < 2:
            raise ValueError("active_nodes must be >= 2.")
        if self.args.active_nodes >= self.num_nodes:
            raise ValueError("active_nodes must be smaller than num_nodes because node 0 is memory.")
        if self.args.pairs_per_step <= 0:
            raise ValueError("pairs_per_step must be > 0.")
        if self.args.num_distractor_edges < 0:
            raise ValueError("num_distractor_edges must be >= 0.")

        self.dataset_name = (
            self.dataset_name
            + f"/associative_recall-{args.num_samples}ns-{args.num_nodes}nn-"
            + f"{args.active_nodes}an-{args.pairs_per_step}pps-"
            + f"{args.num_distractor_edges}nd-{args.val_ratio}vr-{args.test_ratio}tr"
        )

        self.cycle_len = self.num_write_steps + self.lag + 2
        self.query_offset = self.cycle_len - 2
        self.target_offset = self.cycle_len - 1
        self.T = self.args.num_samples * self.cycle_len

        self.node_datatype = np.int64
        if self.T < 2**8:
            self.t_datatype = np.int16
        elif self.T < 2**16:
            self.t_datatype = np.int32
        else:
            self.t_datatype = np.int64

        self.edge_feat_datatype = np.int16
        self.edge_feat_value = 1
        self.memory_node = 0
        self.non_memory_nodes = np.arange(1, self.num_nodes, dtype=self.node_datatype)
        self.targets: Dict[int, Dict[str, object]] = {}

        self.test_num_cycles = int(self.args.num_samples * self.args.test_ratio)
        self.val_num_cycles = int(self.args.num_samples * self.args.val_ratio)
        self.train_num_cycles = (
            self.args.num_samples - self.test_num_cycles - self.val_num_cycles
        )
        if self.train_num_cycles <= 0:
            raise ValueError("Train split has no complete associative-recall cycles.")

        self.train_start_t = 0
        self.val_start_t = self.train_num_cycles * self.cycle_len
        self.test_start_t = (self.train_num_cycles + self.val_num_cycles) * self.cycle_len
        assert self.T == (
            self.train_num_cycles
            + self.val_num_cycles
            + self.test_num_cycles
        ) * self.cycle_len

    @staticmethod
    def get_parser():
        parser = GraphGenerator.get_parser()
        parser.add_argument("--val-ratio", type=float, required=True)
        parser.add_argument("--test-ratio", type=float, required=True)
        parser.add_argument("--visualize", action="store_true")
        parser.add_argument("--num-samples", type=int, required=True)
        parser.add_argument("--active-nodes", type=int, required=True)
        parser.add_argument("--pairs-per-step", type=int, required=True)
        parser.add_argument("--num-distractor-edges", type=int, required=True)
        # Backwards-compatible ignored arguments from the first version.
        parser.add_argument("--num-keys", type=int, default=None)
        parser.add_argument("--num-values", type=int, default=None)
        return parser

    def get_val_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.logical_and(t >= self.val_start_t, t < self.test_start_t)

    def get_test_mask(self, t: np.ndarray) -> torch.Tensor:
        return t >= self.test_start_t

    def get_test_inductive_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.zeros_like(t, dtype=bool)

    def _append_graph(
        self,
        G: nx.Graph,
        now_t: int,
        src: np.ndarray,
        dst: np.ndarray,
        t: np.ndarray,
        edge_feat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if G.number_of_edges() == 0:
            anchor = int(self.non_memory_nodes[0])
            G.add_edge(anchor, anchor, weight=self.edge_feat_value)

        src_t, dst_t, edge_feat_t = nx_undirected_graph_to_sparse(G, return_edge_feat=True)
        src_t = src_t.cpu().numpy().astype(self.node_datatype)
        dst_t = dst_t.cpu().numpy().astype(self.node_datatype)
        edge_feat_t = edge_feat_t.cpu().numpy().astype(self.edge_feat_datatype)
        t_t = np.full_like(src_t, fill_value=now_t).astype(self.t_datatype)

        return (
            np.concatenate([src, src_t]),
            np.concatenate([dst, dst_t]),
            np.concatenate([t, t_t]),
            np.concatenate([edge_feat, edge_feat_t]),
        )

    def _sample_unique_edges(
        self,
        candidates: np.ndarray,
        num_edges: int,
    ) -> List[Tuple[int, int]]:
        max_unique_edges = len(candidates) * (len(candidates) - 1) // 2
        if num_edges > max_unique_edges:
            raise ValueError(
                f"Requested {num_edges} unique edges from {len(candidates)} nodes, "
                f"but only {max_unique_edges} are available."
            )

        edges = set()
        while len(edges) < num_edges:
            u, v = np.random.choice(candidates, size=2, replace=False)
            edges.add(tuple(sorted((int(u), int(v)))))
        return list(edges)

    def _sample_write_edges(self) -> List[Tuple[int, int]]:
        active = np.random.choice(
            self.non_memory_nodes,
            size=self.args.active_nodes,
            replace=False,
        )
        return self._sample_unique_edges(active, self.args.pairs_per_step)

    def _sample_distractor_edges(self) -> List[Tuple[int, int]]:
        if self.args.num_distractor_edges == 0:
            return []
        return self._sample_unique_edges(self.non_memory_nodes, self.args.num_distractor_edges)

    @staticmethod
    def _add_binding(
        bindings: DefaultDict[int, Set[int]],
        edge: Tuple[int, int],
    ) -> Tuple[int, int]:
        key, value = min(edge), max(edge)
        bindings[key].add(value)
        return key, value

    def get_links(self, args_dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        src = np.empty(0, dtype=self.node_datatype)
        dst = np.empty_like(src)
        t = np.empty_like(src, dtype=self.t_datatype)
        edge_feat = np.empty_like(src, dtype=self.edge_feat_datatype)

        if self.args.visualize:
            pos = nx.circular_layout(nx.complete_graph(self.num_nodes))

        with tqdm(total=self.T, desc=self.dataset_name) as pbar:
            for sample_idx in range(self.args.num_samples):
                cycle_start_t = sample_idx * self.cycle_len
                bindings: DefaultDict[int, Set[int]] = defaultdict(set)
                write_edges_by_t = {}

                for write_idx in range(self.num_write_steps):
                    now_t = cycle_start_t + write_idx
                    G = nx.empty_graph(self.num_nodes)
                    write_edges = self._sample_write_edges()
                    write_edges_by_t[now_t] = []

                    for edge in write_edges:
                        key, value = self._add_binding(bindings, edge)
                        write_edges_by_t[now_t].append((key, value))
                        G.add_edge(key, value, weight=self.edge_feat_value)

                    src, dst, t, edge_feat = self._append_graph(G, now_t, src, dst, t, edge_feat)
                    pbar.update(1)

                for lag_idx in range(self.lag):
                    now_t = cycle_start_t + self.num_write_steps + lag_idx
                    G = nx.empty_graph(self.num_nodes)
                    for u, v in self._sample_distractor_edges():
                        G.add_edge(u, v, weight=self.edge_feat_value)
                    src, dst, t, edge_feat = self._append_graph(G, now_t, src, dst, t, edge_feat)
                    pbar.update(1)

                query_t = cycle_start_t + self.query_offset
                query_keys = sorted(bindings.keys())
                G = nx.empty_graph(self.num_nodes)
                for key in query_keys:
                    G.add_edge(self.memory_node, key, weight=self.edge_feat_value)
                src, dst, t, edge_feat = self._append_graph(G, query_t, src, dst, t, edge_feat)
                pbar.update(1)

                target_t = cycle_start_t + self.target_offset
                target_values = sorted(set().union(*bindings.values()))
                G = nx.empty_graph(self.num_nodes)
                for value in target_values:
                    G.add_edge(self.memory_node, value, weight=self.edge_feat_value)

                self.targets[target_t] = {
                    "query_keys": query_keys,
                    "target_values": target_values,
                    "bindings": {int(k): sorted(int(v) for v in values) for k, values in bindings.items()},
                    "write_edges_by_t": write_edges_by_t,
                }
                src, dst, t, edge_feat = self._append_graph(G, target_t, src, dst, t, edge_feat)

                if self.args.visualize and sample_idx < 4:
                    vis_save_dir = os.path.join(args_dict["save_dir"], self.dataset_name, "vis")
                    os.makedirs(vis_save_dir, exist_ok=True)
                    plt.figure(figsize=(20, 10))
                    nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
                    edge_labels = nx.get_edge_attributes(G, "weight")
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                    plt.title(f"Associative recall target timestep {target_t}")
                    plt.savefig(os.path.join(vis_save_dir, f"{target_t}.png"))
                    plt.close()

                pbar.update(1)

        assert t.max() == self.T - 1, f"Last timestep should be {self.T - 1}. Got {t.max()}."
        return src, dst, t, edge_feat

    def create_data(self, args_dict) -> None:
        super().create_data(args_dict)
        fdir = os.path.join(args_dict["save_dir"], self.dataset_name)
        with open(os.path.join(fdir, "associative_targets.pkl"), "wb") as f:
            pickle.dump(
                {
                    "edge_feat_value": self.edge_feat_value,
                    "cycle_len": self.cycle_len,
                    "query_offset": self.query_offset,
                    "target_offset": self.target_offset,
                    "lag": self.lag,
                    "num_write_steps": self.num_write_steps,
                    "active_nodes": self.args.active_nodes,
                    "pairs_per_step": self.args.pairs_per_step,
                    "targets": self.targets,
                },
                f,
            )
