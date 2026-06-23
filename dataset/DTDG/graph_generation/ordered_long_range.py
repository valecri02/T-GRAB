import os
import pickle
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from .graph_generator import GraphGenerator, nx_undirected_graph_to_sparse


class OrderedLongRange(GraphGenerator):
    """Sequential-path variant of the long-range memory-node task.

    This keeps the original long-range endpoint-retrieval target, but reveals
    each branch path one edge per timestamp. After the branch has been fully
    revealed, node 0 connects to all branch endpoints after an optional lag.
    """

    _pattern = r"^\((\d+), (\d+)\)$"
    REGEX = f"{_pattern}"

    def __init__(self, args) -> None:
        super(OrderedLongRange, self).__init__(
            args.num_nodes,
            args.dataset_name,
            args.seed,
        )

        lag_and_branch_len = self.dataset_name.split("/")[0]
        if re.fullmatch(OrderedLongRange._pattern, lag_and_branch_len):
            match = re.fullmatch(OrderedLongRange._pattern, lag_and_branch_len)
            self.lag = int(match.group(1))
            self.branch_len = int(match.group(2))
        else:
            raise NotImplementedError()

        self.args = args
        if self.branch_len < 2:
            raise ValueError("ordered_long_range requires branch_len >= 2.")

        max_required_nodes = 2 + self.args.num_branches * self.branch_len
        if max_required_nodes > self.num_nodes:
            raise ValueError(
                f"num_nodes={self.num_nodes} is too small for "
                f"num_branches={self.args.num_branches}, branch_len={self.branch_len}. "
                f"Need at least {max_required_nodes} nodes."
            )

        self.dataset_name = (
            self.dataset_name
            + f"/ordered_long_range-{args.num_samples}ns-{args.num_nodes}nn-"
            + f"{args.num_branches}nb-{args.val_ratio}vr-{args.test_ratio}tr"
        )
        self.cycle_len = self.branch_len + self.lag + 1
        self.T = self.args.num_samples * self.cycle_len

        self.node_datatype = np.int64
        if self.T < 2**8:
            self.t_datatype = np.int16
        elif self.T < 2**16:
            self.t_datatype = np.int32
        else:
            self.t_datatype = np.int64

        self.edge_feat_datatype = np.int16

        self.effect_node = 0
        self.cause_node = 1
        self.edge_feat_value = 1
        self.sequential_targets: Dict[int, Dict[str, List[int]]] = {}

        self.test_num_cycles = int(self.args.num_samples * self.args.test_ratio)
        self.val_num_cycles = int(self.args.num_samples * self.args.val_ratio)
        self.train_num_cycles = (
            self.args.num_samples - self.test_num_cycles - self.val_num_cycles
        )
        if self.train_num_cycles <= 0:
            raise ValueError("Train split has no complete ordered-long-range cycles.")

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
        parser.add_argument("--num-branches", type=int, required=True)
        # Kept for backwards compatibility with older local scripts. Ignored.
        parser.add_argument("--num-symbols", type=int, default=None)
        parser.add_argument("--positive-ratio", type=float, default=None)
        return parser

    def get_val_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.logical_and(t >= self.val_start_t, t < self.test_start_t)

    def get_test_mask(self, t: np.ndarray) -> torch.Tensor:
        return t >= self.test_start_t

    def get_test_inductive_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.zeros_like(t, dtype=bool)

    def _create_branch_paths(self):
        paths = []
        current_node = self.cause_node + 1
        for _ in range(self.args.num_branches):
            branch = [self.cause_node]
            for _ in range(self.branch_len):
                branch.append(current_node)
                current_node += 1
            paths.append(branch)
        return paths

    def _reorder_nodes_wo_cause_effect(self):
        old_indices = np.arange(self.num_nodes)
        movable = old_indices[2:]
        shuffled = np.random.permutation(movable)
        return np.concatenate(([self.effect_node, self.cause_node], shuffled))

    def get_links(self, args_dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        src = np.empty(0, dtype=self.node_datatype)
        dst = np.empty_like(src)
        t = np.empty_like(src, dtype=self.t_datatype)
        edge_feat = np.empty_like(src, dtype=self.edge_feat_datatype)

        def _append_graph(G: nx.Graph, now_t: int):
            nonlocal src, dst, t, edge_feat
            if G.number_of_edges() == 0:
                G.add_edge(self.effect_node, self.effect_node, weight=self.edge_feat_value)
            src_t, dst_t, edge_feat_t = nx_undirected_graph_to_sparse(G, return_edge_feat=True)
            src_t = src_t.cpu().numpy().astype(self.node_datatype)
            dst_t = dst_t.cpu().numpy().astype(self.node_datatype)
            edge_feat_t = edge_feat_t.cpu().numpy().astype(self.edge_feat_datatype)
            src = np.concatenate([src, src_t])
            dst = np.concatenate([dst, dst_t])
            t = np.concatenate([t, np.full_like(src_t, fill_value=now_t).astype(self.t_datatype)])
            edge_feat = np.concatenate([edge_feat, edge_feat_t])

        paths = self._create_branch_paths()

        if self.args.visualize:
            pos = nx.circular_layout(nx.complete_graph(self.num_nodes))

        with tqdm(total=self.T, desc=self.dataset_name) as pbar:
            for sample_idx in range(self.args.num_samples):
                cycle_start_t = sample_idx * self.cycle_len
                new_node_ids = self._reorder_nodes_wo_cause_effect()
                mapped_branches = []
                endpoints = []

                for branch in paths:
                    mapped_branch = [int(new_node_ids[node]) for node in branch]
                    endpoints.append(mapped_branch[-1])
                    mapped_branches.append(mapped_branch)

                for step in range(self.branch_len):
                    now_t = cycle_start_t + step
                    G = nx.empty_graph(self.num_nodes)
                    for mapped_branch in mapped_branches:
                        u, v = mapped_branch[step], mapped_branch[step + 1]
                        G.add_edge(u, v, weight=self.edge_feat_value)
                    _append_graph(G, now_t)

                    if self.args.visualize and now_t < min(self.T, 4 * self.cycle_len):
                        vis_save_dir = os.path.join(args_dict["save_dir"], self.dataset_name, "vis")
                        os.makedirs(vis_save_dir, exist_ok=True)
                        plt.figure(figsize=(20, 10))
                        nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
                        edge_labels = nx.get_edge_attributes(G, "weight")
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                        plt.title(f"Ordered long-range timestep {now_t}")
                        plt.savefig(os.path.join(vis_save_dir, f"{now_t}.png"))
                        plt.close()

                    pbar.update(1)

                for idle_step in range(self.lag):
                    now_t = cycle_start_t + self.branch_len + idle_step
                    G = nx.empty_graph(self.num_nodes)
                    _append_graph(G, now_t)
                    pbar.update(1)

                now_t = cycle_start_t + self.branch_len + self.lag
                G = nx.empty_graph(self.num_nodes)
                for endpoint in endpoints:
                    G.add_edge(self.effect_node, endpoint, weight=self.edge_feat_value)

                self.sequential_targets[now_t] = {
                    "endpoints": endpoints,
                }

                _append_graph(G, now_t)

                if self.args.visualize and now_t < min(self.T, 4 * self.branch_len):
                    vis_save_dir = os.path.join(args_dict["save_dir"], self.dataset_name, "vis")
                    os.makedirs(vis_save_dir, exist_ok=True)
                    plt.figure(figsize=(20, 10))
                    nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
                    edge_labels = nx.get_edge_attributes(G, "weight")
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                    plt.title(f"Ordered long-range timestep {now_t}")
                    plt.savefig(os.path.join(vis_save_dir, f"{now_t}.png"))
                    plt.close()

                pbar.update(1)

        assert t.max() == self.T - 1, f"Last timestep should be {self.T - 1}. Got {t.max()}."
        return src, dst, t, edge_feat

    def create_data(self, args_dict) -> None:
        super().create_data(args_dict)
        fdir = os.path.join(args_dict["save_dir"], self.dataset_name)
        with open(os.path.join(fdir, "ordered_targets.pkl"), "wb") as f:
            pickle.dump(
                {
                    "edge_feat_value": self.edge_feat_value,
                    "cycle_len": self.cycle_len,
                    "branch_len": self.branch_len,
                    "lag": self.lag,
                    "targets": self.sequential_targets,
                },
                f,
            )
