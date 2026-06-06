from typing import Any, Dict, List
from argparse import ArgumentParser
import re
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score, f1_score
import timeit
import torch
from tgb.utils.utils import set_random_seed, save_results

from .....utils import  EarlyStopMonitor
from ....DTDG.link_pred.trainer import LinkPredTrainer
from .....dataset.DTDG.graph_generation.mem_node import MemoryNodeGenerator


class MemoryNodeTrainer(LinkPredTrainer):
    """ This class is only useful to train memory-node based datasets. 
    For more info about periodic datasets, please checkout the documentation of `MemoryNodeGenerator` class.
    """
    def __init__(self):
        super(MemoryNodeTrainer, self).__init__()
        data_pattern = self.args.data.split("/")[0]
        match = re.fullmatch(MemoryNodeGenerator.REGEX, data_pattern)
        if match:
            self.K = int(match.group(1))

    def get_dataset_regex_pattern(self):
        dataset_regex_pattern = MemoryNodeGenerator.REGEX
        dataset_name_part_to_check = self.args.data.split("/")[0]
        return dataset_regex_pattern, dataset_name_part_to_check
   
    def list_of_metrics_names(self) -> List[str]:
        return [
            "memnode_avg_precision",
            "memnode_avg_recall",
            "memnode_avg_f1",
            "memnode_avg_acc",
            "memnode_avg_loss",
            "memnode_avg_pos_rate",
            "memnode_avg_pred_pos_rate",
            "memnode_avg_pos_score",
            "memnode_avg_neg_score",
            "memnode_avg_auc",
            "memnode_avg_ap",
        ]

    def _get_graph_mask_ground_truth_and_pred(self, graph_mask, curr_snapshot):
        graph_src_indices, graph_dst_indices = torch.nonzero(graph_mask, as_tuple=True)
        graph_ground_truth = curr_snapshot[graph_src_indices, graph_dst_indices].float()
        graph_prediction = self.model['link_pred'](self.z[graph_src_indices], self.z[graph_dst_indices])
        graph_prediction = graph_prediction.squeeze(-1)
        graph_pred_bin = graph_prediction >= 0.5

        return graph_ground_truth, graph_prediction, graph_pred_bin

    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str):
        # The first k snapshots (K the time gap between when discovery and pattern happen) of -
        # test-inductive is not valid as being "inductive" because memory node in those steps -
        # discover patterns appeared in 'test' split which are transductive.
        # Simply skip those steps to avoid errors during evaluation.
        if split_mode == 'test_inductive' and snapshot_idx < self.K:
            return
        
        # Memory node graph evaluation
        num_nodes = self.train_loader.dataset.num_nodes
        memnode_graph_mask = torch.zeros((num_nodes, num_nodes)).to(self.device)
        memnode_graph_mask[0, :] = 1
        memnode_graph_mask[:, 0] = 1

        ## Skip self-loop edges during evaluation
        memnode_graph_mask.fill_diagonal_(0)
        memnode_graph_ground_truth, memnode_graph_prediction, memnode_graph_pred_bin = self._get_graph_mask_ground_truth_and_pred(memnode_graph_mask, curr_snapshot)
        del memnode_graph_mask
        
        memnode_graph_precision = precision_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy(), zero_division=0)
        memnode_graph_recall = recall_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy(), zero_division=0)
        memnode_graph_f1 = f1_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy(), zero_division=0)
        memnode_graph_acc = accuracy_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy())
        memnode_graph_loss = self.criterion(memnode_graph_prediction, memnode_graph_ground_truth)
        memnode_pos_mask = memnode_graph_ground_truth > 0
        memnode_neg_mask = ~memnode_pos_mask
        memnode_pos_rate = memnode_pos_mask.float().mean().item()
        memnode_pred_pos_rate = memnode_graph_pred_bin.float().mean().item()
        memnode_pos_score = memnode_graph_prediction[memnode_pos_mask].mean().item() if torch.any(memnode_pos_mask) else float("nan")
        memnode_neg_score = memnode_graph_prediction[memnode_neg_mask].mean().item() if torch.any(memnode_neg_mask) else float("nan")
        memnode_ground_truth_np = memnode_graph_ground_truth.cpu().numpy()
        memnode_prediction_np = memnode_graph_prediction.detach().cpu().numpy()
        if np.unique(memnode_ground_truth_np).size > 1:
            memnode_auc = roc_auc_score(memnode_ground_truth_np, memnode_prediction_np)
            memnode_ap = average_precision_score(memnode_ground_truth_np, memnode_prediction_np)
        else:
            memnode_auc = float("nan")
            memnode_ap = float("nan")
        
        ## Record metrics in total
        metrics_list[f"memnode_avg_precision"].append(memnode_graph_precision)
        metrics_list[f"memnode_avg_recall"].append(memnode_graph_recall)
        metrics_list[f"memnode_avg_f1"].append(memnode_graph_f1)
        metrics_list[f"memnode_avg_acc"].append(memnode_graph_acc)
        metrics_list[f"memnode_avg_loss"].append(memnode_graph_loss.item())
        metrics_list[f"memnode_avg_pos_rate"].append(memnode_pos_rate)
        metrics_list[f"memnode_avg_pred_pos_rate"].append(memnode_pred_pos_rate)
        metrics_list[f"memnode_avg_pos_score"].append(memnode_pos_score)
        metrics_list[f"memnode_avg_neg_score"].append(memnode_neg_score)
        metrics_list[f"memnode_avg_auc"].append(memnode_auc)
        metrics_list[f"memnode_avg_ap"].append(memnode_ap)


    def _eval_predict_current_timestep(self, curr_snapshot: torch.Tensor):
        """ This function visualizes model's output alongside ground-truth on memory-node graphs. 
        As the model's prediction on memory-node has higher importance, a subpart of graph that only shows memory-node links is visualized, and the rest is discarded.
        """

        z = self.z

        memory_node_subgraph_mask = torch.zeros_like(curr_snapshot)
        memory_node_subgraph_mask[0] = 1
        memory_node_subgraph_mask[:, 0] = 1
        src, dst = torch.nonzero(memory_node_subgraph_mask, as_tuple=True)
        pred = self.model['link_pred'](z[src], z[dst])

        out_2d = torch.zeros_like(curr_snapshot)
        out_2d[src, dst] = pred.squeeze(-1).detach().cpu()
        out_2d.fill_diagonal_(0)  # Model does not predict self-loop edges

        del memory_node_subgraph_mask

        return out_2d


    def early_stopping_checker(self, early_stopper) -> bool:
        if self.test_perf[self.val_first_metric] == 1:
            return True

        return early_stopper.step_check(self._early_stop_metric, self.model, op_to_cont="dec")