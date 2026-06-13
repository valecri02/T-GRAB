import argparse
import os
from typing import Dict

import torch
from torch.nn.modules import Module

from ...dataset.CTDG.torch_dataset.link_pred.node_feat_static import ContinuousTimeLinkPredNodeFeatureStaticDataset
from ...model.TGB import LastAggregator, MeanAggregator, SequentialAggregator, LastNeighborLoader
from ...model.TGB.provids_mlstm import (
    GraphAttentionEmbeddingProvIDS,
    IdentityMessageProvIDS,
    TGNMemoryProvIDSMLSTM,
    compute_provids_delta_t_stats,
)
from .trainer import CTDGTrainer


class TGNTrainer(CTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of TGN units', default=1)
        parser.add_argument('--num-heads', type=int, help='Number of attention heads in TGN', default=2)
        parser.add_argument('--dropout', type=float, help='TGN: dropout rate', default=0.1)
        parser.add_argument('--num-neighbors', type=int, help='TGN: number of neighbors to sample for each node', default=20)
        parser.add_argument('--time-feat-dim', type=int, default=100, help='dimension of the time embedding')
        parser.add_argument('--memory-dim', type=int, default=100, help='dimension of the memory')
        parser.add_argument('--mlstm-num-heads', type=int, default=4, help='Number of heads in the mLSTM memory updater')
        parser.add_argument('--mlstm-state-max-nodes', type=int, default=None, help='Optional cap for cached mLSTM node states')
        parser.add_argument('--message-aggregator', choices=['last', 'mean', 'sequence'], default='last',
                            help='TGN: aggregate messages with last/mean or process them sequentially by timestamp.')
        parser.add_argument('--memory-enhancement', type=int, default=0, choices=[0, 2],
                            help='ProvIDS memory enhancement mode. 2 updates memory messages with GNN embeddings.')
        return parser

    def get_model(self) -> Dict[str, Module]:
        models = super(TGNTrainer, self).get_model()
        self.full_dataset = ContinuousTimeLinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc),
                                                                      self.args.data,
                                                                      "all",
                                                                      self.args.node_feat,
                                                                      self.args.node_feat_dim)
        train_dataset_for_stats = ContinuousTimeLinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc),
                                                                      self.args.data,
                                                                      "train",
                                                                      self.args.node_feat,
                                                                      self.args.node_feat_dim)

        self.edge_feats = self.full_dataset.edge_feat.to(self.device)
        self.t = self.full_dataset.t.to(self.device)

        edge_dim = self.full_dataset.edge_feat.size(-1)
        node_feat_dim = self.full_dataset._node_feat.size(-1)
        init_time = self.full_dataset.t[0].item()
        mean_delta_t, std_delta_t = compute_provids_delta_t_stats(train_dataset_for_stats, init_time)
        edge_encoder = torch.nn.Linear(edge_dim, edge_dim).to(self.device)

        aggregator_modules = {
            'last': LastAggregator,
            'mean': MeanAggregator,
            'sequence': SequentialAggregator,
        }
        aggregator_module = aggregator_modules[self.args.message_aggregator]()

        memory = TGNMemoryProvIDSMLSTM(
                    self.full_dataset.num_nodes,
                    edge_dim,
                    node_feat_dim,
                    self.args.time_feat_dim,
                    self.full_dataset._node_feat.to(self.device),
                    message_module=IdentityMessageProvIDS(edge_dim,
                                                          node_feat_dim,
                                                          self.args.time_feat_dim,
                                                          edge_encoder=edge_encoder),
                    aggregator_module=aggregator_module,
                    mlstm_num_heads=self.args.mlstm_num_heads,
                    mlstm_state_max_nodes=self.args.mlstm_state_max_nodes).to(self.device)

        backbone = GraphAttentionEmbeddingProvIDS(
                in_channels=node_feat_dim * 2,
                out_channels=node_feat_dim,
                msg_dim=edge_dim,
                time_enc=memory.time_enc,
                mean_delta_t=mean_delta_t,
                std_delta_t=std_delta_t,
                encode_edge=True,
                edge_encoder=edge_encoder).to(self.device)

        self.neighbor_loader = LastNeighborLoader(self.full_dataset.num_nodes, size=self.args.num_neighbors, device=self.device)
        self.assoc = torch.empty(self.full_dataset.num_nodes, dtype=torch.long, device=self.device)

        models['memory'] = memory
        models['node_emb'] = backbone

        return models

    def _get_run_save_dir(self) -> str:
        ctdgtrainer_run_save_dir = super(TGNTrainer, self)._get_run_save_dir()
        return os.path.join(ctdgtrainer_run_save_dir, "tgn_provids_mlstm")
