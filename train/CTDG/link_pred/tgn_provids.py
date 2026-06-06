from typing import Optional, Tuple

import torch

from ...CTDG.link_pred.trainer import LinkPredTrainer
from ...CTDG.tgn_provids import TGNTrainer
from ...CTDG.trainer import NODE_EMB_MODEL_NAME


class LinkPredTGNTrainer(LinkPredTrainer, TGNTrainer):
    def __init__(self):
        super().__init__()

    @property
    def parameters(self):
        return (
            set(self.model[NODE_EMB_MODEL_NAME].parameters())
            | set(self.model['memory'].parameters())
            | set(self.model['link_pred'].parameters())
        )

    def before_epoch_training(self):
        self.neighbor_loader.reset_state()
        self.model['memory'].train()
        self.model['memory'].reset_state()

    def after_iteration_training(self):
        self.model['memory'].detach()

    def after_epoch_training(self):
        pass

    def before_epoch_evaluation(self, split_mode: str):
        if split_mode == 'train':
            self.before_epoch_training()
        else:
            self.model['memory'].eval()

    def after_iteration_evaluation(self, split_mode):
        pass

    def after_epoch_evaluation(self, split_mode: str):
        pass

    def supports_temporal_replay(self) -> bool:
        return True

    def reset_temporal_state(self) -> None:
        self.neighbor_loader.reset_state()
        self.model['memory'].reset_state()
        self.model['memory'].eval()

    def replay_loader_for_memory(self, loader) -> None:
        for batch in loader:
            if len(batch) == 6:
                (pos_src, pos_dst), _, curr_t, pos_edge_feat, pos_edge_ids, mask = batch
                seq_len = pos_src.shape[1]
                curr_t = curr_t.unsqueeze(1).repeat(1, seq_len)

                pos_src = pos_src[mask].to(self.device)
                pos_dst = pos_dst[mask].to(self.device)
                pos_t = curr_t[mask].to(self.device)
                pos_edge_ids = pos_edge_ids[mask].to(self.device)
                pos_edge_feat = pos_edge_feat[mask].to(self.device)
            else:
                pos_src, pos_dst, pos_t, pos_edge_ids, pos_edge_feat = batch
                pos_src = pos_src.to(self.device)
                pos_dst = pos_dst.to(self.device)
                pos_t = pos_t.to(self.device)
                pos_edge_ids = pos_edge_ids.to(self.device)
                pos_edge_feat = pos_edge_feat.to(self.device)

            if pos_src.numel() == 0:
                continue

            self.forward_backbone(
                pos_src,
                pos_dst,
                pos_t,
                batch_edge_id=pos_edge_ids,
                batch_edge_feat=pos_edge_feat,
                update_memory=True,
            )

    def forward_backbone(self,
                         batch_src: torch.Tensor,
                         batch_dst: torch.Tensor,
                         batch_t: torch.Tensor,
                         batch_edge_id: torch.Tensor,
                         batch_edge_feat: torch.Tensor,
                         batch_neg: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None,
                         update_memory: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_neg_dst = None
        if batch_neg is not None:
            batch_neg_dst = batch_neg[1]
            n_id = torch.cat([batch_src, batch_dst, batch_neg_dst]).unique()
        else:
            n_id = torch.cat([batch_src, batch_dst]).unique()

        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        z, last_update = self.model['memory'](n_id)
        feats = self.model['memory'].node_raw_features[n_id]
        z = torch.cat([z, feats], dim=-1)

        z = self.model[NODE_EMB_MODEL_NAME](z,
                                            last_update,
                                            edge_index,
                                            self.t[e_id],
                                            self.edge_feats[e_id])

        batch_src_node_embeddings, batch_dst_node_embeddings = z[self.assoc[batch_src]], z[self.assoc[batch_dst]]
        if batch_neg_dst is None:
            batch_neg_src_node_embeddings = None
            batch_neg_dst_node_embeddings = None
        else:
            batch_neg_src = batch_neg[0]
            batch_neg_dst = batch_neg[1]
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = z[self.assoc[batch_neg_src]], z[self.assoc[batch_neg_dst]]

        if update_memory:
            self.model['memory'].update_state(batch_src, batch_dst, batch_t, batch_edge_feat.float())
            self.neighbor_loader.insert(
                batch_src,
                batch_dst
            )

        return (batch_src_node_embeddings, batch_dst_node_embeddings), (batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings)
