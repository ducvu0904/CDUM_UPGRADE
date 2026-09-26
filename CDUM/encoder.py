from typing import Optional
import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        num_features: int = 12,
        num_bins: int = 101,
        embedding_dim: int = 32,
        treatment_dim: Optional[int] = None,
    ):
        super(FeatureEncoder, self).__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        
        self.treatment_dim = treatment_dim if treatment_dim is not None else (embedding_dim * 4)

        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(num_bins, embedding_dim) for _ in range(num_features)]
        )
        self.treatment_embeddings = nn.Embedding(2, self.treatment_dim)

    def encode_treatment(self, t: torch.Tensor):
        t = t.long()
        t_emb = self.treatment_embeddings(t)  # [B, 128]
        return t_emb

    def encode_features(self, x: torch.Tensor):
        feature_embeddings = []
        for j in range(self.num_features):
            feat_values = x[:, j].long()
            feat_emb = self.feature_embeddings[j](feat_values)
            feature_embeddings.append(feat_emb)

        x_emb = torch.stack(feature_embeddings, dim=1)  # [B, 12, 32]
        return x_emb

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x_emb = self.encode_features(x)
        t_emb = self.encode_treatment(t)
        return x_emb, t_emb


class FeatureAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_emb: torch.Tensor, t_emb: Optional[torch.Tensor] = None):
        # Concatenate / flatten non-treatment user feature embeddings: [B, 12, 32] -> [B, 384]
        x_emb = x_emb.flatten(start_dim=1)
        if t_emb is not None:
            return x_emb, t_emb
        return x_emb