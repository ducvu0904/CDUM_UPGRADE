"""
cpm_processor.py — Official Criteo Feature Discretization Scheme for CDUM/CPM
=============================================================================
Matches the official UpliftVideo/CDUM Criteo preprocessing:
1. Denominators are computed ONLY from the training split:
       denominator_j = max(train_feature_j)
   Safely guarded against zero denominators:
       denominator_j = max(1e-8, denominator_j)
2. At transform time:
       scaled = x_j / denominator_j * 100
3. Truncate to integer (matching TensorFlow tf.cast(..., tf.int32)):
       bucket_id = int(scaled)
4. Clamp IDs to [0, 100]:
       bucket_id = clamp(bucket_id, 0, 100)
   Yielding exactly 101 possible bucket IDs: 0, 1, ..., 100.
5. Numerical feature embeddings in CDUM/CPM use nn.Embedding(101, embedding_dim).
   Bucket IDs are returned as torch.long in the range [0, 100].

NOTE:
- Minimum values are NOT subtracted: no (x - min) / (max - min).
- StandardScaler is NOT used for CPM input.
- Fits ONLY on the training split; validation/test data is never used to fit.
"""

from typing import Union, Optional
import numpy as np
import torch


class OfficialCPMBucketer:
    """
    Official Criteo Feature Discretization Scheme for CDUM/CPM.
    
    Formula per feature j:
        denominator_j = max(1e-8, max(train_feature_j))
        scaled = x_j / denominator_j * 100
        bucket_id = clamp(int(scaled), 0, 100)
    
    Produces bucket IDs in [0, 100] (101 possible buckets) with dtype torch.long.
    """

    def __init__(self, num_bins: int = 101, dim: int = 100):
        # dim is the scaling factor (100 in official implementation)
        self.dim = dim
        # num_bins is dim + 1 = 101 (indices 0..100)
        self.num_bins = num_bins if num_bins == (dim + 1) else (dim + 1)
        self.denominators: Optional[torch.Tensor] = None
        self.max_values: Optional[torch.Tensor] = None

    def to(self, device: Union[str, torch.device]):
        """Move fitted denominator and max tensors to the specified device."""
        if self.denominators is not None:
            self.denominators = self.denominators.to(device)
        if self.max_values is not None:
            self.max_values = self.max_values.to(device)
        return self

    def fit(self, X: Union[torch.Tensor, np.ndarray]):
        """
        Fit denominators ONLY on the training split continuous features.
        
        denominator_j = max(1e-8, max(train_feature_j))
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        else:
            X = X.float()

        if X.dim() == 1:
            X = X.unsqueeze(1)

        X = X.contiguous()

        # Compute maximum value per feature along dimension 0
        max_values = X.max(dim=0).values

        # Official formula: denominator = max(1e-8, max(train_feature))
        denominators = torch.clamp(max_values, min=1e-8)

        self.denominators = denominators.contiguous()
        self.max_values = max_values.contiguous()
        return self

    def transform(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Discretize continuous features into bucket IDs in [0, 100].
        
        scaled = x_j / denominator_j * 100
        bucket_id = clamp(int(scaled), 0, 100)
        
        Returns torch.Tensor of dtype torch.long.
        """
        if self.denominators is None:
            raise RuntimeError("The bucketer has not been fitted yet. Call 'fit' before 'transform'.")

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        else:
            X = X.float()

        is_1d = X.dim() == 1
        if is_1d:
            X = X.unsqueeze(0)

        X = X.contiguous()
        num_features = X.shape[1]

        if self.denominators.shape[0] != num_features:
            raise ValueError(
                f"Feature dimension mismatch: bucketer fitted with {self.denominators.shape[0]} features, "
                f"but input has {num_features} features."
            )

        denoms = self.denominators.to(device=X.device, dtype=X.dtype)

        # Scale features: feature / denominator * 100
        scaled = (X / denoms) * float(self.dim)

        # Truncation equivalent to TensorFlow tf.cast(..., tf.int32) / Python int(scaled)
        # Followed by clamp to [0, dim]
        # In PyTorch, .to(torch.int64) truncates towards zero (positive floats floor, negative floats ceil toward 0).
        # Clamping between 0 and self.dim maps negatives to 0 and values > train_max to self.dim (100).
        bucket_ids = torch.clamp(scaled.to(torch.int64), min=0, max=self.dim)

        if is_1d:
            bucket_ids = bucket_ids.squeeze(0)

        return bucket_ids.to(torch.long).contiguous()

    def fit_transform(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Fit on training split and transform in one call."""
        self.fit(X)
        return self.transform(X)


# Backward compatibility aliases
EquidistantBucketer = OfficialCPMBucketer
CriteoBucketer = OfficialCPMBucketer


