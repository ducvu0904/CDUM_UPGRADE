"""Treatment-interaction representation shared by uplift-model variants."""

from collections import OrderedDict
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..experts import Expert


def remap_legacy_treatment_interaction_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    """Map legacy ``valor_branch.*`` checkpoint keys to the canonical name."""
    remapped = OrderedDict()
    for key, value in state_dict.items():
        canonical_key = key.replace("valor_branch.", "treatment_interaction.", 1)
        remapped[canonical_key] = value
    if hasattr(state_dict, "_metadata"):
        remapped._metadata = OrderedDict(  # type: ignore[attr-defined]
            (
                key.replace("valor_branch", "treatment_interaction", 1),
                value,
            )
            for key, value in state_dict._metadata.items()  # type: ignore[attr-defined]
        )
    return remapped


class TreatmentInteraction(nn.Module):
    """Build a treatment-gated interaction representation from ``e_x`` and ``e_t``.

    The branch uses the raw pre-refinement treatment embedding:

    ``h_x = Linear(e_x)``, ``m_t = sigmoid(Linear(e_t))``,
    ``z_I = MLP(h_x * m_t)``.
    """

    def __init__(
        self,
        feature_dim: int,
        treatment_dim: int,
        expert_dim: int,
        expert_hidden_dim: int = 128,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
        interaction_hidden_dim: Optional[int] = None,
        valor_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.treatment_dim = treatment_dim
        self.expert_dim = expert_dim
        if interaction_hidden_dim is not None and valor_hidden_dim is not None:
            if interaction_hidden_dim != valor_hidden_dim:
                raise ValueError(
                    "interaction_hidden_dim and legacy valor_hidden_dim must match"
                )
        requested_hidden_dim = (
            interaction_hidden_dim
            if interaction_hidden_dim is not None
            else valor_hidden_dim
        )
        resolved_hidden_dim = (
            expert_hidden_dim if requested_hidden_dim is None else requested_hidden_dim
        )
        self.expert_hidden_dim = resolved_hidden_dim
        self.interaction_hidden_dim = resolved_hidden_dim
        # Kept as non-architectural metadata for old callers.
        self.valor_hidden_dim = resolved_hidden_dim

        self.linear_x = nn.Linear(feature_dim, expert_dim)
        self.linear_t = nn.Linear(treatment_dim, expert_dim)
        self.mlp = Expert(
            input_dim=expert_dim,
            hidden_dim=resolved_hidden_dim,
            expert_dim=expert_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )

    def forward(
        self,
        e_x: torch.Tensor,
        e_t: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the treatment-interaction representation and intermediates."""
        h_x = self.linear_x(e_x)
        m_t = torch.sigmoid(self.linear_t(e_t))
        interaction = h_x * m_t
        z_I = self.mlp(interaction)

        if return_intermediates:
            return z_I, {
                "h_x": h_x,
                "m_t": m_t,
                "interaction": interaction,
                "z_I": z_I,
                "z_V": z_I,
            }
        return z_I
