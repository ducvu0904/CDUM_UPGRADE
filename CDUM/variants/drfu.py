"""Dynamic Representation Fusion for Uplift Modeling (DRFU).

The prognostic representation is a function of covariates only and is shared
across both candidate outcomes. Observed treatment selects only the factual
prediction; optimisation remains the responsibility of CPMTrainer.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..cpm import CPM
from ..experts import Expert
from .treatment_interaction import (
    TreatmentInteraction,
    remap_legacy_treatment_interaction_keys,
)


class PrognosticBranch(nn.Module):
    """Covariate-only MLP: flattened features [B, F*E] -> z_P [B, D]."""

    def __init__(
        self,
        feature_dim: int,
        expert_dim: int,
        expert_hidden_dim: int = 128,
        prognostic_hidden_dim: Optional[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        self.prognostic_hidden_dim = (
            expert_hidden_dim if prognostic_hidden_dim is None else prognostic_hidden_dim
        )
        self.mlp = Expert(
            input_dim=feature_dim,
            hidden_dim=self.prognostic_hidden_dim,
            expert_dim=expert_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )

    def forward(self, e_x: torch.Tensor) -> torch.Tensor:
        return self.mlp(e_x)


class DRFURouter(nn.Module):
    """Shared router with fixed weight ordering: prognostic, CPM, interaction."""

    branch_names = ("P", "C", "I")

    def __init__(
        self,
        expert_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.expert_dim = expert_dim
        self.hidden_dim = expert_dim if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(3 * expert_dim, self.hidden_dim, bias=True)
        self.relu = nn.ReLU()
        # Only the layer directly producing competitive Softmax logits is
        # bias-free; fc1 remains an ordinary biased representation layer.
        self.fc2 = nn.Linear(self.hidden_dim, 3, bias=False)

    def forward(
        self,
        z_P: torch.Tensor,
        z_C: torch.Tensor,
        z_I: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        q_R = torch.cat((z_P, z_C, z_I), dim=-1)
        router_logits = self.fc2(self.relu(self.fc1(q_R)))
        pi = torch.softmax(router_logits, dim=-1)
        z_F = pi[:, 0:1] * z_P + pi[:, 1:2] * z_C + pi[:, 2:3] * z_I
        if return_intermediates:
            return z_F, {
                "q_R": q_R, "router_logits": router_logits, "pi": pi, "z_F": z_F,
                "fusion_z_P": z_P, "fusion_z_C": z_C, "fusion_z_I": z_I,
            }
        return z_F, pi


class DRFU(CPM):
    """Dynamic Representation Fusion for Uplift Modeling.

    Reuses CPM's encoder, experts, refinement, separate guidance gates and
    separate towers. The TwoBranchDynamicFusion architecture is independent.
    """

    def __init__(
        self,
        num_features: int = 12,
        num_bins: int = 101,
        embedding_dim: int = 32,
        treatment_dim: Optional[int] = None,
        refine_hidden_dim: int = 64,
        refine_dim: int = 32,
        num_experts: int = 3,
        expert_hidden_dim: int = 128,
        expert_dim: int = 64,
        tower_hidden_dim: Optional[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
        router_hidden_dim: Optional[int] = None,
        interaction_hidden_dim: Optional[int] = None,
        valor_hidden_dim: Optional[int] = None,
        prognostic_hidden_dim: Optional[int] = None,
    ):
        resolved_tower_hidden_dim = refine_dim if tower_hidden_dim is None else tower_hidden_dim
        if resolved_tower_hidden_dim != refine_dim:
            raise ValueError(
                f"tower_hidden_dim ({resolved_tower_hidden_dim}) must match refine_dim ({refine_dim}) "
                "because TreatmentTower multiplies its hidden state by e_indicator."
            )
        super().__init__(
            num_features=num_features, num_bins=num_bins, embedding_dim=embedding_dim,
            treatment_dim=treatment_dim, refine_hidden_dim=refine_hidden_dim,
            refine_dim=refine_dim, num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim, expert_dim=expert_dim,
            tower_hidden_dim=resolved_tower_hidden_dim, activation=activation,
            dropout_rate=dropout_rate, use_bn=use_bn,
        )
        self.num_features = num_features
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.refine_hidden_dim = refine_hidden_dim
        self.refine_dim = refine_dim
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.expert_dim = expert_dim
        self.tower_hidden_dim = resolved_tower_hidden_dim
        self.router_hidden_dim = expert_dim if router_hidden_dim is None else router_hidden_dim
        if interaction_hidden_dim is not None and valor_hidden_dim is not None:
            if interaction_hidden_dim != valor_hidden_dim:
                raise ValueError(
                    "interaction_hidden_dim and legacy valor_hidden_dim must match"
                )
        requested_interaction_dim = (
            interaction_hidden_dim
            if interaction_hidden_dim is not None
            else valor_hidden_dim
        )
        self.interaction_hidden_dim = (
            expert_hidden_dim
            if requested_interaction_dim is None
            else requested_interaction_dim
        )
        self.valor_hidden_dim = self.interaction_hidden_dim
        self.prognostic_hidden_dim = (
            expert_hidden_dim if prognostic_hidden_dim is None else prognostic_hidden_dim
        )
        feature_dim = num_features * embedding_dim
        self.prognostic_branch = PrognosticBranch(
            feature_dim=feature_dim, expert_dim=expert_dim,
            prognostic_hidden_dim=self.prognostic_hidden_dim,
            activation=activation, dropout_rate=dropout_rate, use_bn=use_bn,
        )
        self.treatment_interaction = TreatmentInteraction(
            feature_dim=feature_dim, treatment_dim=self.treatment_dim,
            expert_dim=expert_dim,
            interaction_hidden_dim=self.interaction_hidden_dim,
            activation=activation, dropout_rate=dropout_rate, use_bn=use_bn,
        )
        self.router = DRFURouter(expert_dim, self.router_hidden_dim)

    def _forward_three_branch_treatment(
        self,
        expert_outputs: torch.Tensor,
        e_x: torch.Tensor,
        z_P: torch.Tensor,
        treatment_id: int,
        return_diagnostics: bool = False,
    ):
        """Enumerate a candidate, never the observed treatment assignments."""
        candidate = torch.full(
            (e_x.shape[0],), treatment_id, dtype=torch.long, device=e_x.device
        )
        e_t = self.encoder.encode_treatment(candidate)
        e_guidance, e_indicator = self.treatment_refine(e_t)
        gate = self.control_gate if treatment_id == 0 else self.treatment_gate
        a_C = gate(e_guidance)
        z_C = (expert_outputs * a_C.unsqueeze(-1)).sum(dim=1)

        # Treatment interaction receives raw pre-refine e_t, never guidance or indicator.
        if return_diagnostics:
            z_I, interaction_diag = self.treatment_interaction(
                e_x, e_t, return_intermediates=True
            )
            z_F, router_diag = self.router(z_P, z_C, z_I, return_intermediates=True)
        else:
            z_I = self.treatment_interaction(e_x, e_t)
            z_F, _ = self.router(z_P, z_C, z_I)

        tower = self.control_tower if treatment_id == 0 else self.treatment_tower
        y_hat = tower(z_F, e_indicator)
        if return_diagnostics:
            diagnostics = {
                "e_x": e_x, "e_t": e_t,
                "e_guidance": e_guidance, "e_indicator": e_indicator,
                "expert_outputs": expert_outputs, "a_C": a_C,
                "z_P": z_P, "z_C": z_C, "z_I": z_I,
                "h_x": interaction_diag["h_x"], "m_t": interaction_diag["m_t"],
                "interaction": interaction_diag["interaction"],
                **router_diag,
                "y_hat": y_hat, "treatment_id": treatment_id,
            }
            return y_hat, a_C, e_indicator, diagnostics
        return y_hat, a_C, e_indicator

    @property
    def valor_branch(self) -> TreatmentInteraction:
        """Deprecated alias for ``treatment_interaction``."""
        return self.treatment_interaction

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(
            remap_legacy_treatment_interaction_keys(state_dict),
            strict=strict,
            assign=assign,
        )

    def forward(
        self, x_ids: torch.Tensor, t: torch.Tensor, return_diagnostics: bool = False
    ) -> Dict[str, Any]:
        """Compute both outcomes in one call; t [B] or [B,1] selects y_factual."""
        e_x = self.encoder.encode_features(x_ids).flatten(start_dim=1)
        z_P = self.prognostic_branch(e_x)
        expert_outputs = self.user_experts(e_x)
        res0 = self._forward_three_branch_treatment(expert_outputs, e_x, z_P, 0, return_diagnostics)
        res1 = self._forward_three_branch_treatment(expert_outputs, e_x, z_P, 1, return_diagnostics)
        y0_hat, g0, ind0 = res0[:3]
        y1_hat, g1, ind1 = res1[:3]
        t_float = t.float().view(-1, 1)
        outputs = {
            "y_factual": (1.0 - t_float) * y0_hat + t_float * y1_hat,
            "y0": y0_hat, "y1": y1_hat, "y0_hat": y0_hat, "y1_hat": y1_hat,
            "uplift": y1_hat - y0_hat,
            "g0": g0, "g1": g1, "ind0": ind0, "ind1": ind1,
        }
        if return_diagnostics:
            outputs["diagnostics"] = {0: res0[3], 1: res1[3]}
        return outputs
