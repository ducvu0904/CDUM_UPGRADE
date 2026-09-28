"""Two-branch fusion of CPM and treatment-interaction representations.

Combines representation-level expert routing with a treatment-gated interaction
branch and a dynamic fusion router to compute fused representation z_F
for each treatment branch.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from ..cpm import CPM
from .treatment_interaction import (
    TreatmentInteraction,
    remap_legacy_treatment_interaction_keys,
)


class DynamicFusionRouter(nn.Module):
    """Dynamic router combining CPM and treatment-interaction representations.

    Implements:
        q_R = Concat(z_C, z_I)            -> shape [B, 2D]
        hidden = ReLU(fc1(q_R))           -> shape [B, hidden_dim]
        router_logits = fc2(hidden)       -> shape [B, 2]
        pi = Softmax(router_logits, -1)   -> shape [B, 2]
        z_F = pi[..., 0:1]*z_C + pi[..., 1:2]*z_I -> shape [B, D]
    """

    branch_names = ("C", "I")

    def __init__(self, expert_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.expert_dim = expert_dim
        self.hidden_dim = expert_dim if hidden_dim is None else hidden_dim

        self.fc1 = nn.Linear(2 * self.expert_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, 2)

    def forward(
        self,
        z_C: torch.Tensor,
        z_I: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute dynamic fusion weights and combined representation."""
        q_R = torch.cat([z_C, z_I], dim=-1)
        hidden = self.relu(self.fc1(q_R))
        router_logits = self.fc2(hidden)
        pi = torch.softmax(router_logits, dim=-1)
        z_F = pi[..., 0:1] * z_C + pi[..., 1:2] * z_I

        if return_intermediates:
            intermediates = {
                "q_R": q_R,
                "router_logits": router_logits,
                "pi": pi,
                "z_F": z_F,
            }
            return z_F, intermediates
        return z_F, pi


class TwoBranchDynamicFusion(CPM):
    """CPM with a treatment-interaction branch and dynamic fusion router.

    Extends CPM by augmenting expert representations with treatment interactions
    interactions via a dynamic fusion router for each treatment tower.
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
    ):
        resolved_tower_hidden_dim = (
            refine_dim if tower_hidden_dim is None else tower_hidden_dim
        )
        if resolved_tower_hidden_dim != refine_dim:
            raise ValueError(
                f"tower_hidden_dim ({resolved_tower_hidden_dim}) must match refine_dim ({refine_dim}) "
                f"because TreatmentTower performs element-wise multiplication with indicator embedding e_ind."
            )

        super().__init__(
            num_features=num_features,
            num_bins=num_bins,
            embedding_dim=embedding_dim,
            treatment_dim=treatment_dim,
            refine_hidden_dim=refine_hidden_dim,
            refine_dim=refine_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            expert_dim=expert_dim,
            tower_hidden_dim=resolved_tower_hidden_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
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
        self.router_hidden_dim = (
            expert_dim if router_hidden_dim is None else router_hidden_dim
        )
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
        resolved_interaction_hidden_dim = (
            expert_hidden_dim
            if requested_interaction_dim is None
            else requested_interaction_dim
        )
        self.interaction_hidden_dim = resolved_interaction_hidden_dim
        self.valor_hidden_dim = resolved_interaction_hidden_dim

        feature_dim = num_features * embedding_dim
        self.treatment_interaction = TreatmentInteraction(
            feature_dim=feature_dim,
            treatment_dim=self.treatment_dim,
            expert_dim=expert_dim,
            expert_hidden_dim=resolved_interaction_hidden_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            interaction_hidden_dim=resolved_interaction_hidden_dim,
        )
        self.router = DynamicFusionRouter(
            expert_dim=expert_dim,
            hidden_dim=self.router_hidden_dim,
        )

    def _forward_fused_treatment(
        self,
        expert_outputs: torch.Tensor,
        e_x: torch.Tensor,
        treatment_id: int,
        return_diagnostics: bool = False,
    ):
        """Compute predictions and representations for a single treatment branch."""
        batch_size = expert_outputs.shape[0]
        device = expert_outputs.device

        t = torch.full((batch_size,), treatment_id, dtype=torch.long, device=device)
        e_t = self.encoder.encode_treatment(t)
        e_guidance, e_indicator = self.treatment_refine(e_t)

        gate = self.control_gate if treatment_id == 0 else self.treatment_gate
        a_C = gate(e_guidance)
        z_C = (expert_outputs * a_C.unsqueeze(-1)).sum(dim=1)

        if return_diagnostics:
            z_I, interaction_diag = self.treatment_interaction(
                e_x, e_t, return_intermediates=True
            )
            z_F, router_diag = self.router(z_C, z_I, return_intermediates=True)
        else:
            z_I = self.treatment_interaction(e_x, e_t, return_intermediates=False)
            z_F, pi = self.router(z_C, z_I, return_intermediates=False)

        tower = self.control_tower if treatment_id == 0 else self.treatment_tower
        y_hat = tower(z_F, e_indicator)

        if return_diagnostics:
            diagnostics = {
                "e_x": e_x,
                "e_t": e_t,
                "expert_outputs": expert_outputs,
                "a_C": a_C,
                "z_C": z_C,
                "h_x": interaction_diag["h_x"],
                "m_t": interaction_diag["m_t"],
                "interaction": interaction_diag["interaction"],
                "z_I": z_I,
                "z_V": z_I,
                "q_R": router_diag["q_R"],
                "router_logits": router_diag["router_logits"],
                "pi": router_diag["pi"],
                "z_F": z_F,
                "e_guidance": e_guidance,
                "e_indicator": e_indicator,
                "y_hat": y_hat,
                "treatment_id": treatment_id,
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
        self,
        x_ids: torch.Tensor,
        t: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass for TwoBranchDynamicFusion.

        Args:
            x_ids: Discretized feature IDs [B, num_features].
            t: Observed treatment indicators [B] or [B, 1].
            return_diagnostics: Opt-in flag (default False).
                When False, returns exact baseline output dictionary.
                When True, includes outputs['diagnostics'] = {0: diag0, 1: diag1}.

        Returns:
            Dictionary with baseline keys:
                'y_factual', 'y0', 'y1', 'y0_hat', 'y1_hat', 'uplift',
                'g0', 'g1', 'ind0', 'ind1'
            Plus 'diagnostics' mapping {0: diag0, 1: diag1} if return_diagnostics is True.
        """
        # 1. Flatten feature embeddings: [B, num_features, embedding_dim] -> [B, num_features * embedding_dim]
        x_emb = self.encoder.encode_features(x_ids)
        e_x = x_emb.flatten(start_dim=1)

        # 2. Pre-compute shared user experts once: [B, num_experts, expert_dim]
        expert_outputs = self.user_experts(e_x)

        # 3. Treatment branches: treatment 0 then treatment 1
        res0 = self._forward_fused_treatment(expert_outputs, e_x, 0, return_diagnostics)
        res1 = self._forward_fused_treatment(expert_outputs, e_x, 1, return_diagnostics)

        y0_hat, g0, ind0 = res0[:3]
        y1_hat, g1, ind1 = res1[:3]

        # 4. Factual outcome according to observed treatment t
        t_float = t.float().view(-1, 1)
        y_factual = (1.0 - t_float) * y0_hat + t_float * y1_hat

        # 5. Uplift output
        uplift = y1_hat - y0_hat

        outputs: Dict[str, Any] = {
            "y_factual": y_factual,
            "y0": y0_hat,
            "y1": y1_hat,
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "uplift": uplift,
            "g0": g0,
            "g1": g1,
            "ind0": ind0,
            "ind1": ind1,
        }

        if return_diagnostics:
            outputs["diagnostics"] = {0: res0[3], 1: res1[3]}

        return outputs
