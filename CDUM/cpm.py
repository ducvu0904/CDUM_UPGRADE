from typing import Optional
import torch
import torch.nn as nn
from .experts import UserExpert, GuidanceGate
from .encoder import FeatureEncoder
from .treatment_refine import TreatmentRefine

class TreatmentTower(nn.Module):
    def __init__(
        self, 
        input_dim: int = 64,
        hidden_dim: int = 32,
        activation: str = "relu",
    ):
        super(TreatmentTower, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU() if activation.lower() == "relu" else nn.Identity()
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()
        
    def forward(self, mixed: torch.Tensor, e_ind: torch.Tensor):
        h = self.layer1(mixed)
        h = self.relu(h)
        
        h_masked = h * e_ind   
        
        y_hat = self.softplus(self.layer2(h_masked)) 
        return y_hat  

class CPM(nn.Module):
    def __init__(self, 
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
                 use_bn: bool = False):
        super(CPM, self).__init__()
        
        if treatment_dim is None:
            # Official Criteo CPM: treatment embedding dimension = embedding_dim * 4 = 128
            treatment_dim = embedding_dim * 4
        self.treatment_dim = treatment_dim

        if tower_hidden_dim is None:
            tower_hidden_dim = refine_dim  # default 32 to match indicator embedding e_ind
        
        self.encoder = FeatureEncoder(
            num_features=num_features,
            num_bins=num_bins,
            embedding_dim=embedding_dim,
            treatment_dim=treatment_dim,
        )
        self.treatment_refine = TreatmentRefine(
            treatment_dim=treatment_dim,
            hidden_dim=refine_hidden_dim,
            output_dim=refine_dim,
        )
        self.user_experts = UserExpert(
            num_experts=num_experts,
            input_dim=num_features * embedding_dim,
            hidden_dim=expert_hidden_dim,
            expert_dim=expert_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )
        
        # Treatment-specific guidance gates (separate modules)
        self.control_gate = GuidanceGate(guidance_dim=refine_dim, num_experts=num_experts)
        self.treatment_gate = GuidanceGate(guidance_dim=refine_dim, num_experts=num_experts)
        
        # Treatment-specific towers (separate modules)
        self.control_tower = TreatmentTower(
            input_dim=expert_dim,
            hidden_dim=tower_hidden_dim,
            activation=activation,
        )
        self.treatment_tower = TreatmentTower(
            input_dim=expert_dim,
            hidden_dim=tower_hidden_dim,
            activation=activation,
        )
        
    def _forward_treatment(self, 
                           expert_outputs: torch.Tensor,
                           treatment_id: int):
        batch_size = expert_outputs.shape[0]
        
        t = torch.full(
            (batch_size,), 
            treatment_id,
            dtype=torch.long,
            device=expert_outputs.device
        )
        
        t_emb = self.encoder.encode_treatment(t)  # [B, 128]
        
        e_guidance, e_indicator = self.treatment_refine(t_emb)  # [B, 32], [B, 32]
        
        if treatment_id == 0:
            gate_weights = self.control_gate(e_guidance)  # [B, 3]
        else:
            gate_weights = self.treatment_gate(e_guidance)  # [B, 3]
        
        gate_weights = gate_weights.unsqueeze(-1)  # [B, 3, 1]
        mixed = (expert_outputs * gate_weights).sum(dim=1)  # [B, 64]
        
        if treatment_id == 0:
            y_hat = self.control_tower(mixed, e_indicator)  # [B, 1]
        else:
            y_hat = self.treatment_tower(mixed, e_indicator)  # [B, 1]
        
        return y_hat, gate_weights.squeeze(-1), e_indicator
    
    def forward(self, x_ids: torch.Tensor, t: torch.Tensor):
        # 1. Encode user features and flatten: [B, 12, 32] -> [B, 384]
        x_emb = self.encoder.encode_features(x_ids)
        x_star = x_emb.flatten(start_dim=1)
        
        # 2. Compute experts output once: [B, 3, 64]
        expert_outputs = self.user_experts(x_star)
        
        # 3. Compute counterfactual predictions, gates, and indicators
        y0_hat, g0, ind0 = self._forward_treatment(expert_outputs, treatment_id=0)  # [B, 1], [B, 3], [B, 32]
        y1_hat, g1, ind1 = self._forward_treatment(expert_outputs, treatment_id=1)  # [B, 1], [B, 3], [B, 32]
        
        # 4. Factual outcome according to observed treatment t
        t_float = t.float().view(-1, 1)
        y_factual = (1.0 - t_float) * y0_hat + t_float * y1_hat
        
        # 5. Uplift output
        uplift = y1_hat - y0_hat
        
        return {
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
        
        
        