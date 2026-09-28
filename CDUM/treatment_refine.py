import torch
import torch.nn as nn

class TreatmentRefine(nn.Module):
    def __init__(self, 
                 treatment_dim: int = 128,
                 hidden_dim: int = 64,
                 output_dim: int = 32):
        super(TreatmentRefine, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.guidance_hidden = nn.Linear(treatment_dim, hidden_dim)
        self.guidance_output = nn.Linear(hidden_dim, output_dim)

        self.indicator_hidden = nn.Linear(treatment_dim, hidden_dim)
        self.indicator_output = nn.Linear(hidden_dim, output_dim)
        if self.indicator_output.bias is not None:
            nn.init.zeros_(self.indicator_output.bias)

    def forward(self, t_emb: torch.Tensor):
        
        h_guidance = self.guidance_hidden(t_emb)
        h_guidance = self.relu(h_guidance)
        e_guidance = self.guidance_output(h_guidance)
        e_guidance = self.sigmoid(e_guidance)
        
        h_indicator = self.indicator_hidden(t_emb)
        h_indicator = self.relu(h_indicator)
        e_indicator = self.indicator_output(h_indicator)
        e_indicator = self.sigmoid(e_indicator)

        return e_guidance, e_indicator
