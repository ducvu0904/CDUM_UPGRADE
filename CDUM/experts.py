import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        expert_dim: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super(Expert, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, expert_dim)

        self.relu = nn.ReLU() if activation.lower() == "relu" else nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.hidden(x)
        h = self.relu(h)
        f = self.output(h)
        f = self.relu(f)
        return f


class UserExpert(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int = 128,
        expert_dim: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ):
        super(UserExpert, self).__init__()

        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                expert_dim=expert_dim,
                activation=activation,
                dropout_rate=dropout_rate,
                use_bn=use_bn,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        outputs = []

        for expert in self.experts:
            f = expert(x)
            outputs.append(f)

        return torch.stack(outputs, dim=1)


class GuidanceGate(nn.Module):
    def __init__(self, guidance_dim: int = 32, num_experts: int = 3):
        super(GuidanceGate, self).__init__()
        # Official Criteo CPM uses Linear(32, 3, bias=False) followed by Softmax(dim=1)
        self.gate = nn.Linear(guidance_dim, num_experts, bias=False)

    def forward(self, guidance_emb: torch.Tensor):
        gate_logits = self.gate(guidance_emb)
        gate_weights = torch.softmax(gate_logits, dim=1)
        return gate_weights

        