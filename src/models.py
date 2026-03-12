"""
FiLM-Enhanced Multi-Task Detection Model
Combines YOLOv8 backbone with clinical FiLM injection
and multi-task survival prediction head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalEncoder(nn.Module):
    """
    Encodes clinical variables into FiLM modulation parameters.
    Input : [age_normalized, resection_binary]
    Output: gamma (scale), beta (shift) for feature modulation
    """
    def __init__(self, clinical_dim=2, feature_channels=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.gamma_head = nn.Linear(128, feature_channels)
        self.beta_head  = nn.Linear(128, feature_channels)
        nn.init.ones_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, clinical):
        h = self.mlp(clinical)
        return self.gamma_head(h), self.beta_head(h)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def forward(self, feature, gamma, beta):
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return g * feature + b


class SurvivalPredictionHead(nn.Module):
    """
    Predicts patient survival days from pooled neck features
    and clinical embedding jointly.
    """
    def __init__(self, neck_channels=256, clinical_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(neck_channels + clinical_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, neck_feat, clinical):
        pooled = F.adaptive_avg_pool2d(neck_feat, 1).squeeze(-1).squeeze(-1)
        x = torch.cat([pooled, clinical], dim=1)
        return self.net(x).squeeze(-1)


def concordance_index(pred, target, mask):
    """Compute C-index for survival ranking evaluation."""
    pred   = pred[mask].detach().cpu().numpy()
    target = target[mask].detach().cpu().numpy()
    n_concordant = n_pairs = 0
    for i in range(len(pred)):
        for j in range(len(pred)):
            if target[i] != target[j]:
                n_pairs += 1
                if (pred[i] > pred[j]) == (target[i] > target[j]):
                    n_concordant += 1
    return n_concordant / n_pairs if n_pairs > 0 else 0.5
