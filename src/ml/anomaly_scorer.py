"""PyTorch autoencoder for transaction anomaly scoring."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any


class TransactionAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyScorer:
    def __init__(self):
        self.model = TransactionAutoencoder(input_dim=8)
        self.model.eval()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_features(
        self, transaction: Dict[str, Any]
    ) -> torch.Tensor:
        features = [
            float(transaction.get("amount", 0)) / 100000.0,
            float(
                transaction.get("location", {}).get(
                    "latitude", 0
                )
            ) / 90.0,
            float(
                transaction.get("location", {}).get(
                    "longitude", 0
                )
            ) / 180.0,
            1.0 if transaction.get("channel") == "online"
            else 0.0,
            1.0 if transaction.get("channel") == "wire"
            else 0.0,
            1.0 if transaction.get("channel") == "atm"
            else 0.0,
            1.0 if transaction.get("merchant_category")
            in ["crypto_exchange", "gambling", "money_transfer"]
            else 0.0,
            len(
                transaction.get("flagged_reason", "")
            ) / 100.0,
        ]
        return torch.tensor(features, dtype=torch.float32)

    def score(
        self, transaction: Dict[str, Any]
    ) -> float:
        features = self.extract_features(transaction)
        with torch.no_grad():
            reconstructed = self.model(
                features.unsqueeze(0)
            )
            loss = nn.MSELoss()(
                reconstructed, features.unsqueeze(0)
            )
            anomaly_score = float(loss.item())
        return min(anomaly_score * 10.0, 1.0)

    def batch_score(
        self, transactions: List[Dict[str, Any]]
    ) -> List[float]:
        return [self.score(t) for t in transactions]