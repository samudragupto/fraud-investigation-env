"""Transaction feature engineering using PyTorch."""

import torch
import numpy as np
from typing import List, Dict, Any


class FeatureExtractor:
    def compute_velocity_features(
        self,
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not transactions:
            return {
                "mean_amount": 0.0,
                "std_amount": 0.0,
                "max_amount": 0.0,
                "transaction_frequency": 0.0,
                "amount_acceleration": 0.0,
            }

        amounts = torch.tensor(
            [t.get("amount", 0.0) for t in transactions],
            dtype=torch.float32,
        )

        mean_amt = float(torch.mean(amounts))
        std_amt = float(torch.std(amounts)) if len(amounts) > 1 else 0.0
        max_amt = float(torch.max(amounts))

        frequency = len(transactions) / 30.0

        if len(amounts) > 2:
            diffs = amounts[1:] - amounts[:-1]
            acceleration = float(torch.mean(torch.abs(diffs)))
        else:
            acceleration = 0.0

        return {
            "mean_amount": round(mean_amt, 2),
            "std_amount": round(std_amt, 2),
            "max_amount": round(max_amt, 2),
            "transaction_frequency": round(frequency, 4),
            "amount_acceleration": round(acceleration, 2),
        }

    def compute_category_distribution(
        self,
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not transactions:
            return {}

        categories = {}
        for t in transactions:
            cat = t.get("merchant_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        total = len(transactions)
        return {
            cat: round(count / total, 4)
            for cat, count in categories.items()
        }

    def compute_location_entropy(
        self,
        transactions: List[Dict[str, Any]],
    ) -> float:
        if not transactions:
            return 0.0

        countries = {}
        for t in transactions:
            country = t.get(
                "location", {}
            ).get("country", "unknown")
            countries[country] = countries.get(country, 0) + 1

        total = len(transactions)
        probs = torch.tensor(
            [c / total for c in countries.values()],
            dtype=torch.float32,
        )
        entropy = float(-torch.sum(probs * torch.log2(probs + 1e-10)))
        return round(entropy, 4)