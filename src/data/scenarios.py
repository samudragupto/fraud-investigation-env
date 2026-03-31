"""Pre-built fraud scenario templates for each task difficulty."""

import random
import uuid
from typing import Dict, Any, List

from src.data.generator import (
    generate_account,
    generate_fraudulent_transaction,
    _gen_id,
)


def generate_easy_scenario() -> Dict[str, Any]:
    """Single account, single flagged transaction, clear anomaly."""
    account_data = generate_account(
        account_id="ACC_EASY_001",
        name="Arjun Sharma",
        transaction_count=20,
        include_fraud=True,
        fraud_type="high_amount",
        fraud_count=1,
    )

    account = account_data["account"]
    fraud_ids = account_data["fraud_transaction_ids"]

    flagged_txn = None
    for txn in account.recent_transactions:
        if txn.transaction_id in fraud_ids:
            flagged_txn = txn
            break

    return {
        "scenario_id": f"EASY_{uuid.uuid4().hex[:6]}",
        "accounts": {"ACC_EASY_001": account},
        "flagged_transactions": [flagged_txn],
        "ground_truth": {
            flagged_txn.transaction_id: "fraudulent"
        },
        "fraud_pattern": None,
        "fraud_ring": None,
        "key_evidence": [
            "amount_anomaly",
            "foreign_location",
            "unusual_category",
        ],
        "merchants": {
            flagged_txn.merchant_name: {
                "name": flagged_txn.merchant_name,
                "category": flagged_txn.merchant_category,
                "risk_level": "high",
                "country": flagged_txn.location.country,
                "reports": random.randint(5, 50),
            }
        },
    }


def generate_medium_scenario() -> Dict[str, Any]:
    """3 accounts, 8-12 flagged transactions, coordinated pattern."""
    accounts = {}
    all_fraud_ids = []
    ground_truth = {}
    flagged_transactions = []

    for i, (name, fraud_type) in enumerate([
        ("Priya Patel", "card_testing"),
        ("Rahul Verma", "card_testing"),
        ("Sneha Reddy", "money_mule"),
    ]):
        acc_id = f"ACC_MED_{i:03d}"
        acc_data = generate_account(
            account_id=acc_id,
            name=name,
            transaction_count=25,
            include_fraud=True,
            fraud_type=fraud_type,
            fraud_count=random.randint(2, 4),
        )
        accounts[acc_id] = acc_data["account"]
        all_fraud_ids.extend(acc_data["fraud_transaction_ids"])

        for txn in acc_data["account"].recent_transactions:
            if txn.transaction_id in acc_data["fraud_transaction_ids"]:
                ground_truth[txn.transaction_id] = "fraudulent"
                flagged_transactions.append(txn)
            elif random.random() < 0.15:
                ground_truth[txn.transaction_id] = "legitimate"
                txn.flagged_reason = "Unusual timing pattern"
                flagged_transactions.append(txn)

    accounts["ACC_MED_000"].linked_accounts = ["ACC_MED_001"]
    accounts["ACC_MED_001"].linked_accounts = [
        "ACC_MED_000", "ACC_MED_002"
    ]
    accounts["ACC_MED_002"].linked_accounts = ["ACC_MED_001"]

    merchants = {}
    for txn in flagged_transactions:
        if txn.merchant_name not in merchants:
            merchants[txn.merchant_name] = {
                "name": txn.merchant_name,
                "category": txn.merchant_category,
                "risk_level": random.choice(
                    ["low", "medium", "high"]
                ),
                "country": txn.location.country,
                "reports": random.randint(0, 100),
            }

    return {
        "scenario_id": f"MED_{uuid.uuid4().hex[:6]}",
        "accounts": accounts,
        "flagged_transactions": flagged_transactions,
        "ground_truth": ground_truth,
        "fraud_pattern": "card_testing_to_mule_transfer",
        "linked_accounts_truth": {
            "ACC_MED_000": ["ACC_MED_001"],
            "ACC_MED_001": ["ACC_MED_000", "ACC_MED_002"],
        },
        "fraud_ring": None,
        "key_evidence": [
            "rapid_micro_transactions",
            "same_device_fingerprint",
            "temporal_correlation",
            "mule_transfer_pattern",
        ],
        "merchants": merchants,
    }


def generate_hard_scenario() -> Dict[str, Any]:
    """6+ accounts, fraud ring, red herrings."""
    accounts = {}
    ground_truth = {}
    flagged_transactions = []
    ring_members = []

    ring_configs = [
        ("Vikram Singh", "ACC_HARD_000",
         "money_mule", 3, "originator"),
        ("Anita Desai", "ACC_HARD_001",
         "money_mule", 2, "mule_1"),
        ("Karan Mehta", "ACC_HARD_002",
         "money_mule", 2, "mule_2"),
        ("Deepika Nair", "ACC_HARD_003",
         "money_mule", 1, "mule_3"),
        ("Amit Gupta", "ACC_HARD_004",
         "money_mule", 2, "consolidator"),
    ]

    for name, acc_id, fraud_type, fraud_count, role in ring_configs:
        acc_data = generate_account(
            account_id=acc_id,
            name=name,
            transaction_count=30,
            include_fraud=True,
            fraud_type=fraud_type,
            fraud_count=fraud_count,
        )
        accounts[acc_id] = acc_data["account"]
        ring_members.append(acc_id)

        for txn in acc_data["account"].recent_transactions:
            if txn.transaction_id in acc_data[
                "fraud_transaction_ids"
            ]:
                ground_truth[txn.transaction_id] = "fraudulent"
                flagged_transactions.append(txn)

    for i, (name, acc_id) in enumerate([
        ("Neha Joshi", "ACC_HARD_005"),
        ("Sanjay Kumar", "ACC_HARD_006"),
        ("Ritu Agarwal", "ACC_HARD_007"),
    ]):
        acc_data = generate_account(
            account_id=acc_id,
            name=name,
            transaction_count=20,
            include_fraud=False,
        )
        accounts[acc_id] = acc_data["account"]

        for txn in acc_data["account"].recent_transactions:
            if random.random() < 0.2:
                ground_truth[txn.transaction_id] = "legitimate"
                txn.flagged_reason = random.choice([
                    "Unusual amount for this account",
                    "New merchant category",
                    "Flagged by velocity check",
                ])
                flagged_transactions.append(txn)

    ring_structure = {
        "originator": "ACC_HARD_000",
        "mules": [
            "ACC_HARD_001", "ACC_HARD_002", "ACC_HARD_003"
        ],
        "consolidator": "ACC_HARD_004",
        "flow": [
            ("ACC_HARD_000", "ACC_HARD_001"),
            ("ACC_HARD_000", "ACC_HARD_002"),
            ("ACC_HARD_001", "ACC_HARD_003"),
            ("ACC_HARD_002", "ACC_HARD_003"),
            ("ACC_HARD_003", "ACC_HARD_004"),
        ],
    }

    merchants = {}
    for txn in flagged_transactions:
        if txn.merchant_name not in merchants:
            merchants[txn.merchant_name] = {
                "name": txn.merchant_name,
                "category": txn.merchant_category,
                "risk_level": random.choice(
                    ["low", "medium", "high"]
                ),
                "country": txn.location.country,
                "reports": random.randint(0, 200),
            }

    return {
        "scenario_id": f"HARD_{uuid.uuid4().hex[:6]}",
        "accounts": accounts,
        "flagged_transactions": flagged_transactions,
        "ground_truth": ground_truth,
        "fraud_pattern": "money_laundering_ring",
        "fraud_ring": ring_structure,
        "ring_members": ring_members,
        "innocent_accounts": [
            "ACC_HARD_005", "ACC_HARD_006", "ACC_HARD_007"
        ],
        "key_evidence": [
            "wire_transfer_chain",
            "temporal_sequence",
            "amount_splitting",
            "shared_device_fingerprint",
            "geographic_impossibility",
            "new_recipient_pattern",
        ],
        "merchants": merchants,
    }