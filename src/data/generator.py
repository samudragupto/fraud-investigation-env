"""Synthetic fraud data generator."""

import random
import string
import uuid
from typing import List, Dict, Any

from src.models import (
    TransactionDetail, AccountSummary, Location
)

CITIES = [
    {"city": "Mumbai", "country": "India",
     "latitude": 19.076, "longitude": 72.877},
    {"city": "Delhi", "country": "India",
     "latitude": 28.613, "longitude": 77.209},
    {"city": "Bangalore", "country": "India",
     "latitude": 12.971, "longitude": 77.594},
    {"city": "London", "country": "UK",
     "latitude": 51.507, "longitude": -0.127},
    {"city": "New York", "country": "USA",
     "latitude": 40.712, "longitude": -74.006},
    {"city": "Lagos", "country": "Nigeria",
     "latitude": 6.524, "longitude": 3.379},
    {"city": "Dubai", "country": "UAE",
     "latitude": 25.204, "longitude": 55.270},
]

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "fuel",
    "clothing", "pharmacy", "travel", "entertainment",
    "utilities", "jewelry", "crypto_exchange",
    "money_transfer", "gambling",
]

MERCHANTS = {
    "grocery": ["FreshMart", "BigBasket", "DMart"],
    "electronics": ["Croma", "Reliance Digital", "BestBuy"],
    "restaurant": ["Swiggy", "Zomato", "UberEats"],
    "fuel": ["IndianOil", "BPCL", "HP Petrol"],
    "clothing": ["Myntra", "Ajio", "Zara"],
    "pharmacy": ["PharmEasy", "1mg", "Apollo"],
    "travel": ["MakeMyTrip", "Booking.com", "Expedia"],
    "entertainment": ["BookMyShow", "Netflix", "Spotify"],
    "utilities": ["Jio", "Airtel", "BESCOM"],
    "jewelry": ["Tanishq", "Kalyan", "GoldBoutique"],
    "crypto_exchange": ["CoinSwitch", "WazirX", "BinanceP2P"],
    "money_transfer": ["WesternUnion", "PayTM", "QuickRemit"],
    "gambling": ["Dream11", "BetOnline", "LuckyDraw"],
}

NAMES = [
    "Arjun Sharma", "Priya Patel", "Rahul Verma",
    "Sneha Reddy", "Vikram Singh", "Anita Desai",
    "Karan Mehta", "Deepika Nair", "Amit Gupta",
    "Neha Joshi",
]


def _gen_id(prefix: str = "TXN") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8].upper()}"


def _random_location(
    exclude_countries: List[str] | None = None
) -> Location:
    choices = CITIES
    if exclude_countries:
        choices = [
            c for c in CITIES
            if c["country"] not in exclude_countries
        ]
    if not choices:
        choices = CITIES
    loc = random.choice(choices)
    return Location(**loc)


def _random_timestamp(
    base: str = "2026-03-",
    day_range: tuple = (1, 28)
) -> str:
    day = random.randint(*day_range)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{base}{day:02d}T{hour:02d}:{minute:02d}:00Z"


def generate_normal_transaction(
    account_categories: List[str] | None = None,
    amount_range: tuple = (100, 5000),
) -> TransactionDetail:
    if account_categories is None:
        account_categories = [
            "grocery", "electronics", "restaurant",
            "fuel", "clothing", "pharmacy",
            "utilities", "entertainment",
        ]
    category = random.choice(account_categories)
    merchant = random.choice(MERCHANTS[category])
    return TransactionDetail(
        transaction_id=_gen_id("TXN"),
        amount=round(random.uniform(*amount_range), 2),
        currency="INR",
        merchant_name=merchant,
        merchant_category=category,
        timestamp=_random_timestamp(),
        location=_random_location(exclude_countries=None),
        channel=random.choice(["online", "pos"]),
        description=f"Purchase at {merchant}",
        flagged_reason="",
    )


def generate_fraudulent_transaction(
    fraud_type: str = "high_amount",
    base_amount: float = 2000.0,
) -> TransactionDetail:
    if fraud_type == "high_amount":
        amount = base_amount * random.uniform(15, 30)
        category = random.choice(
            ["jewelry", "electronics", "crypto_exchange"]
        )
        merchant = random.choice(MERCHANTS[category])
        return TransactionDetail(
            transaction_id=_gen_id("TXN"),
            amount=round(amount, 2),
            currency="INR",
            merchant_name=merchant,
            merchant_category=category,
            timestamp=_random_timestamp(),
            location=_random_location(
                exclude_countries=["India"]
            ),
            channel="online",
            description=f"High value purchase at {merchant}",
            flagged_reason="Amount exceeds 10x account average",
        )
    elif fraud_type == "card_testing":
        return TransactionDetail(
            transaction_id=_gen_id("TXN"),
            amount=round(random.uniform(1, 10), 2),
            currency="INR",
            merchant_name=random.choice(
                MERCHANTS["entertainment"]
            ),
            merchant_category="entertainment",
            timestamp=_random_timestamp(),
            location=_random_location(),
            channel="online",
            description="Small test transaction",
            flagged_reason="Rapid succession of micro-transactions",
        )
    elif fraud_type == "money_mule":
        return TransactionDetail(
            transaction_id=_gen_id("TXN"),
            amount=round(random.uniform(25000, 100000), 2),
            currency="INR",
            merchant_name=random.choice(
                MERCHANTS["money_transfer"]
            ),
            merchant_category="money_transfer",
            timestamp=_random_timestamp(),
            location=_random_location(),
            channel="wire",
            description="Wire transfer to external account",
            flagged_reason="Large wire to new recipient",
        )
    else:
        return generate_normal_transaction(
            amount_range=(50000, 200000)
        )


def generate_account(
    account_id: str | None = None,
    name: str | None = None,
    transaction_count: int = 15,
    include_fraud: bool = False,
    fraud_type: str = "high_amount",
    fraud_count: int = 1,
) -> Dict[str, Any]:
    if account_id is None:
        account_id = _gen_id("ACC")
    if name is None:
        name = random.choice(NAMES)

    normal_categories = random.sample(
        ["grocery", "restaurant", "fuel",
         "clothing", "pharmacy", "utilities"],
        k=random.randint(3, 5),
    )
    avg_amount = random.uniform(1000, 8000)

    transactions = []
    for _ in range(transaction_count):
        txn = generate_normal_transaction(
            account_categories=normal_categories,
            amount_range=(avg_amount * 0.3, avg_amount * 2.0),
        )
        transactions.append(txn)

    fraud_txn_ids = []
    if include_fraud:
        for _ in range(fraud_count):
            fraud_txn = generate_fraudulent_transaction(
                fraud_type=fraud_type,
                base_amount=avg_amount,
            )
            transactions.append(fraud_txn)
            fraud_txn_ids.append(fraud_txn.transaction_id)

    account = AccountSummary(
        account_id=account_id,
        holder_name=name,
        account_age_days=random.randint(90, 2000),
        average_monthly_volume=round(
            avg_amount * transaction_count / 2, 2
        ),
        recent_transactions=transactions,
        risk_score=round(random.uniform(0.1, 0.4), 2),
        linked_accounts=[],
    )

    return {
        "account": account,
        "fraud_transaction_ids": fraud_txn_ids,
        "normal_categories": normal_categories,
        "average_amount": avg_amount,
    }