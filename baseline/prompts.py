"""Prompt templates for the baseline agent."""

TASK_PROMPTS = {
    "single_transaction_classification": (
        "You are investigating a single flagged transaction. "
        "Query the account history, check the merchant profile, "
        "verify geolocation consistency, then classify the "
        "transaction and submit your investigation."
    ),
    "multi_account_pattern_detection": (
        "You are investigating flagged transactions across "
        "multiple accounts. Look for coordinated patterns like "
        "card testing or money mule activity. Cross-reference "
        "accounts, classify all flagged transactions, and submit."
    ),
    "fraud_ring_detection": (
        "You are investigating a potential fraud ring across "
        "6+ accounts. Map the money flow, identify all ring "
        "members, separate red herrings from real fraud, "
        "write a detailed investigation summary, and submit."
    ),
}