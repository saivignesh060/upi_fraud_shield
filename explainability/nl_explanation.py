"""
UPI FraudShield — Natural Language Explanation Generator
Converts SHAP feature contributions into human-readable alert messages.
Template-based engine matching PRD examples exactly.
"""

from typing import Optional


# ─────────────────────────────────────────────────────────────
# FRAUD TYPE TEMPLATES
# ─────────────────────────────────────────────────────────────

TEMPLATES = {
    "burst": (
        "User made {tx_count} transactions in {seconds} seconds. "
        "Normal cadence is {normal_cadence}. "
        "Velocity {multiplier}x above baseline."
    ),
    "night_anomaly": (
        "Transaction at {hour}:{minute} {am_pm} detected. "
        "User's typical activity window is {active_start}:00–{active_end}:00. "
        "This is {hours_outside} hours outside their normal pattern."
    ),
    "location_jump": (
        "Transaction in {city_b} detected {minutes} minutes after a transaction in {city_a} "
        "({distance_km:,} km apart — physically impossible at normal travel speed)."
    ),
    "device_switch": (
        "New unrecognized device ({device_id}) initiated {tx_count} transactions in {minutes} minutes. "
        "User's known devices: {known_device_count}. "
        "This device has never been seen before."
    ),
    "merchant_anomaly": (
        "User's {history_days}-day history shows only {usual_categories} payments. "
        "This is the first transaction to a {flagged_category} merchant."
    ),
    "amount_spike": (
        "Transaction amount ₹{amount:,} is {multiplier}x above this user's "
        "{history_days}-day average of ₹{avg_amount:,}."
    ),
    "generic": (
        "Transaction flagged with risk score {risk_score}/100. "
        "Top contributing factor: {top_feature}. "
        "Confidence: {confidence}%."
    ),
}

# Short one-line summaries for the alert card header
SHORT_SUMMARIES = {
    "burst": "Abnormal transaction velocity detected — {tx_count} txns in {seconds}s",
    "night_anomaly": "Unusual activity time — transaction at {hour}:{minute} {am_pm}",
    "location_jump": "Geo-velocity anomaly — {city_a} → {city_b} in {minutes} min",
    "device_switch": "Unknown device initiated rapid transactions",
    "merchant_anomaly": "Unusual merchant category — {flagged_category} for first time",
    "amount_spike": "Amount ₹{amount:,} is {multiplier}x above user average",
    "generic": "Fraud pattern detected — Risk Score {risk_score}/100",
}


def generate_explanation(
    fraud_type: str,
    transaction: dict,
    user_profile: dict = None,
    shap_features: list = None,
    risk_score: float = 0.0,
) -> dict:
    """
    Generate a human-readable explanation for a flagged transaction.

    Args:
        fraud_type: One of burst/night_anomaly/location_jump/device_switch/merchant_anomaly/amount_spike/generic
        transaction: Raw transaction dict
        user_profile: User behavioral profile dict (optional)
        shap_features: Top-3 SHAP features from shap_explainer (optional)
        risk_score: Final composite risk score (0-100)

    Returns:
        dict with "short_summary", "full_explanation", "fraud_type", "top_feature"
    """
    up = user_profile or {}
    t = transaction

    try:
        if fraud_type == "burst":
            tx_count = t.get("tx_count_2min", 8)
            seconds = int(tx_count * 12)  # approx
            normal = up.get("avg_daily_tx", 3)
            mult = round(tx_count * 120 / max(normal, 1), 0) if normal > 0 else tx_count * 10

            full = TEMPLATES["burst"].format(
                tx_count=tx_count,
                seconds=seconds,
                normal_cadence=f"{int(normal)} per day",
                multiplier=int(mult),
            )
            short = SHORT_SUMMARIES["burst"].format(tx_count=tx_count, seconds=seconds)

        elif fraud_type == "night_anomaly":
            hour = t.get("hour_of_day", 2)
            minute = t.get("minute", 30) if "minute" in t else 0
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour if hour <= 12 else hour - 12
            active_start = up.get("active_start", 9)
            active_end = up.get("active_end", 21)
            hours_outside = min(
                abs(hour - active_start),
                abs(hour - active_end)
            )

            full = TEMPLATES["night_anomaly"].format(
                hour=display_hour,
                minute=f"{minute:02d}",
                am_pm=am_pm,
                active_start=active_start,
                active_end=active_end,
                hours_outside=hours_outside,
            )
            short = SHORT_SUMMARIES["night_anomaly"].format(
                hour=display_hour, minute=f"{minute:02d}", am_pm=am_pm
            )

        elif fraud_type == "location_jump":
            city_a = t.get("prev_city", "Unknown City")
            city_b = t.get("location_city", "Unknown City")
            minutes = t.get("minutes_gap", 4)
            distance_km = _estimate_distance(city_a, city_b)

            full = TEMPLATES["location_jump"].format(
                city_b=city_b, city_a=city_a, minutes=minutes, distance_km=distance_km
            )
            short = SHORT_SUMMARIES["location_jump"].format(
                city_a=city_a, city_b=city_b, minutes=minutes
            )

        elif fraud_type == "device_switch":
            dev_id = t.get("device_id", "UNKNOWN")[:12] + "..."
            tx_count = t.get("tx_count_5min", 5)
            minutes = round(tx_count * 0.5, 1)
            known_count = len(up.get("known_devices", []))

            full = TEMPLATES["device_switch"].format(
                device_id=dev_id,
                tx_count=tx_count,
                minutes=minutes,
                known_device_count=known_count if known_count > 0 else "1-2",
            )
            short = SHORT_SUMMARIES["device_switch"]

        elif fraud_type == "merchant_anomaly":
            usual = ", ".join(up.get("preferred_cats", ["groceries", "utilities"])[:2])
            flagged_cat = t.get("merchant_category", "crypto_exchange").replace("_", " ")

            full = TEMPLATES["merchant_anomaly"].format(
                history_days=60,
                usual_categories=usual,
                flagged_category=flagged_cat,
            )
            short = SHORT_SUMMARIES["merchant_anomaly"].format(flagged_category=flagged_cat)

        elif fraud_type == "amount_spike":
            amount = int(t.get("amount", 0))
            avg_amount = int(up.get("avg_amount", 150))
            multiplier = round(amount / max(avg_amount, 1), 0)

            full = TEMPLATES["amount_spike"].format(
                amount=amount,
                multiplier=int(multiplier),
                history_days=30,
                avg_amount=avg_amount,
            )
            short = SHORT_SUMMARIES["amount_spike"].format(
                amount=amount, multiplier=int(multiplier)
            )

        else:
            top_feature = shap_features[0]["display_name"] if shap_features else "Unknown"
            confidence = round(risk_score, 0)
            full = TEMPLATES["generic"].format(
                risk_score=int(risk_score),
                top_feature=top_feature,
                confidence=int(confidence),
            )
            short = SHORT_SUMMARIES["generic"].format(risk_score=int(risk_score))

    except Exception as e:
        full = f"Suspicious transaction detected. Risk Score: {int(risk_score)}/100."
        short = f"Transaction flagged — Risk Score {int(risk_score)}/100"

    # Extract top feature from SHAP if available
    top_feature_name = "Unknown"
    if shap_features and len(shap_features) > 0:
        top_feature_name = shap_features[0].get("display_name", "Unknown")

    return {
        "fraud_type": fraud_type,
        "short_summary": short,
        "full_explanation": full,
        "top_feature": top_feature_name,
        "risk_score": round(risk_score, 1),
    }


def _estimate_distance(city_a: str, city_b: str) -> int:
    """Rough distance estimates between Indian cities in km."""
    CITY_DISTANCES = {
        ("Mumbai", "Delhi"): 1415, ("Delhi", "Mumbai"): 1415,
        ("Hyderabad", "Delhi"): 1490, ("Delhi", "Hyderabad"): 1490,
        ("Mumbai", "Bangalore"): 984, ("Bangalore", "Mumbai"): 984,
        ("Chennai", "Delhi"): 2180, ("Delhi", "Chennai"): 2180,
        ("Kolkata", "Mumbai"): 2050, ("Mumbai", "Kolkata"): 2050,
        ("Hyderabad", "Mumbai"): 710, ("Mumbai", "Hyderabad"): 710,
        ("Bangalore", "Hyderabad"): 570, ("Hyderabad", "Bangalore"): 570,
        ("Chennai", "Bangalore"): 350, ("Bangalore", "Chennai"): 350,
        ("Delhi", "Kolkata"): 1500, ("Kolkata", "Delhi"): 1500,
        ("Pune", "Mumbai"): 150, ("Mumbai", "Pune"): 150,
    }
    key = (city_a, city_b)
    return CITY_DISTANCES.get(key, 800)  # default 800km if unknown


if __name__ == "__main__":
    # Test explanations for each fraud type
    test_txn = {"amount": 42500, "hour_of_day": 2, "tx_count_2min": 9,
                "location_city": "Delhi", "prev_city": "Hyderabad",
                "device_id": "DEVABCDEF1234", "merchant_category": "crypto_exchange",
                "minutes_gap": 4}
    test_profile = {"avg_amount": 150, "active_start": 9, "active_end": 21,
                    "preferred_cats": ["groceries", "utilities"],
                    "known_devices": ["DEV123", "DEV456"], "avg_daily_tx": 3}

    for fraud_type in ["burst", "night_anomaly", "location_jump", "device_switch",
                        "merchant_anomaly", "amount_spike"]:
        result = generate_explanation(fraud_type, test_txn, test_profile, risk_score=75.0)
        print(f"\n[{fraud_type.upper()}]")
        print(f"  Short: {result['short_summary']}")
        print(f"  Full : {result['full_explanation']}")
