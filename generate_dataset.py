"""
UPI FraudShield — Synthetic Dataset Generator
Generates 500,000 realistic UPI transactions with injected fraud patterns.
Output: upi_transactions.csv
"""

import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TOTAL_TRANSACTIONS   = 500_000
NUM_USERS            = 5_000
NUM_MERCHANTS        = 800
FRAUD_RATE           = 0.032          # 3.2% → ~16,000 fraud txns
START_DATE           = datetime(2024, 1, 1)
END_DATE             = datetime(2024, 12, 31)

# Fraud type target mix (of total fraud)
FRAUD_MIX = {
    "burst":            0.30,
    "night_anomaly":    0.20,
    "location_jump":    0.18,
    "device_switch":    0.15,
    "merchant_anomaly": 0.12,
    "amount_spike":     0.05,
}

INDIAN_CITIES = [
    ("Mumbai",      19.0760,  72.8777),
    ("Delhi",       28.6139,  77.2090),
    ("Bangalore",   12.9716,  77.5946),
    ("Hyderabad",   17.3850,  78.4867),
    ("Chennai",     13.0827,  80.2707),
    ("Kolkata",     22.5726,  88.3639),
    ("Pune",        18.5204,  73.8567),
    ("Ahmedabad",   23.0225,  72.5714),
    ("Jaipur",      26.9124,  75.7873),
    ("Lucknow",     26.8467,  80.9462),
    ("Surat",       21.1702,  72.8311),
    ("Kanpur",      26.4499,  80.3319),
    ("Nagpur",      21.1458,  79.0882),
    ("Patna",       25.5941,  85.1376),
    ("Bhopal",      23.2599,  77.4126),
    ("Indore",      22.7196,  75.8577),
    ("Coimbatore",  11.0168,  76.9558),
    ("Kochi",        9.9312,  76.2673),
    ("Vizag",       17.6868,  83.2185),
    ("Chandigarh",  30.7333,  76.7794),
]

MERCHANT_CATEGORIES = [
    "groceries", "food_delivery", "utilities", "fuel",
    "clothing", "electronics", "travel", "healthcare",
    "education", "entertainment", "crypto_exchange",
    "gambling", "recharge", "insurance", "mutual_funds"
]

# Category weights — groceries most common, crypto rare
CAT_WEIGHTS = [0.22, 0.18, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05,
               0.04, 0.03, 0.015, 0.01, 0.04, 0.03, 0.02]

UPI_APPS = ["GPay", "PhonePe", "Paytm", "BHIM", "AmazonPay", "WhatsApp Pay"]
APP_WEIGHTS = [0.32, 0.35, 0.15, 0.08, 0.07, 0.03]

# ─────────────────────────────────────────────
# BUILD USER PROFILES
# ─────────────────────────────────────────────
print("Building user profiles...")

users = {}
for i in range(NUM_USERS):
    uid = f"U{i+1:05d}"
    home_city = random.choice(INDIAN_CITIES)
    # Each user has 1-3 preferred merchant categories
    preferred_cats = random.choices(MERCHANT_CATEGORIES, weights=CAT_WEIGHTS,
                                    k=random.randint(1, 3))
    # Typical transaction window: most users active 8am-10pm
    active_start = random.randint(6, 10)
    active_end   = random.randint(19, 23)
    avg_amount   = np.random.lognormal(mean=5.5, sigma=1.2)   # ~Rs 100-3000
    std_amount   = avg_amount * random.uniform(0.3, 0.8)
    primary_app  = random.choices(UPI_APPS, weights=APP_WEIGHTS)[0]
    num_devices  = random.choices([1, 2, 3], weights=[0.7, 0.22, 0.08])[0]
    device_ids   = [f"DEV{uuid.uuid4().hex[:8].upper()}" for _ in range(num_devices)]

    users[uid] = {
        "home_city":     home_city,
        "preferred_cats": preferred_cats,
        "active_start":  active_start,
        "active_end":    active_end,
        "avg_amount":    avg_amount,
        "std_amount":    std_amount,
        "primary_app":   primary_app,
        "device_ids":    device_ids,
    }

# ─────────────────────────────────────────────
# BUILD MERCHANT POOL
# ─────────────────────────────────────────────
merchants = []
for i in range(NUM_MERCHANTS):
    cat = random.choices(MERCHANT_CATEGORIES, weights=CAT_WEIGHTS)[0]
    city = random.choice(INDIAN_CITIES)
    merchants.append({
        "merchant_id": f"M{i+1:04d}",
        "category":    cat,
        "city":        city,
    })

# Group merchants by category for fast lookup
merchants_by_cat = {}
for m in merchants:
    merchants_by_cat.setdefault(m["category"], []).append(m)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def random_timestamp(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def legit_timestamp(user):
    """Timestamp biased toward user's active hours."""
    ts = random_timestamp(START_DATE, END_DATE)
    # 80% chance: force hour into user's active window
    if random.random() < 0.80:
        hour = random.randint(user["active_start"], user["active_end"])
        ts = ts.replace(hour=hour, minute=random.randint(0, 59),
                        second=random.randint(0, 59))
    return ts

def legit_amount(user):
    amt = np.random.normal(user["avg_amount"], user["std_amount"])
    return max(1.0, round(amt, 2))

def pick_merchant(user):
    cat = random.choice(user["preferred_cats"])
    pool = merchants_by_cat.get(cat, merchants)
    return random.choice(pool)

def make_txn(uid, ts, amount, merchant, city, device_id, app,
             is_new_device=False, fraud_label=0, fraud_type=None,
             transaction_type=None):
    if transaction_type is None:
        transaction_type = "P2M" if merchant else "P2P"
    hour = ts.hour
    return {
        "transaction_id":   str(uuid.uuid4()),
        "user_id":          uid,
        "amount":           round(amount, 2),
        "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
        "merchant_id":      merchant["merchant_id"] if merchant else f"P2P_{uuid.uuid4().hex[:6]}",
        "merchant_category": merchant["category"] if merchant else "p2p_transfer",
        "device_id":        device_id,
        "location_city":    city[0],
        "location_lat":     round(city[1] + random.uniform(-0.05, 0.05), 4),
        "location_lon":     round(city[2] + random.uniform(-0.05, 0.05), 4),
        "ip_address":       f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
        "transaction_type": transaction_type,
        "upi_app":          app,
        "is_new_device":    is_new_device,
        "hour_of_day":      hour,
        "fraud_label":      fraud_label,
        "fraud_type":       fraud_type if fraud_type else "none",
    }

# ─────────────────────────────────────────────
# GENERATE LEGITIMATE TRANSACTIONS
# ─────────────────────────────────────────────
print("Generating legitimate transactions...")

legit_count = int(TOTAL_TRANSACTIONS * (1 - FRAUD_RATE))
txns = []

user_ids = list(users.keys())

for _ in range(legit_count):
    uid  = random.choice(user_ids)
    user = users[uid]
    ts   = legit_timestamp(user)
    amt  = legit_amount(user)
    merch = pick_merchant(user)
    city  = user["home_city"]
    # 90% use known device, 10% use "new" (legitimate travel etc.)
    if random.random() < 0.90:
        device = random.choice(user["device_ids"])
        is_new = False
    else:
        device = f"DEV{uuid.uuid4().hex[:8].upper()}"
        is_new = True
    app = user["primary_app"]
    txns.append(make_txn(uid, ts, amt, merch, city, device, app, is_new))

# ─────────────────────────────────────────────
# INJECT FRAUD PATTERNS
# ─────────────────────────────────────────────
print("Injecting fraud patterns...")

fraud_target = TOTAL_TRANSACTIONS - legit_count
fraud_txns   = []

# ── BURST FRAUD ──────────────────────────────
burst_count = int(fraud_target * FRAUD_MIX["burst"])
print(f"  Burst fraud: {burst_count} txns")
bursts_needed = burst_count // 8
for _ in range(bursts_needed):
    uid  = random.choice(user_ids)
    user = users[uid]
    base_ts = random_timestamp(START_DATE, END_DATE)
    device = f"DEV{uuid.uuid4().hex[:8].upper()}"
    app  = random.choice(UPI_APPS)
    city = user["home_city"]
    for j in range(8):
        ts  = base_ts + timedelta(seconds=random.randint(j*5, j*15))
        amt = round(random.uniform(100, 2000), 2)
        merch = random.choice(merchants)
        fraud_txns.append(make_txn(uid, ts, amt, merch, city, device, app,
                                   is_new_device=True,
                                   fraud_label=1, fraud_type="burst"))

# ── NIGHT ANOMALY ────────────────────────────
night_count = int(fraud_target * FRAUD_MIX["night_anomaly"])
print(f"  Night anomaly: {night_count} txns")
for _ in range(night_count):
    uid  = random.choice(user_ids)
    user = users[uid]
    ts   = random_timestamp(START_DATE, END_DATE)
    # Force between 1am–4am
    ts   = ts.replace(hour=random.randint(1, 4),
                      minute=random.randint(0, 59),
                      second=random.randint(0, 59))
    amt  = legit_amount(user)
    merch = pick_merchant(user)
    city = user["home_city"]
    device = random.choice(user["device_ids"])
    fraud_txns.append(make_txn(uid, ts, amt, merch, city, device,
                               user["primary_app"],
                               fraud_label=1, fraud_type="night_anomaly"))

# ── LOCATION JUMP ────────────────────────────
loc_count = int(fraud_target * FRAUD_MIX["location_jump"])
print(f"  Location jump: {loc_count} txns (pairs)")
for _ in range(loc_count // 2):
    uid  = random.choice(user_ids)
    user = users[uid]
    city_a, city_b = random.sample(INDIAN_CITIES, 2)
    ts_a = random_timestamp(START_DATE, END_DATE)
    ts_b = ts_a + timedelta(minutes=random.randint(2, 8))   # impossible gap
    device = random.choice(user["device_ids"])
    app    = user["primary_app"]
    amt    = legit_amount(user)
    merch_a = random.choice(merchants)
    merch_b = random.choice(merchants)
    # First txn legit-looking
    fraud_txns.append(make_txn(uid, ts_a, amt, merch_a, city_a, device, app,
                               fraud_label=1, fraud_type="location_jump"))
    fraud_txns.append(make_txn(uid, ts_b, amt, merch_b, city_b, device, app,
                               fraud_label=1, fraud_type="location_jump"))

# ── DEVICE SWITCH ────────────────────────────
dev_count = int(fraud_target * FRAUD_MIX["device_switch"])
print(f"  Device switch: {dev_count} txns")
for _ in range(dev_count // 5):
    uid   = random.choice(user_ids)
    user  = users[uid]
    ts    = random_timestamp(START_DATE, END_DATE)
    new_dev = f"DEV{uuid.uuid4().hex[:8].upper()}"
    city  = user["home_city"]
    app   = user["primary_app"]
    for j in range(5):
        ts_j  = ts + timedelta(seconds=random.randint(j*20, j*40))
        amt   = round(random.uniform(500, 5000), 2)
        merch = random.choice(merchants)
        fraud_txns.append(make_txn(uid, ts_j, amt, merch, city, new_dev, app,
                                   is_new_device=True,
                                   fraud_label=1, fraud_type="device_switch"))

# ── MERCHANT ANOMALY ─────────────────────────
merch_count = int(fraud_target * FRAUD_MIX["merchant_anomaly"])
print(f"  Merchant anomaly: {merch_count} txns")
anomaly_cats = ["crypto_exchange", "gambling"]
for _ in range(merch_count):
    uid   = random.choice(user_ids)
    user  = users[uid]
    # User whose preferred cats are groceries/utilities suddenly hits crypto
    ts    = legit_timestamp(user)
    amt   = round(random.uniform(1000, 15000), 2)
    cat   = random.choice(anomaly_cats)
    pool  = merchants_by_cat.get(cat, merchants)
    merch = random.choice(pool) if pool else random.choice(merchants)
    city  = user["home_city"]
    device = random.choice(user["device_ids"])
    fraud_txns.append(make_txn(uid, ts, amt, merch, city, device,
                               user["primary_app"],
                               fraud_label=1, fraud_type="merchant_anomaly"))

# ── AMOUNT SPIKE ─────────────────────────────
spike_count = int(fraud_target * FRAUD_MIX["amount_spike"])
print(f"  Amount spike: {spike_count} txns")
for _ in range(spike_count):
    uid   = random.choice(user_ids)
    user  = users[uid]
    ts    = legit_timestamp(user)
    # Amount = 10x-100x user average
    amt   = round(user["avg_amount"] * random.uniform(10, 100), 2)
    amt   = min(amt, 100000.0)
    merch = pick_merchant(user)
    city  = user["home_city"]
    device = random.choice(user["device_ids"])
    fraud_txns.append(make_txn(uid, ts, amt, merch, city, device,
                               user["primary_app"],
                               fraud_label=1, fraud_type="amount_spike"))

# ─────────────────────────────────────────────
# COMBINE & SHUFFLE
# ─────────────────────────────────────────────
print("Combining and shuffling...")
all_txns = txns + fraud_txns
random.shuffle(all_txns)

df = pd.DataFrame(all_txns)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ─────────────────────────────────────────────
# ADD ENGINEERED FEATURES
# ─────────────────────────────────────────────
print("Engineering features...")

df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# Time since last transaction (seconds)
df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)
df["time_since_last_tx"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds().fillna(99999)

# Rolling transaction count (last 2 minutes)
df["tx_count_2min"] = 0
df["tx_count_5min"] = 0

# Vectorised rolling velocity per user
df_sorted = df.copy()
df_sorted["ts_epoch"] = df_sorted["timestamp"].astype(np.int64) // 10**9

for uid_val, grp in df_sorted.groupby("user_id"):
    idx   = grp.index.tolist()
    times = grp["ts_epoch"].values
    v2 = np.zeros(len(times), dtype=int)
    v5 = np.zeros(len(times), dtype=int)
    for i in range(len(times)):
        v2[i] = int(np.sum((times[i] - times[:i]) <= 120))
        v5[i] = int(np.sum((times[i] - times[:i]) <= 300))
    df_sorted.loc[idx, "tx_count_2min"] = v2
    df_sorted.loc[idx, "tx_count_5min"] = v5

df["tx_count_2min"] = df_sorted["tx_count_2min"]
df["tx_count_5min"] = df_sorted["tx_count_5min"]

# User amount z-score
user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).reset_index()
user_stats.columns = ["user_id", "user_mean_amount", "user_std_amount"]
user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(1).replace(0, 1)
df = df.merge(user_stats, on="user_id", how="left")
df["amount_zscore"] = (df["amount"] - df["user_mean_amount"]) / df["user_std_amount"]

# Day of week
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)

# Clean up helper cols
df.drop(columns=["prev_ts"], inplace=True)

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
out_path = "upi_transactions.csv"
df.to_csv(out_path, index=False)

print(f"\n{'='*55}")
print(f"  Dataset saved → {out_path}")
print(f"  Total rows    : {len(df):,}")
print(f"  Fraud rows    : {df['fraud_label'].sum():,}  ({df['fraud_label'].mean()*100:.2f}%)")
print(f"  Fraud types   :")
for ft, cnt in df[df["fraud_label"]==1]["fraud_type"].value_counts().items():
    print(f"    {ft:<25} {cnt:,}")
print(f"  Columns       : {len(df.columns)}")
print(f"  Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"{'='*55}")
