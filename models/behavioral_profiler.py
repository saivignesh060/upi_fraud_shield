import os
import joblib
import pandas as pd
import numpy as np
from time import time

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profiles.pkl")

class UserProfileEngine:
    """
    M3 - Behavioral Profiler Layer.
    Maintains a dictionary of user profiles with their historical baselines.
    """
    def __init__(self):
        self.profiles = {}
        # Default profile for cold-start users
        self.default_profile = {
            "user_mean_amount": 500.0,
            "user_std_amount": 300.0,
            "known_devices": set(),
            "known_cities": set(),
            "preferred_merchant_categories": set(),
            "tx_count": 0,
            "sum_amount": 0.0,
            "sum_sq_amount": 0.0,
            "typical_hours": {} # hour -> count
        }

    def load(self, path=PROFILE_PATH):
        if os.path.exists(path):
            self.profiles = joblib.load(path)
            print(f"[Profiler] Loaded {len(self.profiles)} user profiles from {path}")
        else:
            print(f"[Profiler] No profiles found. Starting fresh.")
            self.profiles = {}

    def save(self, path=PROFILE_PATH):
        joblib.dump(self.profiles, path)
        print(f"[Profiler] Saved {len(self.profiles)} user profiles to {path}")

    def get_profile(self, user_id):
        # Return a copy so the caller doesn't accidentally mutate the dict
        # Sets and dicts inside still hold references, but it's okay for read
        if user_id in self.profiles:
            return self.profiles[user_id]
        return self.default_profile

    def update_profile(self, user_id, txn_dict):
        """
        Incrementally update the user profile with a new transaction.
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "user_mean_amount": 0.0,
                "user_std_amount": 0.0,
                "known_devices": set(),
                "known_cities": set(),
                "preferred_merchant_categories": set(),
                "tx_count": 0,
                "sum_amount": 0.0,
                "sum_sq_amount": 0.0,
                "typical_hours": {}
            }
        
        p = self.profiles[user_id]
        amt = float(txn_dict.get("amount", 0.0))
        
        # Update running tallies
        p["tx_count"] += 1
        p["sum_amount"] += amt
        p["sum_sq_amount"] += amt * amt
        
        # Welford's method approximation for running std deviation
        p["user_mean_amount"] = p["sum_amount"] / p["tx_count"]
        if p["tx_count"] > 1:
            variance = (p["sum_sq_amount"] - (p["sum_amount"]**2 / p["tx_count"])) / (p["tx_count"] - 1)
            p["user_std_amount"] = float(np.sqrt(max(0, variance)))
        else:
            p["user_std_amount"] = 0.0
            
        # Update categorical sets
        if "device_id" in txn_dict:
            p["known_devices"].add(txn_dict["device_id"])
        elif "is_new_device" in txn_dict:
            # If no device_id but we have info, we can't really update the set by ID,
            # but usually transaction has device_id in real life, in our synthetic dataset maybe not.
            pass

        if "location_city" in txn_dict:
            p["known_cities"].add(txn_dict["location_city"])
            
        if "merchant_category" in txn_dict:
            p["preferred_merchant_categories"].add(txn_dict["merchant_category"])
            
        # Update histograms
        hour = txn_dict.get("hour_of_day")
        if hour is not None:
            p["typical_hours"][hour] = p["typical_hours"].get(hour, 0) + 1


def build_profiles_from_csv(csv_path: str):
    """
    Utility to bootstrap the user profiles from the M1 dataset.
    """
    print(f"[Profiler] Building initial profiles from {csv_path}...")
    t0 = time()
    df = pd.read_csv(csv_path)
    
    engine = UserProfileEngine()
    
    # We will build profiles for only the first 5000 users or full dataset.
    # The dataset has 500k rows, iterating over them in pandas is slow but manageable (takes ~10-20 sec)
    for row in df.itertuples(index=False):
        # row._asdict() is fast
        txn = row._asdict()
        engine.update_profile(txn['user_id'], txn)
        
    engine.save()
    print(f"[Profiler] Initial profile building complete in {time() - t0:.2f} seconds.")

if __name__ == "__main__":
    # When run directly, build the user_profiles.pkl
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "upi_transactions.csv")
    build_profiles_from_csv(csv_path)
