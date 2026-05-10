import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("aircraft_vibration.csv")

# ============================================
# FEATURE ENGINEERING
# ============================================

df["thermal_stress"] = (
    df["sensor_temperature"] * df["load_factor"]
)

df["lubrication_risk"] = (
    df["vibration_rms"] / (df["oil_pressure"] + 1e-6)
)

df["mechanical_stress"] = (
    df["rpm"] * df["load_factor"]
)

df["degradation_score"] = (
    df["vibration_rms"] +
    df["sensor_temperature"]/100 -
    df["oil_pressure"]/40
)

# ============================================
# ROLLING FEATURES
# ============================================

df = df.sort_values("time_step")

df["vibration_roll_mean"] = (
    df["vibration_rms"]
    .rolling(10)
    .mean()
)

df["temp_roll_mean"] = (
    df["sensor_temperature"]
    .rolling(10)
    .mean()
)

df = df.dropna()

# ============================================
# FEATURES & TARGET
# ============================================

X = df.drop(columns=["risk_label", "time_step"])

y = df["risk_label"]

# ============================================
# MODEL
# ============================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# ============================================
# TRAIN
# ============================================

model.fit(X, y)

# ============================================
# SAVE MODEL
# ============================================

pickle.dump(
    model,
    open("model.pkl", "wb")
)

print("Model Saved Successfully")