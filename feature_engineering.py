import pandas as pd
import numpy as np
import re

# Load the deduplicated dataset
df = pd.read_csv("after_dedup.csv")

# Helper functions to parse values
def parse_battery(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+[,.]?\d*)", str(val).replace(",", ""))
    return float(match.group(1)) if match else np.nan

def parse_mp(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else np.nan

def parse_resolution(val):
    if pd.isna(val):
        return np.nan, np.nan
    match = re.findall(r"(\d+)", str(val))
    if len(match) >= 2:
        return int(match[0]), int(match[1])
    return np.nan, np.nan

# Apply parsing functions
df["battery_mAh"] = df["battery_capacity"].apply(parse_battery)
df["rear_camera_mp"] = df["primary_camera_rear"].apply(parse_mp)
res = df["resolution"].apply(parse_resolution)
df["res_width"] = res.apply(lambda x: x[0])
df["res_height"] = res.apply(lambda x: x[1])

# Derived Features
df["spec_score"] = (
    df["ram_capacity"].fillna(0) +
    df["internal_memory"].fillna(0) +
    df["rear_camera_mp"].fillna(0) +
    df["battery_mAh"].fillna(0) / 1000
)

df["value_for_money"] = (df["spec_score"] * 100) / df["price"]
df["performance_index"] = (
    df["ram_capacity"].fillna(0) *
    df["internal_memory"].fillna(0) *
    df["processor_speed"].fillna(0)
) / 1000

df["display_quality"] = (
    df["res_width"].fillna(0) *
    df["res_height"].fillna(0) *
    df["refresh_rate"].fillna(0)
) / 1000

# Feature matrix for similarity search
feature_cols = [
    "price", "rating", "has_5g", "num_cores", "processor_speed", "battery_mAh",
    "ram_capacity", "internal_memory", "screen_size", "refresh_rate",
    "res_width", "res_height", "num_rear_cameras", "num_front_cameras",
    "rear_camera_mp", "value_for_money", "performance_index", "display_quality"
]

feature_matrix = df[feature_cols].fillna(0)

# Save locally if needed
df.to_csv("processed_for_similarity.csv", index=False)
feature_matrix.to_csv("feature_matrix_for_similarity.csv", index=False)