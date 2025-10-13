import pandas as pd
import re

def normalize_model(model):
    # Remove content in parentheses, extra spaces, and lowercase
    model = re.sub(r'\([^)]*\)', '', str(model))  # remove text in parentheses
    model = model.replace('+', 'plus')            # treat '+' as 'plus'
    model = model.replace('-', ' ')               # treat '-' as space
    model = model.replace('_', ' ')               # treat '_' as space
    model = model.lower()                         # lowercase everything
    model = re.sub(r'\s+', ' ', model).strip()    # remove extra spaces
    return model

# Load the dataset
df = pd.read_csv("final_dataset (1).csv")

# Save original dataset
df.to_csv("before_dedup.csv", index=False)

# Normalize model names
df["model_normalized"] = df["model"].apply(normalize_model)

# Remove duplicates based on brand and normalized model
df_dedup = df.drop_duplicates(subset=["brand_name", "model_normalized"], keep="first")

# Drop helper column
df_dedup = df_dedup.drop(columns=["model_normalized"])

# Save deduplicated dataset
df_dedup.to_csv("after_dedup.csv", index=False)