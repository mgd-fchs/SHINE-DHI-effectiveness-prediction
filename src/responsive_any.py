import pandas as pd

# Load the CSV
df = pd.read_csv("./data/intervention_time/responsiveness_study2.csv")

# Recalculate 'responsive': 1 if difference_drinks_occasions < 0, else 0
df["responsive"] = (df["difference_drinks_occasions"] < 0).astype(int)

# Save the updated DataFrame to a new CSV
df.to_csv("./data/intervention_time/responsiveness_study2_neg_any.csv", index=False)