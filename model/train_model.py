import pandas as pd

# Load dataset
df = pd.read_csv("dataset/placement_data.csv")

# Drop unnecessary columns
df = df.drop(columns=[
    "student_id",
    "salary_lpa",
    "specialization"
])

print(df.head())
print("\nColumns after dropping:")
print(df.columns)

