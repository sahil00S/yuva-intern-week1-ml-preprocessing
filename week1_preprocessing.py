"""
YuvaIntern Week 1
Python for Machine Learning & Data Preprocessing

Project:
Titanic Passenger Dataset — Data Cleaning, Encoding,
Normalization and Exploratory Data Analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "titanic.csv"
OUTPUT_FILE = DATA_DIR / "titanic_cleaned.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("YUVAINTERN WEEK 1 - TITANIC PREPROCESSING")
print("=" * 60)

print("\n1. INITIAL DATA INSPECTION")
print("-" * 60)

print("Raw shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())


# ============================================================
# 2. MISSING VALUE VISUALIZATION
# ============================================================

missing = df.isna().sum()
missing = missing[missing > 0]

plt.figure(figsize=(8, 5))

missing.sort_values(ascending=False).plot(kind="bar")

plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "missing_values.png",
    dpi=160
)

plt.close()


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

print("\n2. FEATURE SELECTION")
print("-" * 60)

columns_to_drop = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

print("Dropping:")
for column in columns_to_drop:
    print("-", column)

df = df.drop(columns=columns_to_drop)


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\n3. HANDLING MISSING VALUES")
print("-" * 60)

# Age → median
age_median = df["Age"].median()

print("Age missing values:", df["Age"].isna().sum())
print("Age median:", age_median)

df["Age"] = df["Age"].fillna(age_median)


# Embarked → mode
embarked_mode = df["Embarked"].mode()[0]

print("Embarked missing values:", df["Embarked"].isna().sum())
print("Embarked mode:", embarked_mode)

df["Embarked"] = df["Embarked"].fillna(embarked_mode)


# ============================================================
# 5. CATEGORICAL ENCODING
# ============================================================

print("\n4. CATEGORICAL ENCODING")
print("-" * 60)

# Binary encode Sex
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

print("Sex:")
print("male   -> 0")
print("female -> 1")


# One-hot encode Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    dtype=int
)

print("\nEmbarked converted using one-hot encoding.")

print("\nColumns after encoding:")
print(df.columns.tolist())


# ============================================================
# 6. NUMERICAL NORMALIZATION
# ============================================================

print("\n5. NUMERICAL NORMALIZATION")
print("-" * 60)

numeric_columns = [
    "Age",
    "Fare",
    "SibSp",
    "Parch"
]

for column in numeric_columns:

    minimum = df[column].min()
    maximum = df[column].max()

    if maximum != minimum:

        df[column] = (
            df[column] - minimum
        ) / (
            maximum - minimum
        )

    print(f"{column}: Min-Max normalization applied")


# ============================================================
# 7. FINAL MISSING VALUE CHECK
# ============================================================

remaining_missing = df.isna().sum().sum()

print("\n6. FINAL DATA QUALITY CHECK")
print("-" * 60)

print("Remaining missing values:", remaining_missing)

assert remaining_missing == 0, (
    "Missing values remain after preprocessing."
)


# ============================================================
# 8. SURVIVAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 4))

df["Survived"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Target Distribution: Survival")
plt.xlabel("Survived")
plt.ylabel("Passenger Count")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "survival_distribution.png",
    dpi=160
)

plt.close()


# ============================================================
# 9. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 4))

plt.hist(
    df["Age"],
    bins=20
)

plt.title("Age Distribution After Normalization")
plt.xlabel("Normalized Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "age_distribution.png",
    dpi=160
)

plt.close()


# ============================================================
# 10. SURVIVAL BY PASSENGER CLASS
# ============================================================

class_survival = (
    df.groupby("Pclass")["Survived"]
    .mean()
)

print("\nSurvival rate by passenger class:")
print(class_survival)


plt.figure(figsize=(7, 4))

class_survival.plot(
    kind="bar"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "survival_by_class.png",
    dpi=160
)

plt.close()


# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

correlation = df.corr(numeric_only=True)

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "correlation_heatmap.png",
    dpi=160
)

plt.close()


# ============================================================
# 12. EXPORT CLEANED DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n7. EXPORT")
print("-" * 60)

print("Cleaned shape:", df.shape)

print("Saved:", OUTPUT_FILE)

print("\nFinal columns:")
print(df.columns.tolist())


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n8. PROJECT SUMMARY")
print("-" * 60)

print("Raw dataset shape:      (891, 12)")
print("Final dataset shape:    ", df.shape)
print("Remaining missing:      ", remaining_missing)

print("\nFigures generated:")

for file in sorted(FIGURES_DIR.glob("*.png")):
    print("-", file.name)

print("\n" + "=" * 60)
print("WEEK 1 PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)