import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "titanic.csv")
CHART_DIR = os.path.join(BASE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA ONCE
# ============================================================

print("=" * 70)
print("LOADING TITANIC DATASET")
print("=" * 70)

df = sns.load_dataset("titanic")

# Required offline fallback immediately after loading
df.to_csv(CSV_PATH, index=False)

print("\nShape:", df.shape)
print("\nINFO:")
df.info()

print("\nDESCRIBE:")
print(df.describe(include="all").to_string())

print("\nMISSING VALUE PERCENTAGES:")
missing = df.isnull().mean() * 100
missing = missing[missing > 0]

for column, percentage in missing.items():
    print(f"{column}: {percentage:.2f}%")


# ============================================================
# 2. CLEANING
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)

# deck has very high missingness, so drop it.
if "deck" in df.columns:
    print(f"deck: {missing['deck']:.2f}% missing -> column dropped")

    df = df.drop(columns=["deck"])

# Under 5% missing -> drop affected rows.
for column in ["embarked", "embark_town"]:
    if column in df.columns and df[column].isna().any():
        rate = df[column].isna().mean() * 100

        if rate < 5:
            print(
                f"{column}: {rate:.2f}% missing -> "
                "rows containing missing values dropped"
            )
            df = df.dropna(subset=[column])

# 5%-30% -> median imputation for numeric age.
if "age" in df.columns and df["age"].isna().any():
    rate = df["age"].isna().mean() * 100

    if 5 <= rate <= 30:
        print(
            f"age: {rate:.2f}% missing -> "
            "median imputation"
        )
        df["age"] = df["age"].fillna(df["age"].median())

# Save the cleaned version back to the required fallback file.
df.to_csv(CSV_PATH, index=False)

print("\nCleaned shape:", df.shape)


# ============================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)


def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return ((series < lower) | (series > upper)).sum(), lower, upper


for column in ["age", "fare"]:
    count, lower, upper = iqr_outliers(df[column])

    print(
        f"{column}: {count} outliers "
        f"(lower={lower:.2f}, upper={upper:.2f})"
    )


fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]

print("\nFare statistics:")
print(f"Mean   : {fare_mean:.2f}")
print(f"Median : {fare_median:.2f}")
print(f"Mode   : {fare_mode:.2f}")

if fare_mean > fare_median > fare_mode:
    skew_statement = "Fare is right-skewed."
elif fare_mean < fare_median < fare_mode:
    skew_statement = "Fare is left-skewed."
else:
    skew_statement = "Fare is approximately symmetric."

print(skew_statement)


# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "01_age_histogram.png"))
plt.close()

# Age box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Age Box Plot")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "02_age_boxplot.png"))
plt.close()

# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], kde=True)
plt.title("Fare Distribution")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "03_fare_histogram.png"))
plt.close()

# Fare box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"])
plt.title("Fare Box Plot")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "04_fare_boxplot.png"))
plt.close()


# ============================================================
# 4. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SURVIVAL RATES")
print("=" * 70)

sex_survival = df.groupby("sex")["survived"].mean()
pclass_survival = df.groupby("pclass")["survived"].mean()
sex_class_survival = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
)

print("\nBy sex:")
print(sex_survival)

print("\nBy passenger class:")
print(pclass_survival)

print("\nBy sex and passenger class:")
print(sex_class_survival)


# Boolean masking examples
female_survival = df[df["sex"] == "female"]["survived"].mean()
male_survival = df[df["sex"] == "male"]["survived"].mean()

print(f"\nFemale survival rate: {female_survival:.3f}")
print(f"Male survival rate: {male_survival:.3f}")


# ============================================================
# 5. REQUIRED 6-COLUMN CORRELATION MATRIX
# ============================================================

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

corr = df[corr_columns].corr()

print("\n" + "=" * 70)
print("6 x 6 CORRELATION MATRIX")
print("=" * 70)
print(corr)


plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Titanic Numeric Feature Correlation")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "05_correlation_heatmap.png"))
plt.close()


# Find two strongest correlations
pairs = []

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        pairs.append(
            (
                corr.columns[i],
                corr.columns[j],
                corr.iloc[i, j],
                abs(corr.iloc[i, j]),
            )
        )

pairs = sorted(pairs, key=lambda x: x[3], reverse=True)

print("\nTwo strongest absolute correlations:")

for pair in pairs[:2]:
    print(
        f"{pair[0]} vs {pair[1]}: "
        f"{pair[2]:.4f}"
    )


# ============================================================
# 6. MULTIVARIATE DATA STORY — 4+ CHARTS
# ============================================================

# Chart 6
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="sex",
    y="survived"
)
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "06_survival_by_sex.png"))
plt.close()

# Chart 7
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="pclass",
    y="survived"
)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "07_survival_by_class.png"))
plt.close()

# Chart 8
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="pclass",
    y="age"
)
plt.title("Age Distribution by Passenger Class")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "08_age_by_class.png"))
plt.close()

# Chart 9
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)
plt.title("Age vs Fare by Survival")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "09_age_fare_survival.png"))
plt.close()


# ============================================================
# 7. EXPLORATORY STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZATION CHECK")
print("=" * 70)

for column in ["age", "fare"]:
    mean = df[column].mean()
    std = df[column].std()

    standardized = (df[column] - mean) / std

    print(f"\n{column}")
    print(f"Original mean: {mean:.4f}")
    print(f"Original std : {std:.4f}")
    print(f"Standardized mean: {standardized.mean():.4f}")
    print(f"Standardized std : {standardized.std():.4f}")


# ============================================================
# 8. WRITE INTERPRETATIONS
# ============================================================

readme_text = f"""
# Analytics Pipeline

## Dataset
The Titanic dataset was loaded once using `sns.load_dataset("titanic")`.
Immediately after loading, the raw DataFrame was saved to `titanic.csv`
as the offline fallback.

## Missing Values

Missing percentages were measured before cleaning.

{missing.to_string()}

Cleaning decisions:
- `age` was median-imputed because its missing percentage falls between 5% and 30%.
- `embarked` and `embark_town` were handled by dropping rows because their missing percentages are below 5%.
- `deck` was dropped because its missing percentage is too high for reliable imputation.

## Outliers and Fare Distribution

Age and fare outliers were identified using the IQR rule.

Fare:
- Mean = {fare_mean:.2f}
- Median = {fare_median:.2f}
- Mode = {fare_mode:.2f}

{skew_statement}

## Data Story

1. **Survival by sex:** The survival-rate chart shows a substantial difference
   between female and male passengers. This indicates that sex was strongly
   associated with survival in this dataset.

2. **Survival by passenger class:** Survival rates differ across passenger
   classes. Passenger class therefore provides useful information about
   survival outcomes.

3. **Age by passenger class:** The box plot shows differences in the age
   distributions across passenger classes, helping explain how passenger
   characteristics varied between classes.

4. **Age, fare and survival:** The scatter plot combines age, fare and survival.
   It shows that survival was related to passenger characteristics and fare,
   although the relationship is not perfectly separated.

5. **Correlation heatmap:** The required six numeric columns were examined.
   The two strongest absolute correlations were:
   - {pairs[0][0]} vs {pairs[0][1]} = {pairs[0][2]:.4f}
   - {pairs[1][0]} vs {pairs[1][1]} = {pairs[1][2]:.4f}

## Standardization

Age and fare were standardized using:

`z = (x - mean) / standard deviation`

The resulting means are approximately zero and standard deviations are
approximately one. This was an exploratory check only and was not used as
the modeling pipeline's preprocessing.

## Charts

The `charts` directory contains the generated EDA charts.
"""

with open(
    os.path.join(BASE, "README.md"),
    "w",
    encoding="utf-8"
) as f:
    f.write(readme_text)


print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print(f"Clean dataset saved to: {CSV_PATH}")
print(f"Charts saved to: {CHART_DIR}")