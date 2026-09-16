"""
1. observation:
2. variables:
3. unique restaurants:
4. food types:
5. each variable data type:
6. missing value counts:
7. appropriate
    - serving size:
    - calories:
    - sodium:
    - sugars:
8. present my result clearly
9. 2-3 sentences to describe what I learned?
"""

import json
from pathlib import Path
from pprint import pprint

import pandas as pd


def dumps(obj: object) -> str:
    """Pretty JSON: one item per line, readable names, numpy-safe."""
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


DATA_PATH = Path(__file__).parent.parent / "00_doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)
n = df.shape[0]
m = df.shape[1]

print("shape: ", df.shape)

n_observations = df.shape[0]
print("\n[][] The number of observations: ", n_observations)

n_variables = df.shape[1]
print("\n[][] The number of variables: ", n_variables)

print("\n[][] unique restaurant: ",  df["restaurant"].unique())
print("\n[][] The number of unique restaurant: ",  df["restaurant"].nunique())

print("\n[][] unique food types: ", df["type"].unique())
print("\n[][] The number of unique food types: ", df["type"].nunique())

print("\n[][] data type of each variable: \n", df.dtypes)

print("\n[][] missing value count per variable: \n", df.isna().sum())

NUMERIC_COLS = ["serving_size", "calories", "sodium", "sugars"]
# describe() already reports the median as the "50%" row - renamed here so the
# table reads against the lecture checklist. IQR is the only added row.
desc = df[NUMERIC_COLS].describe().rename(index={"50%": "median"})
desc.loc["IQR"] = desc.loc["75%"] - desc.loc["25%"]
print("\n[][] descriptive statistics: \n", desc.round(2))

# Mode is printed separately, not as a table row: these variables can have SEVERAL
# modal values (calories has a 4-way tie), and a one-cell row would silently show
# only the smallest of them.
print("\n[][] mode(s) per variable: ")
for col in NUMERIC_COLS:
    modes = df[col].mode().tolist()
    tie = " (tied)" if len(modes) > 1 else ""
    print(f"    {col}: {modes}{tie}  - each occurs {int(df[col].value_counts().max())}x")
# Skewness decides mean-vs-median: a skewed variable is summarized by the median.
print("\n[][] skewness: \n", df[NUMERIC_COLS].skew().round(2))






# print("[][] shape (rows, cols):", n, m)

# print("\n[][] header (column names):")
# print(df.columns.tolist())

# print("\n[][] first column:")
# pprint(df.iloc[:, 0].tolist(), compact=True)

# print("\n[][] first column and unique:")
# pprint(df.iloc[:, 0].unique().tolist(), compact=True)

# print("\n[][] first column and unique (json):")
# print(dumps(df.iloc[:, 0].unique().tolist()))


# """
# Quick recap of the method chain:
# - df.iloc[:, 0] → pandas Series (the restaurant column)
# - .value_counts() → pandas, counts each value → Series
# - .to_dict() → pandas, Series → native Python dict
# - dumps(...) → stdlib json, dict → pretty JSON string
# """

# print("\n[][] unique count:")
# print(df.iloc[:, 0].nunique())

# print("\n[][] count per restaurant (json):")
# print(dumps(df.iloc[:, 0].value_counts().to_dict()))

# print("\n[][] first row:")
# pprint(df.iloc[0].tolist(), compact=True)
