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


DATA_PATH = Path(__file__).parent.parent / "doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)
n = df.shape[0]
m = df.shape[1]

print("[][] shape (rows, cols):", n, m)

print("\n[][] header (column names):")
print(df.columns.tolist())

print("\n[][] first column:")
pprint(df.iloc[:, 0].tolist(), compact=True)

print("\n[][] first column and unique:")
pprint(df.iloc[:, 0].unique().tolist(), compact=True)

print("\n[][] first column and unique (json):")
print(dumps(df.iloc[:, 0].unique().tolist()))


"""
Quick recap of the method chain:
- df.iloc[:, 0] → pandas Series (the restaurant column)
- .value_counts() → pandas, counts each value → Series
- .to_dict() → pandas, Series → native Python dict
- dumps(...) → stdlib json, dict → pretty JSON string
"""

print("\n[][] unique count:")
print(df.iloc[:, 0].nunique())

print("\n[][] count per restaurant (json):")
print(dumps(df.iloc[:, 0].value_counts().to_dict()))

print("\n[][] first row:")
pprint(df.iloc[0].tolist(), compact=True)
