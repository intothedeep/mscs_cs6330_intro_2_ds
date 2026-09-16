"""
Question 7 - Debug an analysis

Goal: the restaurant with the HIGHEST mean calories.

Given (buggy):
    df.groupby('restaurant')['calories'].mean().sort_values().head(1)

1. mean calories per restaurant, sorted descending - the reference table
2. result of the error code
3. result of the corrected code
4. explain the error and verify
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "00_doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)


# 1. mean calories per restaurant, highest -> lowest
#    the answer must be the FIRST row of this table; the bug returns the last one.
means = (
    df.groupby("restaurant")["calories"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
    .rename("mean_calories")
    .reset_index()
)
print("\nmean calories by restaurant (desc):\n", means.to_string(index=False))
'''
     restaurant  mean_calories
       Hardee's         691.67   <- highest, the intended answer
    Whataburger         603.33
          Sonic         588.18
     Carl's Jr.         585.31
        Wendy's         549.17
Jack in the Box         541.54
    Burger King         540.00
    Dairy Queen         516.36
In-N-Out Burger         505.00
     McDonald's         441.82
    Chick-fil-A         330.00
   White Castle         319.00   <- lowest, what the buggy code returns
'''


# 2. error code - runs without raising, returns the wrong end of the table
buggy = df.groupby("restaurant")["calories"].mean().sort_values().head(1)
print("\nerror code result:\n", buggy.round(2).to_string())
'''
restaurant
White Castle    319.0
'''


# 3. corrected code
fixed = df.groupby("restaurant")["calories"].mean().sort_values(ascending=False).head(1)
print("\ncorrected code result:\n", fixed.round(2).to_string())
'''
restaurant
Hardee's    691.67
'''

# same answer without sorting at all - independent check of the fix
s = df.groupby("restaurant")["calories"].mean()
print("\nidxmax  :", s.idxmax(), round(s.max(), 2))
print("nlargest:", s.nlargest(1).round(2).to_dict())
print("idxmin  :", s.idxmin(), round(s.min(), 2), "(reproduces the error output)")


# 4. tie-safe version - same idea as Q6's transform("min") filter
#    head(1)/idxmax() return ONE row even when several restaurants tie for the top,
#    so a tie would be silently hidden. Comparing against the maximum keeps every
#    restaurant that attains it, exactly as Q6 kept every restaurant at the minimum.
g = means  # columns: restaurant, mean_calories  (built in step 1)
top_all = g[g["mean_calories"] == g["mean_calories"].max()]
print("\ntie-safe highest:\n", top_all.to_string(index=False), "\nrows:", len(top_all))
'''
restaurant  mean_calories
  Hardee's         691.67
rows: 1
'''

# no tie occurs here, so the result is a single row and agrees with steps 3-4.
# with a tie it would return every tied restaurant instead of an arbitrary first one.


# 5. explanation
desc = """
Error: sort_values() defaults to ascending=True, so .head(1) takes the SMALLEST mean,
not the largest. The code is syntactically valid and raises nothing - it silently
answers the opposite question. It returned White Castle (319.00), the LAST row of the
table in step 1.

Fix: sort descending, sort_values(ascending=False).head(1), or skip the sort entirely
with idxmax(), which states the intent directly and has no default to get wrong.

Verification: the corrected code returns Hardee's at 691.67, the FIRST row of the step-1
table. idxmax() and nlargest(1), two routes that do not sort at all, return the same
restaurant and the same value, and idxmin() reproduces the buggy output exactly - so the
sort direction was the only defect; the grouping and the mean were already correct.

Ties: head(1) and idxmax() both return a single row, so if two restaurants shared the
highest mean one of them would vanish without warning. The step-4 filter compares every
mean against the maximum and returns all rows that match, the same tie rule used in
Question 6. No tie occurs in this dataset, so it returns exactly one row, Hardee's.
"""

print("\n", desc)
