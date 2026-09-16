"""
Question 4 - Summarize by food type

for each food type,
1. the number of menu items
2. calories: mean, std, min, max
3. write 2 observations supported by values in my table
   * use the sample std used by your software's standard summary function
     -> pandas .std() is already the sample std (ddof=1)
   * sort the table from highest to lowest mean calories
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "00_doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)


# 1. the number of menu items -> n="count"
# 2. calories: mean, std, min, max   (std = sample std, pandas default ddof=1)
#    sorted highest -> lowest mean calories; displayed values rounded to 2 decimals
summary = (
    df.groupby("type")["calories"]
    .agg(n="count", mean="mean", std="std", min="min", max="max")
    .sort_values("mean", ascending=False)
    .round(2)
    .reset_index()
)

print("\ncalories by food type:\n", summary.to_string(index=False))
'''
                    type  n   mean    std  min  max
                  Burger 69 620.20 277.16  140 1240
               Milkshake 12 606.67  98.84  430  800
Breaded Chicken Sandwich 11 522.09 117.04  380  750
Grilled Chicken Sandwich 11 408.00 124.02  180  590
            French Fries 12 314.08  49.69  230  395
         Chicken Nuggets 11 274.55 119.78  130  530
'''


# overall row: computed on all 126 items at once, NOT an average of the six group means
# (the six groups are unequal in size, so averaging the group means would under-weight
# Burger, which alone holds 69 of the 126 items)
overall = pd.DataFrame(
    [{
        "type": "Overall",
        "n": len(df),
        "mean": round(df["calories"].mean(), 2),
        "std": round(df["calories"].std(), 2),
        "min": df["calories"].min(),
        "max": df["calories"].max(),
    }]
)
print("\noverall:\n", overall.to_string(index=False))
'''
   type   n   mean    std  min  max
Overall 126 532.49 250.84  130 1240
'''


# 3. two observations supported by values in the table
desc = """
(1) Burgers and milkshakes have almost the same mean calories (620.20 vs 606.67),
    but they are not equally predictable: the burger standard deviation is 277.16
    against 98.84 for milkshakes, and the burger range spans 140-1240 while every
    milkshake falls between 430 and 800. So "burger" describes a very wide class
    of items, whereas a milkshake is close to a fixed portion.

(2) Breading, not the chicken, drives the gap in chicken sandwiches: breaded ones
    average 522.09 calories against 408.00 for grilled, a difference of 114.09 on
    the same item count (n = 11 each). The lowest type, Chicken Nuggets (274.55),
    still reaches a 530-calorie maximum, so the type mean alone does not bound
    what a single order can cost.
"""

print("\n", desc)

