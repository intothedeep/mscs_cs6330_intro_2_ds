"""
Question 5 - Summarize by restaurant

for each restaurant,
1. n menu items, mean of calories, sodium, sugars; sorted first by mean calories desc
2. identification: which restaurant is highest and lowest on each mean nutritional measure
3. why these restaurant summaries should not automatically be interpreted as comparisons
   of otherwise equivalent menus
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "00_doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)

MEASURES = ["calories", "sodium", "sugars"]

# for each restaurant
# 1. n menu items + mean calories / sodium / sugars, sorted by mean calories desc
#    (displayed values rounded to 2 decimals)
summary = (
    df.groupby("restaurant")
    .agg(
        n=("item", "count"),
        mean_calories=("calories", "mean"),
        mean_sodium=("sodium", "mean"),
        mean_sugars=("sugars", "mean"),
    )
    .sort_values("mean_calories", ascending=False)
    .round(2)
    .reset_index()
)

print("\nby restaurant:\n", summary.to_string(index=False))
'''
     restaurant  n  mean_calories  mean_sodium  mean_sugars
       Hardee's 12         691.67      1399.17        14.92
    Whataburger 12         603.33      1149.17        14.75
          Sonic 11         588.18       926.36        10.00
     Carl's Jr. 13         585.31      1085.92        12.99
        Wendy's 12         549.17      1057.50        14.17
Jack in the Box 13         541.54      1082.31        11.69
    Burger King 11         540.00       817.27        14.00
    Dairy Queen 11         516.36      1001.82        12.09
In-N-Out Burger  5         505.00       731.00        19.00
     McDonald's 11         441.82       766.36        12.45
    Chick-fil-A  5         330.00       654.00        18.50
   White Castle 10         319.00       568.00        10.60
'''


# overall row: computed on all 126 items at once, NOT an average of the 12 group means
# (the group means are unweighted, so averaging them would over-weight small menus)
overall = pd.DataFrame(
    [{
        "restaurant": "Overall",
        "n": len(df),
        "mean_calories": round(df["calories"].mean(), 2),
        "mean_sodium": round(df["sodium"].mean(), 2),
        "mean_sugars": round(df["sugars"].mean(), 2),
    }]
)
print("\noverall:\n", overall.to_string(index=False))


# 2. identification: highest and lowest restaurant on each mean measure
# idxmax/idxmin is safe here only because no two restaurants tie on any mean;
# the assert makes that assumption fail loudly instead of silently dropping a tie.
for measure in MEASURES:
    col = f"mean_{measure}"
    assert not summary[col].duplicated().any(), f"tie on {col}: report all tied rows"
    hi = summary.loc[summary[col].idxmax()]
    lo = summary.loc[summary[col].idxmin()]
    print(f"\nmean {measure}: highest = {hi.restaurant} ({hi[col]}), lowest = {lo.restaurant} ({lo[col]})")
'''
mean calories: highest = Hardee's (691.67),  lowest = White Castle (319.0)
mean sodium:   highest = Hardee's (1399.17), lowest = White Castle (568.0)
mean sugars:   highest = In-N-Out Burger (19.0), lowest = Sonic (10.0)
'''


# 3. why these summaries are not comparisons of otherwise equivalent menus
# supporting evidence: the food-type mix behind each restaurant mean, and the menu sizes
mix = pd.crosstab(df["restaurant"], df["type"])
print("\nfood-type mix per restaurant:\n", mix.to_string())

desc = """
These means describe only the items from each restaurant that are in this dataset, not the
whole menu, so Hardee's 691.67 is the average of 12 listed items. The menus are also not
made up of the same things: Chick-fil-A has no burger in this dataset, so its low mean
(330.00) mostly shows that the highest-calorie food type is missing from its list, not that
its food is lighter. The number of items also differs a lot, from 5 for In-N-Out Burger to
13 for Carl's Jr., so a mean based on only 5 items is much less reliable than one based on
13. To compare restaurants fairly, I would have to compare the same food type across
restaurants.
"""

print("\n", desc)
