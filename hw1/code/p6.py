"""
Question 6 - Use two grouping variables

1. mean calories for every restaurant x food type combination
2. restaurant with lowest mean calories for each food type
3. a six-row table with food type, restaurant, mean calories,
   the number of items used to calculate that mean
4. state how your procedure would handle ties, even if no tie occurs
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "00_doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)


# 1. mean calories for every restaurant x food type combination
#    groupby keeps only the combinations that APPEAR in the data (68 of the 6 x 12 = 72
#    grid). unstack()/pivot would materialise the 4 absent pairs as NaN rows.
combos = (
    df.groupby(["type", "restaurant"])["calories"]
    .agg(mean_cal="mean", n="count")
    .reset_index()
)
print("\ncombinations:", len(combos), "of", df["type"].nunique() * df["restaurant"].nunique())

# full 6 x 12 = 72 grid: the 4 absent pairs are filled with 0 so the listing shows every
# restaurant under every food type. within each food type, cheapest mean first.
full_grid = pd.MultiIndex.from_product(
    [sorted(df["type"].unique()), sorted(df["restaurant"].unique())],
    names=["type", "restaurant"],
)
all_combos = (
    combos.set_index(["type", "restaurant"])
    .reindex(full_grid, fill_value=0)
    .round(2)
    .reset_index()
    .sort_values(["type", "mean_cal"])
    .reset_index(drop=True)
)
print("\nall combinations (72 rows, absent pairs as 0):\n", all_combos.to_string(index=False))
print("\nn sums to", all_combos["n"].sum(), "and", (all_combos["n"] == 0).sum(), "rows are absent pairs")


# 2. restaurant with lowest mean calories for each food type
#    computed on `combos` (the 68 real pairs), NOT on the zero-filled grid: a filled 0
#    would win every minimum it touches and would mean "sells a 0-calorie item".
#    transform("min") broadcasts each type's minimum back onto its rows, so the
#    comparison keeps EVERY restaurant holding that minimum - see step 4.
type_min = combos.groupby("type")["mean_cal"].transform("min")
lowest = combos[combos["mean_cal"] == type_min]

print("\nlowest mean calories:\n",
      lowest.sort_values("type")[["type", "restaurant", "mean_cal"]]
      .round(2).to_string(index=False))
'''
                    type   restaurant  mean_cal
Breaded Chicken Sandwich White Castle     380.0
                  Burger White Castle     244.0
         Chicken Nuggets  Chick-fil-A     130.0
            French Fries   McDonald's     230.0
Grilled Chicken Sandwich White Castle     180.0
               Milkshake  Whataburger     430.0
'''


# 3. six-row table: type, restaurant, mean calories, n items behind the mean
table = (
    lowest.sort_values("type")
    .round(2)
    .rename(columns={"type": "food_type", "mean_cal": "mean_calories", "n": "n_items"})
    .reset_index(drop=True)
)
print("\nlowest mean calories per food type:\n", table.to_string(index=False))
'''
               food_type   restaurant  mean_calories  n_items
Breaded Chicken Sandwich White Castle          380.0        1
                  Burger White Castle          244.0        5
         Chicken Nuggets  Chick-fil-A          130.0        1
            French Fries   McDonald's          230.0        1
Grilled Chicken Sandwich White Castle          180.0        1
               Milkshake  Whataburger          430.0        1
'''
print("\nrows:", len(table))


# 4. tie handling
#    why the question asks for this even though no tie occurs:
#    on THIS data all three of these return the same six rows, so the output cannot tell
#    them apart:
#        g.loc[g.groupby("type").mean_cal.idxmin()]
#        g.sort_values("mean_cal").groupby("type").head(1)
#        g[g.mean_cal == g.groupby("type").mean_cal.transform("min")]   <- used above
#    the first two keep only ONE row per type, so a tied restaurant is dropped with no
#    error and the table still looks correct. the bug is already there; the data happens
#    to hide it. since no tie occurs, running the code proves nothing, and the only way
#    to show the procedure is safe is to state what it does when a tie appears.
desc = """
Ties: the procedure compares each combination's mean against its food type's minimum
(the transform("min") filter above), so if two restaurants shared the lowest mean for a
food type BOTH rows would be returned and the table would have more than six rows.
idxmin() was deliberately avoided: it returns the first matching row only, which would
hide the tie and still print a six-row table. No tie occurs in this dataset, so the
result is exactly six rows.

Caveat visible in the n_items column: five of the six winners rest on a single menu
item, so only the Burger row (White Castle, n = 5) is a stable estimate.
"""

print("\n", desc)
