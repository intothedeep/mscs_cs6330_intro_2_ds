"""
Question 3 - Filter with multiple conditions

1. subset: calories < 500 AND sodium < 1000
2. number of qualifying items:
3. qualifying items per restaurant, sorted highest -> lowest:
4. ties: report all tied restaurants
5. what does each row of the restaurant summary represent?
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "00_doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)


# 1. subset: fewer than 500 calories AND less than 1000 mg sodium
light = df[(df["calories"] < 500) & (df["sodium"] < 1000)]


# 2. number of qualifying items
print("\nqualifying items:", len(light))  # 49 of 126


# 3. count by restaurant, sorted highest -> lowest
# stable sort: value_counts already orders by count, so ties keep its order
counts = (
    light["restaurant"]
    .value_counts()
    .rename_axis("restaurant")
    .reset_index(name="n")
    .sort_values("n", ascending=False, kind="stable")
    .reset_index(drop=True)
)
print("\ncount by restaurant:\n", counts.to_string(index=False))
'''
     restaurant  n
Jack in the Box  7
   White Castle  7
     McDonald's  5
    Burger King  5
        Wendy's  5
     Carl's Jr.  4
    Chick-fil-A  3
          Sonic  3
    Dairy Queen  3
    Whataburger  3
       Hardee's  2
In-N-Out Burger  2
'''


# 4. ties - all restaurants sharing a count
# every restaurant is listed above, so a tie is already fully reported;
# this just names which counts are shared.
ties = counts[counts.duplicated("n", keep=False)]
print("\ntied counts:\n", ties.to_string(index=False) if len(ties) else " none")


# 5. what each row represents
desc = """
Each row is one restaurant and the number of ITS menu items that satisfy both
conditions at once (calories < 500 AND sodium < 1000 mg). It is a count of
qualifying items, not of restaurants, and not a share of that restaurant's menu:
a restaurant with a large menu can rank high simply by listing more items.
Restaurants with no qualifying item do not appear as a zero row at all.
"""

print("\n", desc)
