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


# five menu items with highest sodium content
# restaurant, item, food type, sodium and calories, sorted from highest to lowest sodium
columns = ["restaurant", "item", "type", "sodium", "calories"]
print(df.nlargest(5, "sodium")[["restaurant", "item", "type", "sodium", "calories"]])
'''
          restaurant                          item    type  sodium  calories
93          Hardee's     2/3 LB Double Thickburger  Burger    2460      1140
90          Hardee's    1/2 LB Original Thinkbuger  Burger    2240      1030
42   Jack in the Box  Bacon Ultimate Cheeseburger™  Burger    2190       910
114      Whataburger    A.1. Thick & Hearty Burger  Burger    2080       990
26           Wendy's      3/4 lb. Triple w/ Cheese  Burger    1990      1090
'''
print(df.nlargest(5, "sodium")[["restaurant", "item", "type", "sodium", "calories"]].to_string(index=False))

# identify min max calroie menu items
min_cal_items = df[df['calories'] == df['calories'].min()]
max_cal_items = df[df['calories'] == df['calories'].max()]



# report all 8 variables for each
print("\nmin: \n", min_cal_items.to_string(index=False))
print("\nmax: \n", max_cal_items.to_string(index=False))

# explain why examining complete records is more informative than reporting only the extreme values?

desc = """

"""

print("\n", desc)