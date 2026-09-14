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
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "doc" / "01_Fast foods - Nutritional information.xlsx"

df = pd.read_excel(DATA_PATH)
n = df.shape[0]
print(n)