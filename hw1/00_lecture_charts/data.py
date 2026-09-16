"""Load the fast-food nutrition dataset. Pure read + derived views, no plotting."""

from pathlib import Path

import pandas as pd

__all__ = ["TYPE_ORDER", "load", "series_by_type"]

DATA_PATH = (
    Path(__file__).parent.parent / "00_doc"
    / "01_Fast foods - Nutritional information.xlsx"
)

# Fixed display order: ascending median calories, so every by-type chart in the
# gallery reads left-to-right the same way and panels stay comparable.
TYPE_ORDER: list[str] = [
    "Chicken Nuggets",
    "French Fries",
    "Grilled Chicken Sandwich",
    "Breaded Chicken Sandwich",
    "Milkshake",
    "Burger",
]

SHORT_LABEL: dict[str, str] = {
    "Chicken Nuggets": "Nuggets",
    "French Fries": "Fries",
    "Grilled Chicken Sandwich": "Grilled\nChicken",
    "Breaded Chicken Sandwich": "Breaded\nChicken",
    "Milkshake": "Milkshake",
    "Burger": "Burger",
}


def load() -> pd.DataFrame:
    """Read the workbook; 126 menu items x 8 variables, no missing values."""
    return pd.read_excel(DATA_PATH)


def series_by_type(df: pd.DataFrame, column: str) -> tuple[list[pd.Series], list[str]]:
    """Return (values per food type, short axis labels) in TYPE_ORDER."""
    groups = [df.loc[df["type"] == t, column] for t in TYPE_ORDER]
    labels = [SHORT_LABEL[t] for t in TYPE_ORDER]
    return groups, labels
