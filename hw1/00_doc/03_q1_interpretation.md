# HW1 Question 1 — Understand the dataset

Data: `01_Fast foods - Nutritional information.xlsx` — 126 observations, 8 variables,
12 unique restaurants, 6 food types, **no missing values** in any column.
Numbers produced by `hw1/code/p1.py`; figures by `hw1/code/p1_figures.py` → `hw1/figures/`.

## Descriptive statistics

| | serving_size (g) | calories (kcal) | sodium (mg) | sugars (g) |
|---|---|---|---|---|
| mean | 224.27 | 532.49 | 973.75 | 13.29 |
| std | 108.03 | 250.84 | 523.41 | 21.02 |
| min | 44 | 130 | 50 | 0 |
| 25% | 126.50 | 330 | 569.25 | 3 |
| median | 217.50 | 515 | 930 | 7 |
| 75% | 315.25 | 670 | 1285.25 | 11 |
| max | 467 | 1240 | 2460 | 93 |
| **skew** | **0.24** | **0.65** | **0.42** | **2.71** |

Mean and standard deviation are appropriate for serving_size, calories, and sodium
(|skew| < 1). For sugars, the mean (13.29) sits *above* the 75th percentile (11), so the
**median and IQR** are the honest summary — a mean is meaningless for a bimodal variable.

## Figure 1 — Histograms with a fitted normal curve (`fig1_histograms.png`)

Form: histogram (Lecture 5, slides 12/14) overlaid with a normal PDF fitted from the
sample mean and standard deviation (Lecture 6, slides 4–7, 10).

- **serving_size** (skew 0.24) is near-symmetric but *flat*, not bell-shaped: the bars
  track the dotted uniform reference line far better than the normal curve across
  44–467 g. Low skew alone does not make a variable normal.
- **calories** (0.65) and **sodium** (0.42) are single-peaked with a mild right tail.
  The fitted normal tracks the body of both reasonably well; **normal is a workable
  approximation** here.
- **sugars** (2.71) is nowhere near normal. The fitted curve puts substantial mass at
  30–50 g where the data has *none*, and cannot reach the cluster at 56–93 g. A single
  normal fit is the wrong model for this variable.

## Figure 2 — Box plots by food type (`fig2_boxplot_by_type.png`)

Form: box plot for comparing samples (Lecture 5, slides 20–21). Box = IQR,
whisker = 1.5 × IQR, dots = outliers.

- Calories rise monotonically by type: Chicken Nuggets (median ~240) → French Fries →
  Grilled Chicken Sandwich → Breaded Chicken Sandwich → Milkshake → **Burger**
  (median ~600, and the widest spread — 130 to 1240 kcal).
- Sodium runs the *opposite* way from sugars: Burger (median 1170 mg), Breaded Chicken
  Sandwich (1120) and Grilled Chicken Sandwich (1040) all exceed 1000 mg, while
  Milkshake (310) and French Fries (295) stay near 300 mg.
- Sugars separates completely: every non-shake type has a median of 0–9 g, while
  **Milkshake has a median of 75 g** with no overlap against any other group.

## Figure 3 — Why sugars is skewed (`fig3_sugars_bimodal.png`)

Splitting the sugars histogram by type shows the 2.71 skew is not a long tail at all —
it is **two separate populations**: 114 items at 0–20 g, and 12 milkshakes at 56–93 g,
with no observation whatsoever between 20 g and 56 g.

## What I learned (2–3 sentences, for submission)

The dataset is complete — 126 items across 12 restaurants and 6 food types with zero
missing values — but its four numeric variables do not share one distributional shape:
serving size is nearly flat across its range, calories and sodium are single-peaked with
a mild right skew, and sugars is strongly skewed (2.71) and bimodal. Grouping sugars by
food type shows that skew is not a heavy tail but a mixture of two populations —
milkshakes (median 75 g) versus every other item (median 0–9 g) — so the overall mean of
13.29 g describes no actual menu item and the median should be reported instead. More
generally, the summary statistics alone concealed structure that the plots made obvious,
including that the nutritional burden splits by category: savory items carry the sodium
(burgers and chicken sandwiches above 1000 mg) while milkshakes carry the sugar.
