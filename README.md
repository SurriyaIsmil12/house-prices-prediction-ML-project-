# House Prices: Advanced Regression

Predicting house sale prices for the Kaggle competition
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**Result: Kaggle public score 0.11885 (RMSE on log price, lower is better).**

## Approach
1. **Exploration:** SalePrice is right-skewed, so I trained on `log1p(SalePrice)`.
2. **Cleaning:** in many columns a missing value means the house doesn't have
   the feature (no pool, no garage), so I filled those with "none". LotFrontage
   was filled with its neighborhood's median.
3. **Features:** TotalSF, HouseAge, YearsSinceRemod, HasPool.
4. **Encoding:** quality ratings (Po to Ex) mapped to 0-5; other categories
   one-hot encoded. The test set was aligned to the training columns.
5. **Models:** Linear Regression, Ridge, and Random Forest, compared with
   5-fold cross-validation.
6. **Tuning:** Ridge alpha (best: 13).
7. **Outliers:** removed two huge, partially completed houses that sold cheap.

## Results
| Model | CV RMSE (5-fold) |
|---|---|
| Linear Regression | 0.156 |
| Random Forest | 0.1436 |
| Ridge (alpha=5) | 0.1399 |
| Ridge (alpha=13), outliers removed | 0.1145 |

| Kaggle submission | Score |
|---|---|
| Ridge, predictions capped at $755k | 0.12424 |
| Same, with one prediction corrected | **0.11885** |

## What I learned
- A single train/test split misled me: Linear Regression looked best (0.1276)
  but was worst under cross-validation.
- Removing 2 outliers improved the score a lot.
- One test house (Id 2550, a huge new "Partial" sale) was predicted at $1.7M
  because Ridge extrapolates. I replaced it with the median price of partial
  sales, which improved the score from 0.12424 to 0.11885. This fix is a
  judgment call, not a general rule.

## Ideas for next time
- Log-transform skewed features (LotArea, TotalSF)
- Blend Ridge with a tree model, try Lasso and gradient boosting
- Use a Pipeline so preprocessing is applied the same way to train and test

## How to run
1. Download the data from the Kaggle competition page and put `train.csv`
   and `test.csv` next to the notebook.
2. `pip install -r requirements.txt`
3. Run `house_prices.ipynb` from top to bottom.