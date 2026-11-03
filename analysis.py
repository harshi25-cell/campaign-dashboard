"""Data preparation, RFM/CLTV segmentation, fatigue levels and response model.

All the heavy lifting lives here so app.py only has to display results.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CAMPAIGNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
             "AcceptedCmp4", "AcceptedCmp5", "Response"]
CAMPAIGN_LABELS = ["Campaign 1", "Campaign 2", "Campaign 3",
                   "Campaign 4", "Campaign 5", "Campaign 6"]
PRIOR = CAMPAIGNS[:5]
SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumDealsPurchases", "NumWebPurchases",
                 "NumCatalogPurchases", "NumStorePurchases"]

FATIGUE_ORDER = [
    "A. Engaged (2+ accepts)",
    "B. Warm (1 accept, latest = Cmp 5)",
    "C. Cooling (1 accept, earlier only)",
    "D. Ignoring buyers (active, never respond)",
    "E. Dormant (inactive, never respond)",
]
CLTV_ORDER = ["High", "Medium", "Low"]


def load_data(path=None):
    """Find marketing_campaign.csv anywhere in the project and load it."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    matches = sorted(here.rglob("marketing_campaign.csv"))
    if not matches:
        raise FileNotFoundError(
            "marketing_campaign.csv was not found in the repository. "
            "Upload it to GitHub next to app.py.")
    return pd.read_csv(matches[0], encoding="utf-8-sig")


def _fatigue_level(row, median_spend):
    n = sum(row[c] for c in PRIOR)
    if n >= 2:
        return FATIGUE_ORDER[0]
    if n == 1 and row["AcceptedCmp5"] == 1:
        return FATIGUE_ORDER[1]
    if n == 1:
        return FATIGUE_ORDER[2]
    if row["Recency"] <= 50 and row["Spend"] >= median_spend:
        return FATIGUE_ORDER[3]
    return FATIGUE_ORDER[4]


def _quintile(series, higher_is_better=True):
    ranks = series.rank(method="first", ascending=higher_is_better)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def prepare(raw):
    df = raw.copy()

    # --- cleaning -------------------------------------------------------
    df.loc[df["Income"] > 200000, "Income"] = np.nan      # one 666,666 outlier
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Marital"] = df["Marital_Status"].replace(
        {"Alone": "Single", "Absurd": "Single", "YOLO": "Single",
         "Together": "Partner", "Married": "Partner"})
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["TenureM"] = ((df["Dt"].max() - df["Dt"]).dt.days / 30.4).clip(lower=1)
    df["Kids"] = df["Kidhome"] + df["Teenhome"]
    df["Spend"] = df[SPEND_COLS].sum(axis=1)
    df["F"] = df[PURCHASE_COLS].sum(axis=1)
    df["nAcc"] = df[CAMPAIGNS].sum(axis=1)
    df["nPrior"] = df[PRIOR].sum(axis=1)

    # --- RFM scores (5 = best) and CLTV segments ------------------------
    df["R"] = _quintile(df["Recency"], higher_is_better=False)
    df["Fs"] = _quintile(df["F"])
    df["Ms"] = _quintile(df["Spend"])
    df["RFM"] = df["R"] + df["Fs"] + df["Ms"]
    retention = 1 - df["Recency"] / 100 * 0.8          # 1.0 (just bought) .. 0.2
    df["CLTV"] = (df["Spend"] / df["TenureM"]) * 24 * retention
    df["CLTV_seg"] = pd.qcut(df["RFM"].rank(method="first"), 3,
                             labels=["Low", "Medium", "High"]).astype(str)

    # --- fatigue levels (campaigns 1-5 only) ----------------------------
    med = df["Spend"].median()
    df["Fatigue"] = df.apply(_fatigue_level, axis=1, median_spend=med)

    # age / income bands for profiling
    df["Income band"] = pd.qcut(df["Income"], 4,
                                labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype(str)
    df["Recency band"] = pd.cut(df["Recency"], [-1, 19, 39, 59, 79, 99],
                                labels=["0-19", "20-39", "40-59", "60-79", "80-99"]).astype(str)
    df["Household"] = np.where(df["Kids"] == 0, "No children", "With children")
    return df


MODEL_FEATURES = ["Income", "Recency", "Spend", "F", "Kids", "Age", "TenureM",
                  "NumWebVisitsMonth", "NumDealsPurchases", "NumCatalogPurchases",
                  "NumWebPurchases", "NumStorePurchases", "MntWines",
                  "MntMeatProducts", "MntGoldProds", "nPrior"] + PRIOR


def add_model_scores(df):
    """Cross-validated probability of responding to Campaign 6."""
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=1)
    p = cross_val_predict(model, df[MODEL_FEATURES], df["Response"],
                          cv=cv, method="predict_proba")[:, 1]
    out = df.copy()
    out["p_response"] = p
    auc = roc_auc_score(df["Response"], p)
    model.fit(df[MODEL_FEATURES], df["Response"])
    importance = (pd.Series(model.feature_importances_, index=MODEL_FEATURES)
                  .sort_values(ascending=False))
    return out, auc, importance


def tier_from_p(p, breakeven):
    return np.select(
        [p >= breakeven, p >= 0.15, p >= 0.08],
        ["Tier 1 - contact (multi-channel)",
         "Tier 2 - test (low-cost channels)",
         "Tier 3 - light touch (email/app only)"],
        "Tier 4 - suppress from paid channels")
