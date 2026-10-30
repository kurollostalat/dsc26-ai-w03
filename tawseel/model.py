"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations
import re
from unittest import result

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
# TODO: this is every column of the table. Keep only the ones known at decision time.
FEATURE_COLUMNS: list[str] = [
     "customer_id", "merchant_id", "area", "created_at", "promised_slot",
    "cod_amount", "n_items", "address_text",
]

numeric_features = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate"]

categorical_features = [
            "customer_id",
            "merchant_id",
            "area",
            "promised_slot",
            "address_text"]




def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["original_order"] = range(len(result))
    result["original_index"] = list(result.index)

    history = (
        result.groupby(["customer_id", "created_at"], dropna=False)
        .agg(
            current_parcels=("failed", "size"),
            current_failed=("failed", "sum")
        )
        .reset_index()
        .sort_values(["customer_id", "created_at"])
    )

    history["customer_prior_parcels"] = (
        history.groupby("customer_id")["current_parcels"].cumsum()
        - history["current_parcels"]
    )

    history["prior_failed"] = (
        history.groupby("customer_id")["current_failed"].cumsum()
        - history["current_failed"]
    )

    result = result.merge(
        history[
            ["customer_id", "created_at",
             "customer_prior_parcels", "prior_failed"]
        ],
        on=["customer_id", "created_at"],
        how="left",
        sort=False
    )

    result["customer_prior_parcels"] = (
        result["customer_prior_parcels"].fillna(0).astype(int)
    )

    result["customer_prior_fail_rate"] = (
        result["prior_failed"]
        / result["customer_prior_parcels"].replace(0, np.nan)
    )

    result = result.sort_values("original_order")
    result.index = df.index.copy()
    result.index.name = df.index.name

    result = result.drop(
        columns=["original_order", "original_index", "prior_failed"]
    )

    return result




"""Return a copy of df (same index and row order) with two new columns.

    Needs customer_id, created_at and failed.
    - customer_prior_parcels (int): the same customer's parcels with created_at strictly
      earlier than this row.
    - customer_prior_fail_rate (float): share of those that failed; NaN when there are none.
    Parcels with the same timestamp are not prior to each other. The result must not depend
    on the input row order. Never modify the input.
    """
    



def time_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    n_test = int(np.ceil(len(df) * test_fraction))

    sorted_df = df.sort_values(time_col)

    boundary = sorted_df.iloc[-n_test][time_col]

    train = sorted_df[sorted_df[time_col] < boundary]
    test = sorted_df[sorted_df[time_col] >= boundary]

    return train, test


    
"""Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """
    


def build_pipeline(seed: int = 0) -> Pipeline:
    
   
    numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    
    preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)],remainder="drop")
    
    model_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(random_state=seed, max_iter=1000))])
    return model_pipeline

    """Return an UNFITTED Pipeline.

    Input: a DataFrame with at least FEATURE_COLUMNS plus the two history columns; ignore any
    other column. The final step must support predict_proba. Must not fail on unseen
    categories, missing numbers, missing categories or empty address strings. Same seed and
    same data must give identical predictions.
    """
    


def evaluate(model, test_df: pd.DataFrame) -> dict:
    selected_features = numeric_features + categorical_features
    y_prod= model.predict_proba(test_df[selected_features])
    return {
        "roc_auc": roc_auc_score(test_df[TARGET], y_prod[:, 1]),
        "pr_auc": average_precision_score(test_df[TARGET], y_prod[:, 1]),
        "brier": brier_score_loss(test_df[TARGET], y_prod[:, 1]),
        "base_rate": test_df[TARGET].mean(),
        "n": len(test_df)
    }
    """{"roc_auc", "pr_auc", "brier", "base_rate", "n"} from model.predict_proba(test_df)[:, 1]
    against test_df["failed"] (roc_auc_score, average_precision_score, brier_score_loss)."""
    


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    selected_features = numeric_features + categorical_features
    build_model=build_pipeline(seed).fit(train_df[selected_features], train_df[TARGET])
    dummy_model = DummyClassifier(strategy="prior").fit(train_df[selected_features], train_df[TARGET])
    model_eval = evaluate(build_model, test_df)
    dummy_eval = evaluate(dummy_model, test_df)
    return {
        "model": model_eval,
        "dummy": dummy_eval,
        "lift_pr_auc": model_eval["pr_auc"] - dummy_eval["pr_auc"]
    }
    """Fit build_pipeline(seed) and DummyClassifier(strategy="prior") on train_df only, evaluate
    both on test_df, return {"model": {...}, "dummy": {...}, "lift_pr_auc": model - dummy PR-AUC}."""
    
