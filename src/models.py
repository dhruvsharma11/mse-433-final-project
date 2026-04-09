"""
models.py
=========
Regression and ML model wrappers for predicting Olympic medal counts.

Models:
  1. OLS on log(medals+1)               [baseline]
  2. Poisson GLM                         [count baseline]
  3. Zero-Inflated Negative Binomial     [primary statistical model]
  4. Gradient Boosting Regressor         [ML comparison]

All models use the same feature set and are fit separately for Summer and Winter.
Cross-validation is chronological (train on earlier Games, test on later Games).

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

FEATURES = [
    "log_delegation_size",
    "log_gdp_per_capita",
    "log_population",
    "is_host",
    "female_ratio",
    "sport_hhi",
    "sport_count",
]
TARGET_COUNT = "total_medals"
TARGET_LOG   = "log_total_medals"


def prepare_model_data(panel: pd.DataFrame,
                       min_delegation: int = 3) -> pd.DataFrame:
    """
    Filter panel to rows suitable for regression:
    - Complete cases (no NaN in features or target)
    - Minimum delegation size (avoids tiny delegations distorting ratios)
    """
    df = panel.copy()
    df = df[df["delegation_size"] >= min_delegation]
    required = FEATURES + [TARGET_COUNT, TARGET_LOG, "Year"]
    df = df.dropna(subset=required).reset_index(drop=True)
    df["const"] = 1.0
    return df


def chronological_cv_splits(years: pd.Series, n_folds: int = 5):
    """
    Yield (train_idx, test_idx) pairs where each fold tests on ~1/n_folds
    of the most recent Games, trained on all prior Games.
    """
    unique_years = sorted(years.unique())
    n = len(unique_years)
    fold_size = max(1, n // n_folds)

    for i in range(n_folds):
        test_start_idx = n - (n_folds - i) * fold_size
        test_end_idx   = n - (n_folds - i - 1) * fold_size
        if test_start_idx <= 0:
            continue
        train_years = unique_years[:test_start_idx]
        test_years  = unique_years[test_start_idx:test_end_idx]
        train_idx = years[years.isin(train_years)].index
        test_idx  = years[years.isin(test_years)].index
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue
        yield train_idx, test_idx


def fit_ols(df: pd.DataFrame):
    """OLS baseline: log(medals+1) target."""
    X = sm.add_constant(df[FEATURES])
    y = df[TARGET_LOG]
    return sm.OLS(y, X).fit()


def fit_poisson(df: pd.DataFrame):
    """Poisson GLM baseline."""
    X = sm.add_constant(df[FEATURES])
    y = df[TARGET_COUNT]
    return sm.GLM(y, X, family=sm.families.Poisson()).fit()


def fit_negbin(df: pd.DataFrame):
    """Negative Binomial: handles overdispersion."""
    X = sm.add_constant(df[FEATURES])
    y = df[TARGET_COUNT]
    return sm.NegativeBinomial(y, X).fit(disp=False, method="nm", maxiter=500)


def fit_zinb(df: pd.DataFrame):
    """Zero-Inflated Negative Binomial: handles overdispersion and zero-inflation.
    Two-component: logistic zero model + count model for non-zero medals."""
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    X = sm.add_constant(df[FEATURES])
    y = df[TARGET_COUNT]
    exog_infl = np.ones((len(y), 1))  # intercept-only inflation equation
    model = ZeroInflatedNegativeBinomialP(y, X, exog_infl=exog_infl, p=2)
    return model.fit(method="bfgs", maxiter=1000, disp=False)


def fit_gradient_boosting(df: pd.DataFrame):
    """Gradient Boosting: non-parametric ML model for comparison.
    Hyperparameters: 300 trees, depth=4, lr=0.05."""
    X = df[FEATURES].values
    y = df[TARGET_COUNT].values
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X, y)
    return model


def predict_ols(model, df: pd.DataFrame) -> np.ndarray:
    """Back-transform OLS log predictions to count scale."""
    X = sm.add_constant(df[FEATURES], has_constant="add")
    return np.expm1(model.predict(X))


def predict_count(model, df: pd.DataFrame) -> np.ndarray:
    """Predict from Poisson or Negative Binomial GLM."""
    X = sm.add_constant(df[FEATURES], has_constant="add")
    return model.predict(X)


def predict_zinb(model, df: pd.DataFrame) -> np.ndarray:
    """Predict from ZINB: E[y] accounting for zero-inflation and count components."""
    X = sm.add_constant(df[FEATURES], has_constant="add")
    exog_infl = np.ones((len(df), 1))
    return model.predict(X, exog_infl=exog_infl)


def predict_gb(model, df: pd.DataFrame) -> np.ndarray:
    """Predict from Gradient Boosting. Clips at 0 for non-negative counts."""
    X = df[FEATURES].values
    return model.predict(X).clip(min=0)


def cross_validate(df: pd.DataFrame,
                   model_fn,
                   predict_fn,
                   n_folds: int = 5,
                   label: str = "Model") -> dict:
    """Chronological cross-validation for model evaluation."""
    rmse_list, mae_list = [], []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
            chronological_cv_splits(df["Year"], n_folds), start=1):
        train = df.loc[train_idx].reset_index(drop=True)
        test  = df.loc[test_idx].reset_index(drop=True)
        try:
            fitted = model_fn(train)
            preds  = predict_fn(fitted, test)
            actual = test[TARGET_COUNT].values

            # Skip folds where predictions are non-finite (ZINB convergence failure)
            if not np.all(np.isfinite(preds)):
                print(f"  Fold {fold} skipped: non-finite predictions (model did not converge)")
                continue

            rmse = np.sqrt(mean_squared_error(actual, preds))
            mae  = mean_absolute_error(actual, preds)
            rmse_list.append(rmse)
            mae_list.append(mae)
            fold_results.append({
                "Fold": fold,
                "Train Games": sorted(df.loc[train_idx, "Year"].unique()),
                "Test Games":  sorted(df.loc[test_idx,  "Year"].unique()),
                "RMSE": round(rmse, 3),
                "MAE":  round(mae, 3),
            })
        except Exception as e:
            print(f"  Fold {fold} failed: {e}")

    return {
        "label": label,
        "mean_rmse": np.mean(rmse_list) if rmse_list else np.nan,
        "std_rmse":  np.std(rmse_list)  if rmse_list else np.nan,
        "mean_mae":  np.mean(mae_list)  if mae_list  else np.nan,
        "folds": pd.DataFrame(fold_results),
    }


def model_summary_table(results: list) -> pd.DataFrame:
    """Build model comparison table."""
    rows = []
    for r in results:
        rows.append({
            "Model":           r["label"],
            "CV RMSE (mean)":  round(r["mean_rmse"], 3),
            "CV RMSE (std)":   round(r["std_rmse"],  3),
            "CV MAE (mean)":   round(r["mean_mae"],  3),
        })
    return pd.DataFrame(rows).set_index("Model")


def coef_table(model, model_type: str = "zinb") -> pd.DataFrame:
    """Extract coefficients, CIs, and p-values. For ZINB, shows count component only."""
    summary = model.summary2().tables[1].copy()
    summary.index.name = "Feature"
    summary = summary.rename(columns={
        "Coef.":    "Coefficient",
        "Std.Err.": "Std Error",
        "z":        "z-stat",
        "t":        "t-stat",
        "P>|z|":    "p-value",
        "P>|t|":    "p-value",
        "[0.025":   "CI Lower",
        "0.975]":   "CI Upper",
    })

    # For ZINB: drop inflate_* rows — show count component only
    if model_type == "zinb":
        summary = summary[~summary.index.str.startswith("inflate_")]

    if "p-value" not in summary.columns:
        # Fallback for column name variants
        for col in summary.columns:
            if col.startswith("P>"):
                summary = summary.rename(columns={col: "p-value"})
                break

    summary["Significant"] = summary["p-value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )
    log_features = ["log_delegation_size", "log_gdp_per_capita", "log_population"]
    summary["Interpretation"] = summary.index.map(
        lambda f: f"1% ↑ → {summary.loc[f,'Coefficient']:.2f}% ↑ medals"
        if f in log_features else ""
    )
    return summary.round(4)


def gb_feature_importance(model, feature_names=None) -> pd.DataFrame:
    """Extract and rank feature importances from Gradient Boosting model."""
    if feature_names is None:
        feature_names = FEATURES
    imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    imp["Importance"] = imp["Importance"].round(4)
    return imp


def compute_residuals(df: pd.DataFrame,
                      model,
                      predict_fn,
                      top_n: int = 10) -> pd.DataFrame:
    """Compute per-country mean residuals for over/underperformer analysis."""
    pred = predict_fn(model, df)
    df = df.copy()
    df["predicted"] = pred
    df["residual"]  = df[TARGET_COUNT] - df["predicted"]

    by_country = (
        df.groupby("NOC")[["residual", TARGET_COUNT, "predicted"]]
        .mean()
        .sort_values("residual", ascending=False)
    )
    return by_country


def sensitivity_tornado(model, predict_fn, baseline_row: pd.Series,
                        delta: float = 0.10) -> pd.DataFrame:
    """Sensitivity analysis: vary each log feature ±delta and measure impact on predictions."""
    base_df   = pd.DataFrame([baseline_row])
    base_pred = predict_fn(model, base_df)[0]

    rows = []
    for feat in FEATURES:
        if feat.startswith("log_"):
            for direction, sign in [("+10%", 1), ("-10%", -1)]:
                mod_row = baseline_row.copy()
                mod_row[feat] = baseline_row[feat] * (1 + sign * delta)
                mod_df   = pd.DataFrame([mod_row])
                mod_pred = predict_fn(model, mod_df)[0]
                rows.append({
                    "Feature":                    feat,
                    "Direction":                  direction,
                    "Change in Predicted Medals": round(mod_pred - base_pred, 3),
                    "Base Prediction":            round(base_pred, 3),
                })
    return pd.DataFrame(rows).sort_values(
        "Change in Predicted Medals", key=abs, ascending=False
    )
