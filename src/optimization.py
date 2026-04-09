"""
optimization.py
===============
Gurobi 12 Integer Linear Programming model for Olympic sport portfolio optimization.

Generic parameterized framework: any NSA can input their profile and receive
an optimal sport allocation recommendation.

Model:
  Maximize:  Σ_s (r_s × x_s)
  Subject to:
    Σ_s x_s ≤ B              (delegation budget)
    x_s ≥ m_s × y_s  ∀s     (minimum athletes if entering sport)
    x_s ≤ M_s × y_s  ∀s     (IOC quota ceiling)
    x_s ∈ ℤ≥0,  y_s ∈ {0,1}

"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("WARNING: Gurobi not available. Using scipy.optimize.milp fallback.")

from src.models import FEATURES, prepare_model_data
import statsmodels.api as sm

ROOT   = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "input"
OUTPUT = ROOT / "data" / "output"

def compute_sport_params(season: str = "Summer",
                         min_sport_obs: int = 5) -> pd.DataFrame:
    """Compute per-sport parameters: medal rate (r_base), min/max athletes."""
    athletes = pd.read_csv(INPUT / "athlete_events.csv")
    athletes = athletes[athletes["Year"] >= 1960]
    athletes = athletes[athletes["Season"] == season]
    athletes["has_medal"] = athletes["Medal"].notna() & (athletes["Medal"] != "NA")

    medal_events = (athletes[athletes["has_medal"]]
                    .drop_duplicates(subset=["NOC", "Year", "Event", "Medal"]))

    athlete_sport = (athletes.groupby(["NOC", "Year", "Sport"])["ID"].nunique()
                     .reset_index().rename(columns={"ID": "athletes_in_sport"}))

    medal_sport = (medal_events.groupby(["NOC", "Year", "Sport"]).size()
                   .reset_index().rename(columns={0: "medals_in_sport"}))

    sport_df = athlete_sport.merge(medal_sport, on=["NOC", "Year", "Sport"], how="left")
    sport_df["medals_in_sport"] = sport_df["medals_in_sport"].fillna(0)
    sport_df["medal_rate_sport"] = sport_df["medals_in_sport"] / sport_df["athletes_in_sport"]

    sport_params = (sport_df.groupby("Sport")
                    .agg(r_base=("medal_rate_sport", "mean"),
                         m_s=("athletes_in_sport", lambda x: int(x.quantile(0.10))),
                         M_s=("athletes_in_sport", lambda x: int(x.quantile(0.90))),
                         n_obs=("NOC", "count"))
                    .reset_index())

    sport_params = sport_params[sport_params["n_obs"] >= min_sport_obs].copy()
    sport_params["m_s"] = sport_params["m_s"].clip(lower=1).astype(int)
    sport_params["M_s"] = sport_params["M_s"].clip(lower=sport_params["m_s"]).astype(int)
    sport_params = sport_params.sort_values("r_base", ascending=False).reset_index(drop=True)

    return sport_params


def adjust_r_for_profile(sport_params: pd.DataFrame,
                         zinb_model,
                         nsa_profile: dict,
                         sport_params_baseline: Optional[dict] = None) -> pd.DataFrame:
    """Adjust sport medal rates for NSA profile using ZINB count-component coefficients."""
    all_params = zinb_model.params
    params = all_params[~all_params.index.str.startswith("inflate_")]

    log_adj = 0.0
    feature_map = {
        "log_gdp_per_capita": np.log(max(nsa_profile.get("gdp_per_capita", 10000), 1)),
        "log_population": np.log(max(nsa_profile.get("population", 1e7), 1)),
        "female_ratio": nsa_profile.get("female_ratio", 0.3),
        "is_host": float(nsa_profile.get("is_host", 0)),
    }

    baseline_vals = {
        "log_gdp_per_capita": 9.0,   # ~$8,100 per capita
        "log_population": 16.0,      # ~9 million
        "female_ratio": 0.27,
        "is_host": 0.0,
    }

    for feat, val in feature_map.items():
        if feat in params.index:
            delta = val - baseline_vals.get(feat, 0)
            log_adj += params[feat] * delta

    adj_factor = np.exp(log_adj)

    sp = sport_params.copy()
    sp["r_adjusted"] = (sp["r_base"] * adj_factor).clip(lower=0)
    sp["adj_factor"] = adj_factor

    return sp


def solve_sport_allocation(sport_params: pd.DataFrame,
                           delegation_budget: int,
                           r_col: str = "r_adjusted",
                           top_n_sports: int = 40,
                           verbose: bool = False) -> dict:
    """Solve sport allocation ILP with Gurobi: maximize expected medals subject to budget."""
    sp = sport_params.nlargest(top_n_sports, r_col).reset_index(drop=True)
    S = len(sp)
    r  = sp[r_col].values
    ms = sp["m_s"].values
    Ms = sp["M_s"].values

    if not GUROBI_AVAILABLE:
        return _solve_fallback(sp, delegation_budget, r, ms, Ms)

    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if verbose else 0)
        env.start()
        model = gp.Model(env=env)

        x = model.addVars(S, vtype=GRB.INTEGER, name="x")
        y = model.addVars(S, vtype=GRB.BINARY, name="y")

        model.setObjective(gp.quicksum(r[s] * x[s] for s in range(S)), GRB.MAXIMIZE)

        budget_constr = model.addConstr(
            gp.quicksum(x[s] for s in range(S)) <= delegation_budget, name="budget")

        for s in range(S):
            model.addConstr(x[s] >= ms[s] * y[s], name=f"min_{s}")
            model.addConstr(x[s] <= Ms[s] * y[s], name=f"max_{s}")

        model.optimize()

        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            x_vals = np.array([round(x[s].X) for s in range(S)])
            y_vals = np.array([round(y[s].X) for s in range(S)])

            selected = sp.copy()
            selected["athletes_allocated"] = x_vals
            selected["enter_sport"] = y_vals.astype(int)
            selected["expected_medals"] = r * x_vals
            selected = selected[selected["enter_sport"] == 1].sort_values(
                "expected_medals", ascending=False
            )

            total_expected = sum(r[s] * x_vals[s] for s in range(S))
            total_athletes = sum(x_vals)

            # Shadow price: solve at B+1 to get marginal return per additional athlete
            shadow_price = None
            try:
                model2 = gp.Model(env=env)
                model2.setParam("OutputFlag", 0)
                x2 = model2.addVars(S, vtype=GRB.INTEGER, name="x2")
                y2 = model2.addVars(S, vtype=GRB.BINARY,  name="y2")
                model2.setObjective(gp.quicksum(r[s]*x2[s] for s in range(S)), GRB.MAXIMIZE)
                model2.addConstr(gp.quicksum(x2[s] for s in range(S)) <= delegation_budget + 1)
                for s in range(S):
                    model2.addConstr(x2[s] >= ms[s] * y2[s])
                    model2.addConstr(x2[s] <= Ms[s] * y2[s])
                model2.optimize()
                if model2.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    shadow_price = round(model2.ObjVal - total_expected, 4)
            except Exception:
                pass

            return {
                "status": "optimal",
                "objective": round(total_expected, 3),
                "total_athletes": int(total_athletes),
                "delegation_budget": delegation_budget,
                "shadow_price": round(shadow_price, 4) if shadow_price is not None else None,
                "allocation": selected,
                "gurobi_status": model.Status,
            }
        else:
            return {"status": "infeasible", "gurobi_status": model.Status}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _solve_fallback(sp, budget, r, ms, Ms):
    """Greedy fallback: sort by efficiency and fill budget."""
    efficiency = r / np.maximum(ms, 1)
    order = np.argsort(-efficiency)

    x_vals = np.zeros(len(sp), dtype=int)
    remaining = budget
    for s in order:
        if remaining >= ms[s]:
            alloc = min(Ms[s], remaining)
            x_vals[s] = alloc
            remaining -= alloc

    selected = sp.copy()
    selected["athletes_allocated"] = x_vals
    selected["enter_sport"] = (x_vals > 0).astype(int)
    selected["expected_medals"] = r * x_vals
    selected = selected[selected["enter_sport"] == 1].sort_values("expected_medals", ascending=False)

    return {
        "status": "greedy_fallback",
        "objective": round(float((r * x_vals).sum()), 3),
        "total_athletes": int(x_vals.sum()),
        "delegation_budget": budget,
        "shadow_price": None,
        "allocation": selected,
    }


def budget_sensitivity(sport_params: pd.DataFrame,
                       budgets: list[int],
                       r_col: str = "r_adjusted",
                       top_n_sports: int = 40) -> pd.DataFrame:
    """Solve ILP at multiple budget levels to generate medal-budget curve."""
    rows = []
    for B in budgets:
        result = solve_sport_allocation(sport_params, B, r_col=r_col,
                                        top_n_sports=top_n_sports, verbose=False)
        rows.append({
            "delegation_budget": B,
            "expected_medals": result.get("objective", np.nan),
            "shadow_price": result.get("shadow_price", np.nan),
            "status": result.get("status", "error"),
        })
    return pd.DataFrame(rows)


def r_sensitivity(sport_params: pd.DataFrame,
                  delegation_budget: int,
                  deltas: list[float] = [-0.2, -0.1, 0, 0.1, 0.2],
                  r_col: str = "r_adjusted",
                  top_n_sports: int = 40) -> pd.DataFrame:
    """Vary all medal rates ±delta to test solution robustness."""
    rows = []
    for delta in deltas:
        sp_mod = sport_params.copy()
        sp_mod[r_col] = sp_mod[r_col] * (1 + delta)
        result = solve_sport_allocation(sp_mod, delegation_budget,
                                        r_col=r_col, top_n_sports=top_n_sports)
        rows.append({
            "r_delta": f"{delta:+.0%}",
            "expected_medals": result.get("objective", np.nan),
            "status": result.get("status"),
        })
    return pd.DataFrame(rows)


def load_zinb(season: str = "Summer"):
    """Load fitted ZINB model from data/output/."""
    path = OUTPUT / f"zinb_{season.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Run notebooks/02_Predictive_Model.ipynb first to generate {path}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)
