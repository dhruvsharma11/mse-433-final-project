"""

MSE 433 — Evaluating and Optimizing Olympic Medal Performance

Allows any NSA to input their profile and receive an optimal
sport allocation recommendation from the Gurobi ILP model.

Run with:
  streamlit run src/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from src.optimization import (
    compute_sport_params, adjust_r_for_profile,
    solve_sport_allocation, budget_sensitivity,
    r_sensitivity, load_zinb
)

st.set_page_config(
    page_title="Olympic Sport Portfolio Optimizer",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_params():
    zinb_summer = load_zinb("Summer")
    zinb_winter = load_zinb("Winter")
    sp_summer = compute_sport_params("Summer", min_sport_obs=5)
    sp_winter = compute_sport_params("Winter", min_sport_obs=3)
    return zinb_summer, zinb_winter, sp_summer, sp_winter

with st.spinner("Loading models and sport data..."):
    try:
        zinb_summer, zinb_winter, sp_summer, sp_winter = load_models_and_params()
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run notebooks/02_Predictive_Model.ipynb first.\n{e}")
        models_loaded = False
        st.stop()

st.markdown("""
<h1 style="margin-bottom: 0.5em;">
    <i class="fas fa-medal" style="margin-right: 0.5em;"></i>Olympic Sport Portfolio Optimizer
</h1>
""", unsafe_allow_html=True)
st.markdown("""
**Gurobi 12 Integer Linear Program for Olympic Sport Portfolio Optimization**

Input your country's profile in the sidebar and click **Run Optimization** to receive a data-driven
recommendation for which sports to enter and how many athletes to allocate to each, subject to your delegation budget and IOC quotas.

The optimizer uses a Zero-Inflated Negative Binomial regression model to estimate expected medals per sport,
adjusted for your country's GDP, population, female participation ratio, and host nation status.
""")

st.divider()

with st.sidebar:
    st.markdown("""
    <h2 style="margin-bottom: 1em;">
        <i class="fas fa-globe" style="margin-right: 0.5em;"></i>NSA Profile
    </h2>
    """, unsafe_allow_html=True)
    st.caption("Configure your country's profile below")

    season = st.selectbox("Olympic Season", ["Summer", "Winter"], index=0)

    gdp_per_capita = st.slider(
        "GDP per Capita (USD)",
        min_value=500, max_value=100_000, value=15_000, step=500,
        help="Proxy for sport infrastructure and athlete development capacity"
    )

    population = st.slider(
        "Population (millions)",
        min_value=0.5, max_value=1_500.0, value=30.0, step=0.5,
        help="Proxy for available talent pool"
    ) * 1_000_000

    female_ratio = st.slider(
        "Female Athlete Ratio",
        min_value=0.0, max_value=1.0, value=0.40, step=0.05,
        help="Share of female athletes in delegation (e.g., 0.5 = 50%)"
    )

    is_host = st.toggle("Hosting Nation?", value=False,
                        help="Is your country hosting the Olympic Games?")

    delegation_budget = st.slider(
        "Delegation Budget (total athletes)",
        min_value=10, max_value=400, value=100, step=5,
        help="Total number of athletes to allocate across sports"
    )

    st.divider()
    run_button = st.button("⚡ Run Optimization", use_container_width=True, type="primary")

if run_button:
    nsa_profile = {
        "gdp_per_capita": gdp_per_capita,
        "population":     population,
        "female_ratio":   female_ratio,
        "is_host":        int(is_host),
    }
    sp_raw = sp_summer if season == "Summer" else sp_winter
    zinb   = zinb_summer if season == "Summer" else zinb_winter

    with st.spinner("Adjusting sport medal rates for your profile..."):
        sp_adj = adjust_r_for_profile(sp_raw, zinb, nsa_profile)

    with st.spinner("Solving ILP with Gurobi 12..."):
        result = solve_sport_allocation(
            sp_adj, delegation_budget, r_col="r_adjusted",
            top_n_sports=40, verbose=False
        )

    with st.spinner("Running budget sensitivity..."):
        budgets = list(range(10, min(delegation_budget * 3 + 50, 401), 10))
        sens_df = budget_sensitivity(sp_adj, budgets, r_col="r_adjusted", top_n_sports=40)

    with st.spinner("Running parameter sensitivity..."):
        rsens = r_sensitivity(
            sp_adj, delegation_budget,
            deltas=[-0.20, -0.10, 0.0, 0.10, 0.20],
            r_col="r_adjusted", top_n_sports=40
        )

    st.session_state.update({
        "result":            result,
        "sp_adj":            sp_adj,
        "nsa_profile":       nsa_profile,
        "season":            season,
        "delegation_budget": delegation_budget,
        "sens_df":           sens_df,
        "rsens":             rsens,
    })

# Display results if a run has been completed
if "result" in st.session_state:
    result            = st.session_state["result"]
    sp_adj            = st.session_state["sp_adj"]
    run_budget        = st.session_state["delegation_budget"]
    run_season        = st.session_state["season"]
    sens_df           = st.session_state["sens_df"]
    rsens             = st.session_state["rsens"]

    # --- Status banner ---
    status = result.get("status", "error")
    obj    = result.get("objective", 0)
    alloc  = result.get("allocation", pd.DataFrame())
    shadow = result.get("shadow_price")

    if status in ("optimal", "greedy_fallback"):
        status_text = "Optimal solution" if status == "optimal" else "Completed (greedy fallback)"
        st.success(f"✓ {status_text} — Expected medals: **{obj:.2f}** ({run_season}, {run_budget} athletes)")
    else:
        st.error(f"Optimization failed: {result.get('message', status)}")
        st.stop()

    # --- KPI row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Medals", f"{obj:.2f}")
    col2.metric("Athletes Allocated", f"{result.get('total_athletes', 0)} / {run_budget}")
    col3.metric("Sports Selected", len(alloc))
    if shadow is not None:
        col4.metric("Shadow Price", f"{shadow:.4f}",
                    help="Marginal medals per additional athlete (marginal cost analysis)")
    else:
        col4.metric("Season", run_season)

    st.divider()

    # --- Two-column layout: table + bar chart ---
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown("""
        <h3 style="margin-top: 0;">
            <i class="fas fa-list-check" style="margin-right: 0.5em;"></i>Recommended Sport Allocation
        </h3>
        """, unsafe_allow_html=True)
        if len(alloc) > 0:
            display_alloc = alloc[["Sport", "athletes_allocated", "expected_medals"]].copy()
            display_alloc["athletes_allocated"] = display_alloc["athletes_allocated"].astype(int)
            display_alloc["expected_medals"] = display_alloc["expected_medals"].round(3)
            display_alloc.columns = ["Sport", "Athletes", "Exp. Medals"]
            st.dataframe(display_alloc.reset_index(drop=True), use_container_width=True, height=400)
        else:
            st.warning("No feasible sport allocation found. Try increasing delegation budget.")

    with col_right:
        st.markdown("""
        <h3 style="margin-top: 0;">
            <i class="fas fa-chart-bar" style="margin-right: 0.5em;"></i>Expected Medals by Sport
        </h3>
        """, unsafe_allow_html=True)
        if len(alloc) > 0:
            fig_bar = px.bar(
                alloc.sort_values("expected_medals"),
                x="expected_medals",
                y="Sport",
                orientation="h",
                color="expected_medals",
                color_continuous_scale="Blues",
                labels={"expected_medals": "Expected Medals", "Sport": ""},
                text="athletes_allocated",
            )
            fig_bar.update_traces(texttemplate="%{text} athletes", textposition="outside")
            fig_bar.update_layout(
                height=400,
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=40, t=10, b=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- Budget sensitivity ---
    st.markdown("""
    <h3>
        <i class="fas fa-chart-line" style="margin-right: 0.5em;"></i>Budget Sensitivity Analysis
    </h3>
    """, unsafe_allow_html=True)
    st.caption("Medal yield vs. delegation size — shows diminishing returns and marginal value of additional athletes")

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sens_df["delegation_budget"],
        y=sens_df["expected_medals"],
        mode="lines+markers",
        line=dict(color="#2196F3", width=2.5),
        marker=dict(size=5),
        name="Expected Medals",
    ))
    fig_sens.add_vline(
        x=run_budget,
        line_dash="dash", line_color="grey",
        annotation_text=f"Budget used ({run_budget})",
        annotation_position="top right",
    )
    fig_sens.update_layout(
        xaxis_title="Delegation Budget (athletes)",
        yaxis_title="Expected Total Medals",
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    # Shadow price curve
    if sens_df["shadow_price"].notna().any():
        st.caption("Marginal value per additional athlete — shows where budget increases yield the most medals")
        fig_shadow = px.line(
            sens_df.dropna(subset=["shadow_price"]),
            x="delegation_budget", y="shadow_price",
            labels={"delegation_budget": "Delegation Budget", "shadow_price": "Marginal Medals / Athlete"},
            color_discrete_sequence=["#FF5722"],
        )
        fig_shadow.add_vline(x=run_budget, line_dash="dash", line_color="grey")
        fig_shadow.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_shadow, use_container_width=True)

    st.divider()

    # --- r_s sensitivity ---
    st.markdown("""
    <h3>
        <i class="fas fa-flask" style="margin-right: 0.5em;"></i>Solution Robustness — Parameter Sensitivity
    </h3>
    """, unsafe_allow_html=True)
    st.caption("Tests solution robustness — what if sport medal rate estimates are off by ±20%?")

    fig_rsens = px.bar(
        rsens,
        x="r_delta", y="expected_medals",
        color="expected_medals",
        color_continuous_scale="RdYlGn",
        labels={"r_delta": "Parameter Change", "expected_medals": "Expected Medals"},
        text="expected_medals",
    )
    fig_rsens.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_rsens.update_layout(height=300, coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_rsens, use_container_width=True)

    # --- Model explanation expander ---
    with st.expander("ℹ️ How the Model Works"):
        st.markdown(f"""
        ### Optimization Model (Gurobi 12 ILP)

        **Objective:** Maximize total expected medals
        $$\\text{{Maximize}} \\quad \\sum_{{s \\in S}} r_s \\cdot x_s$$

        **Subject to:**
        | Constraint | Formula | Interpretation |
        |---|---|---|
        | Delegation budget | $\\sum_s x_s \\leq {run_budget}$ | Total athletes ≤ budget |
        | Participation minimum | $x_s \\geq m_s \\cdot y_s$ | Min athletes if entering sport |
        | IOC quota ceiling | $x_s \\leq M_s \\cdot y_s$ | Max athletes per sport |
        | Integrality | $x_s \\in \\mathbb{{Z}}_{{\\geq 0}},\\ y_s \\in \\{{0,1\\}}$ | Integer & binary |

        **Where:**
        - $r_s$ = expected medals per athlete in sport $s$ (estimated from Zero-Inflated Negative Binomial model, adjusted for your NSA profile)
        - $m_s$ = minimum athletes required to compete (10th percentile from historical data)
        - $M_s$ = IOC quota ceiling (90th percentile from historical data)

        **Your NSA Profile Adjustment Factor:** `{sp_adj['adj_factor'].iloc[0]:.4f}`
        (based on your GDP per capita, population, female ratio, and host status relative to median country)

        **Knowledge Base Connections:**
        - **(A)** Linear Programming, Integer Programming, Gurobi, Resource Allocation, Sensitivity Analysis
        - **(B)** Negative Binomial Distribution, Zero-Inflation (underlying count model), Risk Analysis
        - **(C)** Predictive Analytics (ZINB model coefficients used to compute r_s)
        - **(F)** Marginal Cost Analysis (shadow price on budget constraint)
        """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
MSE 433 Individual Project • Dhruv Sharma, University of Waterloo<br>
Data: 1960–2016 Olympic Games • Model: Zero-Inflated Negative Binomial (ZINB) + Gurobi 12
</div>
""", unsafe_allow_html=True)
