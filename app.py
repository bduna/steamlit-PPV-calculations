# app.py
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------
# Page setup
# --------------------------
st.set_page_config(
    page_title="PPV Math & A/B Profit Calculator",
    page_icon="📈",
    layout="wide"
)

# --------------------------
# Defaults & helpers
# --------------------------
DEFAULTS = dict(
    traffic=250_000,   # visitors in the selected period
    cr=2.0,            # %
    aov=85.0,          # $
    gm=60.0,           # %
    cpv=0.35,          # $ cost per visitor
    project_cost=40_000.0,  # $ one-time cost for ROI/payback math
    period_label="30 days"  # purely cosmetic
)

def reset_to_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    # Scenario B mirrors A initially
    st.session_state.update({f"b_{k}": st.session_state[k] for k in ["cr", "aov", "gm", "cpv"]})
    st.session_state["traffic_b"] = st.session_state["traffic"]
    st.success("Inputs reset to sensible defaults.")

# One-time initialization
if "initialized" not in st.session_state:
    reset_to_defaults()
    st.session_state["initialized"] = True

def pct_to_dec(pct: float) -> float:
    return float(pct) / 100.0

def ppv(cr_pct: float, aov: float, gm_pct: float, cpv: float) -> float:
    """
    PPV = (CR × AOV × GM) − CPV
    CR and GM accepted as percentages (0-100).
    """
    cr = pct_to_dec(cr_pct)
    gm = pct_to_dec(gm_pct)
    return cr * aov * gm - cpv

def profit(visitors: int, ppv_value: float) -> float:
    """Profit over the period = visitors × PPV."""
    return float(visitors) * float(ppv_value)

def safe_float(x: float) -> float:
    return float(x)

@dataclass
class Scenario:
    visitors: int
    cr_pct: float
    aov: float
    gm_pct: float
    cpv: float

    @property
    def ppv(self) -> float:
        return ppv(self.cr_pct, self.aov, self.gm_pct, self.cpv)

    @property
    def total_profit(self) -> float:
        return profit(self.visitors, self.ppv)

def ab_delta(sA: Scenario, sB: Scenario) -> Dict[str, float]:
    """Compute A vs B deltas and lifts."""
    d = {}
    d["ppv_A"] = sA.ppv
    d["ppv_B"] = sB.ppv
    d["ppv_diff"] = sB.ppv - sA.ppv
    d["ppv_lift_pct"] = (d["ppv_diff"] / abs(sA.ppv)) * 100.0 if sA.ppv != 0 else np.nan

    d["profit_A"] = sA.total_profit
    d["profit_B"] = sB.total_profit
    d["profit_diff"] = d["profit_B"] - d["profit_A"]
    d["profit_lift_pct"] = (d["profit_diff"] / abs(d["profit_A"])) * 100.0 if d["profit_A"] != 0 else np.nan
    return d

def tornado_sensitivity(s: Scenario, change_pct: float = 10.0) -> pd.DataFrame:
    """
    One-way sensitivity on PPV for ±change_pct on each input.
    Returns a tidy DataFrame with ΔPPV for +/- shocks.
    """
    base = s.ppv
    rows = []
    shocks = [ -change_pct, +change_pct ]

    # Factors: CR (%), AOV ($), GM (%), CPV ($)
    for factor in ["CR (%)", "AOV ($)", "GM (%)", "CPV ($)"]:
        for shock in shocks:
            if factor == "CR (%)":
                new = ppv(s.cr_pct * (1 + shock/100), s.aov, s.gm_pct, s.cpv)
            elif factor == "AOV ($)":
                new = ppv(s.cr_pct, s.aov * (1 + shock/100), s.gm_pct, s.cpv)
            elif factor == "GM (%)":
                new = ppv(s.cr_pct, s.aov, s.gm_pct * (1 + shock/100), s.cpv)
            else:
                # CPV increases reduce PPV directly
                new = ppv(s.cr_pct, s.aov, s.gm_pct, s.cpv * (1 + shock/100))

            rows.append({
                "Factor": factor,
                "Shock": f"{'+' if shock>0 else ''}{shock:.0f}%",
                "ΔPPV ($/visitor)": new - base
            })
    df = pd.DataFrame(rows)
    # Sort by absolute impact of +shock to give a "tornado-ish" view
    return df.sort_values(by="ΔPPV ($/visitor)", key=lambda c: c.abs(), ascending=False)

# --------------------------
# Sidebar inputs
# --------------------------
with st.sidebar:
    st.title("Inputs")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        visitors = st.number_input(
            "Visitors in period",
            min_value=0, step=1000, value=st.session_state.get("traffic", DEFAULTS["traffic"]),
            help="Total site visitors in the selected time window."
        )
        st.session_state["traffic"] = visitors

        cr = st.number_input(
            "Conversion Rate (%)",
            min_value=0.0, step=0.05, value=st.session_state.get("cr", DEFAULTS["cr"]),
            help="Percent of visitors who convert (orders/visitors × 100)."
        )
        st.session_state["cr"] = cr

        aov = st.number_input(
            "Average Order Value ($)",
            min_value=0.0, step=1.0, value=st.session_state.get("aov", DEFAULTS["aov"]),
            help="Average revenue per order."
        )
        st.session_state["aov"] = aov

    with col_b:
        gm = st.number_input(
            "Gross Margin (%)",
            min_value=0.0, max_value=100.0, step=0.5, value=st.session_state.get("gm", DEFAULTS["gm"]),
            help="Percent of revenue kept after COGS."
        )
        st.session_state["gm"] = gm

        cpv = st.number_input(
            "Cost per Visitor ($)",
            min_value=0.0, step=0.01, value=st.session_state.get("cpv", DEFAULTS["cpv"]),
            help="Average blended cost to acquire one visitor (ads, affiliates, etc.)."
        )
        st.session_state["cpv"] = cpv

        project_cost = st.number_input(
            "Project/Experiment Cost ($)",
            min_value=0.0, step=1000.0, value=st.session_state.get("project_cost", DEFAULTS["project_cost"]),
            help="Use this to compute ROI and payback from A→B uplift."
        )
        st.session_state["project_cost"] = project_cost

    st.markdown("---")

    compare_mode = st.checkbox("Compare a second scenario (B)", value=True)
    if compare_mode:
        st.subheader("Scenario B (target)")
        col1, col2 = st.columns(2)
        with col1:
            visitors_b = st.number_input(
                "Visitors in period (B)",
                min_value=0, step=1000, value=st.session_state.get("traffic_b", st.session_state["traffic"]),
                help="If traffic allocation differs between A and B, change this."
            )
            st.session_state["traffic_b"] = visitors_b

            cr_b = st.number_input(
                "CR B (%)", min_value=0.0, step=0.05,
                value=st.session_state.get("b_cr", st.session_state["cr"])
            )
            st.session_state["b_cr"] = cr_b

        with col2:
            aov_b = st.number_input(
                "AOV B ($)", min_value=0.0, step=1.0,
                value=st.session_state.get("b_aov", st.session_state["aov"])
            )
            st.session_state["b_aov"] = aov_b

            gm_b = st.number_input(
                "GM B (%)", min_value=0.0, max_value=100.0, step=0.5,
                value=st.session_state.get("b_gm", st.session_state["gm"])
            )
            st.session_state["b_gm"] = gm_b

        cpv_b = st.number_input(
            "CPV B ($)", min_value=0.0, step=0.01,
            value=st.session_state.get("b_cpv", st.session_state["cpv"])
        )
        st.session_state["b_cpv"] = cpv_b

    st.markdown("---")
    if st.button("Reset to defaults"):
        reset_to_defaults()

# --------------------------
# Build scenarios
# --------------------------
A = Scenario(
    visitors=st.session_state["traffic"],
    cr_pct=st.session_state["cr"],
    aov=st.session_state["aov"],
    gm_pct=st.session_state["gm"],
    cpv=st.session_state["cpv"],
)
if compare_mode:
    B = Scenario(
        visitors=st.session_state["traffic_b"],
        cr_pct=st.session_state["b_cr"],
        aov=st.session_state["b_aov"],
        gm_pct=st.session_state["b_gm"],
        cpv=st.session_state["b_cpv"],
    )

# --------------------------
# Header & equations
# --------------------------
st.title("📈 PPV Math & A/B Profit Calculator")

with st.expander("Show key equations", expanded=False):
    st.latex(r"\textbf{PPV} \;=\; (\text{CR}\times \text{AOV}\times \text{GM}) \;-\; \text{CPV}")
    st.latex(r"\textbf{Profit over period} \;=\; \text{Visitors}\times \text{PPV}")
    st.latex(r"\Delta \text{PPV} \;=\; \text{PPV}_B-\text{PPV}_A")
    st.latex(r"\%\text{Lift} \;=\; \dfrac{\Delta \text{PPV}}{|\text{PPV}_A|}\times 100")
    st.latex(r"\text{Incremental Profit} \;=\; \text{Visitors}_B\times \Delta \text{PPV}")
    st.latex(r"\text{ROI} \;=\; \dfrac{\text{Incremental Profit}-\text{Cost}}{\text{Cost}}\times 100")

# --------------------------
# Single-scenario cards
# --------------------------
colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("Scenario A (current)")
    mA = {
        "Visitors (period)": f"{A.visitors:,}",
        "CR (%)": f"{A.cr_pct:.2f}",
        "AOV ($)": f"{A.aov:,.2f}",
        "GM (%)": f"{A.gm_pct:.2f}",
        "CPV ($)": f"{A.cpv:,.2f}",
        "PPV ($/visitor)": f"{A.ppv:,.4f}",
        "Profit (period, $)": f"{A.total_profit:,.2f}",
    }
    st.dataframe(pd.DataFrame.from_dict(mA, orient="index", columns=["Value"]))

with colB:
    if compare_mode:
        st.subheader("Scenario B (target)")
        mB = {
            "Visitors (period)": f"{B.visitors:,}",
            "CR (%)": f"{B.cr_pct:.2f}",
            "AOV ($)": f"{B.aov:,.2f}",
            "GM (%)": f"{B.gm_pct:.2f}",
            "CPV ($)": f"{B.cpv:,.2f}",
            "PPV ($/visitor)": f"{B.ppv:,.4f}",
            "Profit (period, $)": f"{B.total_profit:,.2f}",
        }
        st.dataframe(pd.DataFrame.from_dict(mB, orient="index", columns=["Value"]))

# --------------------------
# A/B comparison
# --------------------------
if compare_mode:
    st.markdown("### A → B Comparison")
    d = ab_delta(A, B)

    comp_df = pd.DataFrame({
        "Metric": [
            "PPV (A)", "PPV (B)", "ΔPPV (B−A)", "% PPV lift",
            "Profit (A)", "Profit (B)", "ΔProfit (B−A)", "% Profit lift"
        ],
        "Value": [
            f"${d['ppv_A']:.4f}",
            f"${d['ppv_B']:.4f}",
            f"${d['ppv_diff']:.4f}",
            f"{d['ppv_lift_pct']:.2f}%",
            f"${d['profit_A']:,.2f}",
            f"${d['profit_B']:,.2f}",
            f"${d['profit_diff']:,.2f}",
            f"{d['profit_lift_pct']:.2f}%"
        ]
    })
    st.dataframe(comp_df, use_container_width=True)

    # ROI & Payback (if a project cost is provided)
    if project_cost > 0:
        roi_pct = ((d["profit_diff"] - project_cost) / project_cost) * 100.0
        payback_visitors_needed = project_cost / d["ppv_diff"] if d["ppv_diff"] > 0 else np.inf
        payback_text = "N/A"
        if np.isfinite(payback_visitors_needed) and B.visitors > 0:
            # Approximate payback in "periods equivalent" based on B.visitors per period
            periods = payback_visitors_needed / B.visitors
            payback_text = f"~{periods:.2f} × your selected period"

        st.info(
            f"**Incremental profit (B−A):** ${d['profit_diff']:,.2f}  \n"
            f"**ROI on project cost:** {roi_pct:.2f}%  \n"
            f"**Payback visitors needed:** {payback_visitors_needed:,.0f}  \n"
            f"**Payback time (rough):** {payback_text}"
        )

# --------------------------
# Sensitivity
# --------------------------
with st.expander("Quick sensitivity (±10% shocks on each factor)", expanded=False):
    s_df = tornado_sensitivity(A, change_pct=10.0)
    st.dataframe(s_df, use_container_width=True)
 # Guard: make sure the columns exist
required = {"Factor", "Shock", "ΔPPV ($/visitor)"}
missing = required - set(s_df.columns)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Pivot to wide so Streamlit can plot grouped bars
df = s_df[["Factor", "Shock", "ΔPPV ($/visitor)"]].copy()
pivot = df.pivot(index="Factor", columns="Shock", values="ΔPPV ($/visitor)")
st.bar_chart(pivot)  # grouped bars by Shock for each Factor

# --------------------------
# Glossary (short, practical)
# --------------------------
with st.expander("Plain-English glossary", expanded=False):
    st.markdown(
        """
- **Conversion Rate (CR)** — orders ÷ visitors × 100.
- **Average Order Value (AOV)** — average revenue per order.
- **Gross Margin (GM)** — % of revenue kept after COGS.
- **Cost per Visitor (CPV)** — blended cost to acquire one visitor.
- **PPV** — profit per visitor = (CR × AOV × GM) − CPV.
- **Profit (period)** — visitors × PPV.
- **ΔPPV**, **% lift** — difference and relative change of PPV from A to B.
- **Incremental profit** — visitors_B × ΔPPV.
- **ROI** — ((incremental profit − project cost) ÷ project cost) × 100.
        """
    )

st.caption("Tip: Use the sidebar to tweak inputs. Click **Reset to defaults** any time.")


