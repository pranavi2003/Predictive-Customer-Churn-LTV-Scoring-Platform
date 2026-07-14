"""
Churn Risk Monitoring Dashboard (Streamlit).

Local/open-source counterpart of the Tableau workbook shipped to business
stakeholders — same views: risk-tier distribution, revenue at risk by
segment, decile lift, and a drill-down table for retention targeting.

Run: streamlit run dashboards/streamlit_app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Risk Command Center", layout="wide")
st.title("📉 Customer Churn Risk & LTV Command Center")

scored = pd.read_parquet("data/processed/scored_customers.parquet")
lift = pd.read_csv("reports/decile_lift.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers Scored", f"{len(scored):,}")
c2.metric("Critical-Risk Customers", f"{(scored.risk_tier == 'critical').sum():,}")
c3.metric("Total Revenue at Risk", f"${scored.revenue_at_risk.sum():,.0f}")
c4.metric("Avg Predicted LTV", f"${scored.predicted_ltv.mean():,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Customers by Risk Tier")
    st.bar_chart(scored["risk_tier"].value_counts())
with right:
    st.subheader("Revenue at Risk by Tier")
    st.bar_chart(scored.groupby("risk_tier", observed=True)["revenue_at_risk"].sum())

st.subheader("Decile Lift — Model Targeting Power")
st.dataframe(lift.round(3), use_container_width=True)
st.caption("Decile 1 = highest-risk 10%. Lift shows how concentrated real churners "
           "are in top-scored deciles — the basis for retention-campaign targeting.")

st.subheader("Churn Probability vs Predicted LTV (retention priority map)")
sample = scored.sample(min(5000, len(scored)), random_state=0)
st.scatter_chart(sample, x="churn_probability", y="predicted_ltv", color="risk_tier")

st.subheader("Top Retention Targets (high LTV x high churn risk)")
st.dataframe(
    scored.nlargest(100, "revenue_at_risk")[
        ["customer_id", "churn_probability", "predicted_ltv",
         "revenue_at_risk", "risk_tier", "tenure_months", "monthly_charges"]
    ].reset_index(drop=True),
    use_container_width=True,
)
