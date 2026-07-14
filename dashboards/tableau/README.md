# Tableau Dashboard Setup

The batch scorer (`python -m churn_platform.score`) exports
`dashboards/tableau/scored_extract.csv` — the data source for the
stakeholder workbook.

## Connect
1. Open Tableau Desktop → **Connect → Text File** → select `scored_extract.csv`
   (or publish it to Tableau Server/Cloud as a published data source named
   `churn_risk_scores` and let the Airflow task `refresh_tableau_extract`
   refresh it after each monthly scoring run via `tabcmd refreshextracts`).

## Recommended sheets
| Sheet | Rows/Cols | Marks |
|---|---|---|
| Risk Tier Distribution | risk_tier → Columns, CNT(customer_id) → Rows | Bar |
| Revenue at Risk by Tier | risk_tier → Columns, SUM(revenue_at_risk) → Rows | Bar |
| Churn Prob vs LTV | churn_probability → Cols, predicted_ltv → Rows, risk_tier → Color | Scatter |
| Retention Target List | customer_id, churn_probability, predicted_ltv, revenue_at_risk | Table, filter risk_tier = critical/high |

## Calculated fields
- `Revenue at Risk` = `[churn_probability] * [predicted_ltv]` (already precomputed)
- `Priority Rank` = `RANK(SUM([revenue_at_risk]))`

Combine sheets into a dashboard with a risk_tier quick filter and publish.
