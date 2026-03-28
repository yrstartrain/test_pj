# test_pj - MLOps Practice Environment

A personal MLOps practice environment with automated monitoring and reporting.

## Structure

- `weather/` - Weather data fetching scripts
  - `weather_demo.py` - Fetches weather forecast for Kazo and Yoshimi cities
- `mlops/` - MLOps monitoring pipeline
  - `mlops_setup.py` - Train dummy churn prediction model (Logistic Regression)
  - `mlops_monitor.py` - Monitor data drift (KS test, PSI) and model performance
  - `reports/` - Auto-generated monitoring reports

## MLOps Pipeline

The monitoring pipeline detects:
- **Data Drift**: Kolmogorov-Smirnov test + Population Stability Index (PSI)
- **Model Performance**: Accuracy and F1 Score tracking
- **System Resources**: CPU, memory, disk usage

Runs hourly via Claude Dispatch scheduled tasks.
