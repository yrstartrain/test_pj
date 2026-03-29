# test_pj - MLOps Practice Environment

A personal MLOps practice environment with automated monitoring and reporting.

## Structure

- `weather/` - Weather data fetching scripts
  - `weather_demo.py` - Fetches weather forecast for Kazo and Yoshimi cities
- `mlops/` - MLOps monitoring pipeline
  - `mlops_setup.py` - Train dummy churn prediction model (Logistic Regression)
  - `mlops_monitor.py` - Monitor data drift (KS test, PSI) and model performance
  - `mlops_retrain.py` - Auto-retrain model when drift or accuracy degradation is detected
  - `reports/` - Auto-generated monitoring reports

## MLOps Pipeline

The monitoring pipeline detects:
- **Data Drift**: Kolmogorov-Smirnov test + Population Stability Index (PSI)
- **Model Performance**: Accuracy and F1 Score tracking
- **System Resources**: CPU, memory, disk usage

Runs hourly via Claude Dispatch scheduled tasks.

## Auto-Retraining

`mlops_retrain.py` reads the latest monitoring report and retrains only when triggered:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| PSI (data drift) | > 0.20 | Retrain |
| F1 Score | < 0.70 | Retrain |
| Accuracy | < 0.72 | Retrain |

If no conditions are met, retraining is skipped and the result is logged to `retrain_log.json`.
