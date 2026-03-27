"""
mlops_setup.py
==============
MLOps練習環境 - セットアップスクリプト（初回のみ実行）

実行内容:
  - 顧客離反予測用ダミーデータの生成
  - ロジスティック回帰モデルをnumpyでスクラッチ実装・訓練
  - ベースラインデータ・モデルパラメータを保存

依存ライブラリ: numpy, pandas, psutil（標準インストール済みのもののみ使用）
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

RANDOM_SEED = 42
N_TRAIN = 2000
FEATURES = ["age", "usage_freq", "contract_months", "monthly_fee", "support_calls", "satisfaction_score"]


def generate_churn_data(n, seed=RANDOM_SEED, drift_factor=0.0):
    rng = np.random.RandomState(seed)
    age             = rng.normal(40 + drift_factor * 8,  10, n).clip(18, 80)
    usage_freq      = rng.normal(15 - drift_factor * 5,   5, n).clip(0, 30)
    contract_months = rng.normal(24 + drift_factor * 10, 12, n).clip(1, 60)
    monthly_fee     = rng.normal(5000 + drift_factor * 1500, 1500, n).clip(1000, 12000)
    support_calls   = rng.poisson(2 + drift_factor * 3, n).clip(0, 15)
    satisfaction    = rng.normal(3.5 - drift_factor * 1.0, 0.8, n).clip(1, 5)
    logit = (
        -3.0 + 0.02 * (age - 40) - 0.10 * (usage_freq - 15)
        - 0.05 * (contract_months - 24) + 0.0003 * (monthly_fee - 5000)
        + 0.30 * support_calls - 0.80 * (satisfaction - 3.5) + drift_factor * 1.5
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < prob_churn).astype(int)
    return pd.DataFrame({
        "age": age, "usage_freq": usage_freq, "contract_months": contract_months,
        "monthly_fee": monthly_fee, "support_calls": support_calls,
        "satisfaction_score": satisfaction, "churn": churn,
    })


class LogisticRegression:
    def __init__(self, lr=0.05, n_iter=2000, tol=1e-6):
        self.lr = lr; self.n_iter = n_iter; self.tol = tol
        self.weights = None; self.bias = 0.0; self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _standardize(self, X, fit=False):
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def fit(self, X, y):
        X = self._standardize(X, fit=True)
        n, m = X.shape
        self.weights = np.zeros(m); self.bias = 0.0
        for i in range(self.n_iter):
            y_hat = self._sigmoid(X @ self.weights + self.bias)
            error = y_hat - y
            self.weights -= self.lr * (X.T @ error) / n
            self.bias    -= self.lr * error.mean()
            loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
            self.loss_history.append(loss)
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                break
        return self

    def predict_proba(self, X):
        return self._sigmoid(self._standardize(X) @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    print("=" * 60)
    print("  MLOps 練習環境 セットアップ開始")
    print("=" * 60)

    print("\n[1/4] 訓練データ生成中...")
    df_train = generate_churn_data(N_TRAIN, seed=RANDOM_SEED, drift_factor=0.0)
    print(f"  データ件数: {len(df_train):,} 件  離反率: {df_train['churn'].mean()*100:.1f}%")

    print("\n[2/4] モデル訓練中...")
    X = df_train[FEATURES].values
    y = df_train["churn"].values
    model = LogisticRegression()
    model.fit(X, y)

    print("\n[3/4] 訓練精度評価...")
    metrics = compute_metrics(y, model.predict(X))
    for k, v in metrics.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")

    print("\n[4/4] モデルとベースラインデータを保存中...")
    params = {
        "weights": model.weights.tolist(), "bias": model.bias,
        "mean_": model.mean_.tolist(), "std_": model.std_.tolist(),
        "features": FEATURES, "trained_at": datetime.now().isoformat(),
        "baseline_metrics": metrics,
    }
    with open(os.path.join(MODEL_DIR, "model_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    baseline_stats = {}
    for feat in FEATURES:
        col = df_train[feat]
        baseline_stats[feat] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "min": float(col.min()), "max": float(col.max()),
            "p25": float(col.quantile(0.25)), "p50": float(col.quantile(0.50)), "p75": float(col.quantile(0.75)),
            "samples": col.sample(min(500, len(col)), random_state=RANDOM_SEED).tolist(),
        }
    with open(os.path.join(MODEL_DIR, "baseline_stats.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  セットアップ完了！ 次のコマンドで監視を開始: python3 mlops_monitor.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
