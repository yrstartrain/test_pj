"""
mlops_retrain.py
================
MLOpsモニタリングでモデル劣化が検出された際に再学習を行うスクリプト。

使い方:
    python3 mlops_retrain.py

動作:
    1. 現在のrun_countを読み取り、ドリフト係数を算出
    2. 現在の本番データ分布でモデルを再学習（ドメイン適応）
    3. ベースライン統計を更新
    4. run_countをリセット（ドリフトカウンタをゼロに戻す）
    5. 再学習前後の性能比較レポートを出力
"""

import json, math, os, time
from datetime import datetime
import numpy as np

# --- パス設定（Windowsホストの永続パス） ---
TASK_DIR = r"C:\\Users\\海保航平\\Documents\\Claude\\Scheduled\\mlops-monitoring"
RUN_COUNT_FILE = os.path.join(TASK_DIR, "run_count.json")
MODEL_FILE     = os.path.join(TASK_DIR, "model_params.json")
BASELINE_FILE  = os.path.join(TASK_DIR, "baseline_stats.json")
RETRAIN_LOG    = os.path.join(TASK_DIR, "retrain_log.json")

os.makedirs(TASK_DIR, exist_ok=True)

FEATURES    = ["age", "usage_freq", "contract_months", "monthly_fee", "support_calls", "satisfaction_score"]
RANDOM_SEED = 42


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def generate_data(n, seed, drift=0.0):
    rng = np.random.RandomState(seed)
    age = rng.normal(40 + drift * 8, 10, n).clip(18, 80)
    uf  = rng.normal(15 - drift * 5, 5,  n).clip(0,  30)
    cm  = rng.normal(24 + drift * 10, 12, n).clip(1,  60)
    mf  = rng.normal(5000 + drift * 1500, 1500, n).clip(1000, 12000)
    sc  = rng.poisson(2 + drift * 3, n).clip(0, 15)
    sat = rng.normal(3.5 - drift * 1.0, 0.8, n).clip(1, 5)
    logit = (-3.0
             + 0.02  * (age - 40)
             - 0.10  * (uf  - 15)
             - 0.05  * (cm  - 24)
             + 0.0003* (mf  - 5000)
             + 0.30  * sc
             - 0.80  * (sat - 3.5)
             + drift * 1.5)
    prob  = sigmoid(logit)
    churn = (rng.uniform(0, 1, n) < prob).astype(int)
    return np.column_stack([age, uf, cm, mf, sc, sat]), churn


def train_model(X, y, lr=0.05, epochs=2000):
    mean = X.mean(0); std = X.std(0) + 1e-8
    Xs   = (X - mean) / std
    w    = np.zeros(Xs.shape[1]); b = 0.0
    for _ in range(epochs):
        yh  = sigmoid(Xs @ w + b)
        err = yh - y
        w  -= lr * (Xs.T @ err) / len(y)
        b  -= lr * err.mean()
    return w, b, mean, std


def predict(X, w, b, mean, std):
    Xs = (X - mean) / std
    return (sigmoid(Xs @ w + b) >= 0.5).astype(int)


def calc_metrics(yt, yp):
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    acc  = int(np.sum(yp == yt)) / len(yt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return acc, f1


def get_run_count():
    if os.path.exists(RUN_COUNT_FILE):
        with open(RUN_COUNT_FILE) as f:
            return json.load(f).get("count", 0)
    return 0


def save_model(w, b, mean, std):
    params = {
        "weights": w.tolist(),
        "bias":    float(b),
        "mean":    mean.tolist(),
        "std":     std.tolist(),
        "trained_at": datetime.now().isoformat(),
    }
    with open(MODEL_FILE, "w") as f:
        json.dump(params, f, indent=2)


def save_baseline(X, y, drift):
    stats = {}
    for i, feat in enumerate(FEATURES):
        stats[feat] = {
            "mean": float(X[:, i].mean()),
            "std":  float(X[:, i].std()),
            "min":  float(X[:, i].min()),
            "max":  float(X[:, i].max()),
        }
    stats["churn_rate"]   = float(y.mean())
    stats["drift_factor"] = drift
    stats["updated_at"]   = datetime.now().isoformat()
    with open(BASELINE_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def append_retrain_log(entry):
    log = []
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG) as f:
            log = json.load(f)
    log.append(entry)
    with open(RETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def main():
    t0  = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  MLOps 再学習スクリプト")
    print(f"  実行日時: {now}")
    print("=" * 60)

    old_run = get_run_count()
    old_drift = min(old_run * 0.08, 0.8)
    print(f"\n[1] 現在の状態")
    print(f"    実行回数: #{old_run}  /  ドリフト係数: {old_drift:.3f}")

    print("\n[2] 再学習前の性能測定...")
    X_orig, y_orig = generate_data(2000, RANDOM_SEED, drift=0.0)
    w_old, b_old, mean_old, std_old = train_model(X_orig, y_orig)
    X_prod, y_prod = generate_data(500, old_run * 100, old_drift)
    yp_old = predict(X_prod, w_old, b_old, mean_old, std_old)
    acc_before, f1_before = calc_metrics(y_prod, yp_old)
    print(f"    Accuracy（再学習前）: {acc_before:.4f}")
    print(f"    F1 Score（再学習前）: {f1_before:.4f}")

    print("\n[3] 再学習データ生成...")
    X_new_base, y_new_base = generate_data(2000, RANDOM_SEED, drift=0.0)
    X_new_prod, y_new_prod = generate_data(1000, old_run * 999, old_drift)
    X_retrain = np.vstack([X_new_base, X_new_prod])
    y_retrain = np.concatenate([y_new_base, y_new_prod])
    print(f"    再学習データ: {len(X_retrain)}件（ベースライン2000 + 現本番1000）")
    print(f"    離反率: {y_retrain.mean() * 100:.1f}%")

    print("\n[4] 再学習実行中...")
    w_new, b_new, mean_new, std_new = train_model(X_retrain, y_retrain)
    print("    完了")

    print("\n[5] 再学習後の性能測定...")
    yp_new = predict(X_prod, w_new, b_new, mean_new, std_new)
    acc_after, f1_after = calc_metrics(y_prod, yp_new)
    print(f"    Accuracy（再学習後）: {acc_after:.4f}")
    print(f"    F1 Score（再学習後）: {f1_after:.4f}")

    print("\n[6] モデル・ベースライン保存...")
    save_model(w_new, b_new, mean_new, std_new)
    save_baseline(X_retrain, y_retrain, old_drift)
    with open(RUN_COUNT_FILE, "w") as f:
        json.dump({"count": 0}, f)
    print(f"    モデルパラメータ保存: {MODEL_FILE}")
    print(f"    ベースライン統計保存: {BASELINE_FILE}")
    print(f"    run_count リセット: #{old_run} -> #0")

    elapsed = time.time() - t0
    log_entry = {
        "retrained_at":    now,
        "trigger_run":     old_run,
        "trigger_drift":   old_drift,
        "acc_before":      round(acc_before, 4),
        "f1_before":       round(f1_before, 4),
        "acc_after":       round(acc_after, 4),
        "f1_after":        round(f1_after, 4),
        "acc_improvement": round(acc_after - acc_before, 4),
        "f1_improvement":  round(f1_after - f1_before, 4),
        "elapsed_sec":     round(elapsed, 2),
    }
    append_retrain_log(log_entry)
    print(f"    再学習ログ追記: {RETRAIN_LOG}")

    acc_diff = acc_after - acc_before
    f1_diff  = f1_after  - f1_before

    print("\n" + "=" * 60)
    print("  再学習サマリ")
    print("=" * 60)
    print(f"  トリガー条件  : run #{old_run} / ドリフト係数 {old_drift:.3f}")
    print(f"  処理時間      : {elapsed:.2f}秒")
    print(f"  Accuracy : 再学習前 {acc_before:.4f} -> 再学習後 {acc_after:.4f} ({acc_diff:+.4f})")
    print(f"  F1 Score : 再学習前 {f1_before:.4f} -> 再学習後 {f1_after:.4f} ({f1_diff:+.4f})")
    print()
    if acc_diff > 0:
        print("  \u2705 再学習成功: モデル性能が改善されました")
        print("     次回監視からドリフトカウンタがリセットされます")
    else:
        print("  \u26a0\ufe0f  性能改善が限定的です。データ量やハイパーパラメータの見直しを検討してください")
    print("=" * 60)


if __name__ == "__main__":
    main()
