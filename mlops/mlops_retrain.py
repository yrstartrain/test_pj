"""
mlops_retrain.py
================
MLOpsモニタリングレポートを解析し、ドリフトまたは精度低下が検知された場合のみ
モデルを再学習するスクリプト。

使い方:
    python3 mlops_retrain.py

動作:
    1. reports/ 内の最新レポート(.md)を読み取りトリガー判定
    2. トリガー条件を満たす場合のみ再学習を実行
    3. 現在の本番データ分布でモデルを再学習（ドメイン適応）
    4. ベースライン統計を更新
    5. run_countをリセット（ドリフトカウンタをゼロに戻す）
    6. 再学習前後の性能比較レポートを出力

トリガー条件（いずれか一つで発動）:
    - PSI > 0.2          : データドリフト検知
    - F1スコア < 0.70    : モデル精度低下
    - Accuracy < 0.72    : モデル精度低下
"""

import glob, json, math, os, re, time
from datetime import datetime
import numpy as np

# --- パス設定 (Windowsホストの永続パス) ---
TASK_DIR       = r"C:\Users\海保航平\Documents\Claude\Scheduled\mlops-monitoring"
RUN_COUNT_FILE = os.path.join(TASK_DIR, "run_count.json")
MODEL_FILE     = os.path.join(TASK_DIR, "model_params.json")
BASELINE_FILE  = os.path.join(TASK_DIR, "baseline_stats.json")
RETRAIN_LOG    = os.path.join(TASK_DIR, "retrain_log.json")
REPORT_DIR     = os.path.join(TASK_DIR, "reports")

os.makedirs(TASK_DIR, exist_ok=True)

FEATURES    = ["age", "usage_freq", "contract_months", "monthly_fee", "support_calls", "satisfaction_score"]
RANDOM_SEED = 42

# --- 再学習トリガー閾値 ---
PSI_THRESHOLD      = 0.20
F1_THRESHOLD       = 0.70
ACCURACY_THRESHOLD = 0.72


# ════════════════════════════════════════════════════════════
# 1. モニタリングレポート解析
# ════════════════════════════════════════════════════════════

def get_latest_report():
    """reports/ ディレクトリ内の最新レポートを返す。"""
    pattern = os.path.join(REPORT_DIR, "*.md")
    reports = [r for r in glob.glob(pattern) if "retrain" not in os.path.basename(r)]
    if not reports:
        return None
    return max(reports, key=os.path.getmtime)


def parse_report(report_path):
    """
    マークダウンレポートからPSI・Accuracy・F1を抽出する。
    返り値例:
      {"psi_max": 0.25, "accuracy": 0.80, "f1_score": 0.68,
       "drift_detected": True, "perf_degraded": True}
    """
    with open(report_path, encoding="utf-8") as f:
        text = f.read()

    metrics = {"psi_values": {}, "accuracy": None, "f1_score": None}

    # PSI テーブル行: | feature | 0.25 | ... |
    for m in re.finditer(r"\|\s*([^\|]+?)\s*\|\s*([0-9]+\.[0-9]+)\s*\|", text):
        feat, val = m.group(1).strip(), float(m.group(2))
        if feat.lower() not in ("feature", "psi", "variable", "指標", "特徴量"):
            metrics["psi_values"][feat] = val

    # Accuracy / F1 Score
    for label, key in [("accuracy", "accuracy"), (r"f1[_\s]?score", "f1_score"), ("f1", "f1_score")]:
        m = re.search(rf"{label}\s*[:\|]\s*([0-9]+\.?[0-9]*)", text, re.IGNORECASE)
        if m and metrics[key] is None:
            metrics[key] = float(m.group(1))

    psi_max = max(metrics["psi_values"].values()) if metrics["psi_values"] else 0.0
    metrics["psi_max"] = psi_max
    metrics["drift_detected"]  = psi_max >= PSI_THRESHOLD
    metrics["perf_degraded"]   = (
        (metrics["f1_score"]  is not None and metrics["f1_score"]  < F1_THRESHOLD) or
        (metrics["accuracy"]  is not None and metrics["accuracy"]  < ACCURACY_THRESHOLD)
    )
    return metrics


def check_trigger():
    """
    最新レポートを解析してトリガー判定を行う。
    戻り値: (should_retrain: bool, reasons: list[str], metrics: dict)
    """
    report_path = get_latest_report()
    if report_path is None:
        print("  [!] レポートファイルが見つかりません。run_countのみで判断します。")
        return True, ["レポート未検出（フォールバック）"], {}

    print(f"  参照レポート: {os.path.basename(report_path)}")
    metrics = parse_report(report_path)

    reasons = []
    if metrics["drift_detected"]:
        reasons.append(f"データドリフト検知 (PSI max={metrics['psi_max']:.3f} >= {PSI_THRESHOLD})")
    if metrics["perf_degraded"]:
        if metrics["f1_score"] is not None and metrics["f1_score"] < F1_THRESHOLD:
            reasons.append(f"F1スコア低下 ({metrics['f1_score']:.4f} < {F1_THRESHOLD})")
        if metrics["accuracy"] is not None and metrics["accuracy"] < ACCURACY_THRESHOLD:
            reasons.append(f"Accuracy低下 ({metrics['accuracy']:.4f} < {ACCURACY_THRESHOLD})")

    return bool(reasons), reasons, metrics


# ════════════════════════════════════════════════════════════
# 2. モデル学習ユーティリティ（既存ロジック）
# ════════════════════════════════════════════════════════════

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def generate_data(n, seed, drift=0.0):
    rng = np.random.RandomState(seed)
    age = rng.normal(40 + drift * 8, 10, n).clip(18, 80)
    uf  = rng.normal(15 - drift * 5,  5, n).clip(0,  30)
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
    Xs = (X - mean) / std
    w  = np.zeros(Xs.shape[1]); b = 0.0
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
    tp   = int(np.sum((yp == 1) & (yt == 1)))
    fp   = int(np.sum((yp == 1) & (yt == 0)))
    fn   = int(np.sum((yp == 0) & (yt == 1)))
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
        "weights":    w.tolist(),
        "bias":       float(b),
        "mean":       mean.tolist(),
        "std":        std.tolist(),
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


# ════════════════════════════════════════════════════════════
# 3. メイン処理
# ════════════════════════════════════════════════════════════

def main():
    t0  = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  MLOps 再学習スクリプト")
    print(f"  実行日時: {now}")
    print("=" * 60)

    # ── [1] トリガー判定 ──────────────────────────────────────
    print("\n[1] モニタリングレポート解析...")
    old_run   = get_run_count()
    old_drift = min(old_run * 0.08, 0.8)
    print(f"    実行回数: {old_run}  /  ドリフト係数: {old_drift:.3f}")

    should_retrain, reasons, report_metrics = check_trigger()

    if not should_retrain:
        print("\n  トリガー条件を満たしていません。再学習をスキップします。")
        print(f"  PSI max : {report_metrics.get('psi_max', 'N/A')}")
        print(f"  Accuracy: {report_metrics.get('accuracy', 'N/A')}")
        print(f"  F1 Score: {report_metrics.get('f1_score', 'N/A')}")
        print("=" * 60)
        append_retrain_log({
            "retrained_at":  now,
            "skipped":       True,
            "reason":        "トリガー条件未達",
            "report_psi":    report_metrics.get("psi_max"),
            "report_acc":    report_metrics.get("accuracy"),
            "report_f1":     report_metrics.get("f1_score"),
        })
        return

    print(f"\n  ⚠️  再学習トリガー発動:")
    for r in reasons:
        print(f"     - {r}")

    # ── [2] 再学習前の性能測定 ───────────────────────────────
    print("\n[2] 再学習前の性能測定...")
    X_orig, y_orig = generate_data(2000, RANDOM_SEED, drift=0.0)
    w_old, b_old, mean_old, std_old = train_model(X_orig, y_orig)
    X_prod, y_prod = generate_data(500, old_run * 100, old_drift)
    yp_old = predict(X_prod, w_old, b_old, mean_old, std_old)
    acc_before, f1_before = calc_metrics(y_prod, yp_old)
    print(f"    Accuracy (再学習前): {acc_before:.4f}")
    print(f"    F1 Score (再学習前): {f1_before:.4f}")

    # ── [3] 再学習データ生成 ─────────────────────────────────
    print("\n[3] 再学習データ生成...")
    X_new_base, y_new_base = generate_data(2000, RANDOM_SEED, drift=0.0)
    X_new_prod, y_new_prod = generate_data(1000, old_run * 999, old_drift)
    X_retrain = np.vstack([X_new_base, X_new_prod])
    y_retrain = np.concatenate([y_new_base, y_new_prod])
    print(f"    再学習データ: {len(X_retrain)}件 (ベースライン2000 + 現本番1000)")
    print(f"    離反率: {y_retrain.mean() * 100:.1f}%")

    # ── [4] 再学習実行 ───────────────────────────────────────
    print("\n[4] 再学習実行中...")
    w_new, b_new, mean_new, std_new = train_model(X_retrain, y_retrain)
    print("    完了")

    # ── [5] 再学習後の性能測定 ───────────────────────────────
    print("\n[5] 再学習後の性能測定...")
    yp_new = predict(X_prod, w_new, b_new, mean_new, std_new)
    acc_after, f1_after = calc_metrics(y_prod, yp_new)
    print(f"    Accuracy (再学習後): {acc_after:.4f}")
    print(f"    F1 Score (再学習後): {f1_after:.4f}")

    # ── [6] モデル・ベースライン保存 ─────────────────────────
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
        "skipped":         False,
        "trigger_reasons": reasons,
        "trigger_run":     old_run,
        "trigger_drift":   old_drift,
        "report_psi":      report_metrics.get("psi_max"),
        "report_acc":      report_metrics.get("accuracy"),
        "report_f1":       report_metrics.get("f1_score"),
        "acc_before":      round(acc_before, 4),
        "f1_before":       round(f1_before, 4),
        "acc_after":       round(acc_after, 4),
        "f1_after":        round(f1_after, 4),
        "acc_improvement": round(acc_after - acc_before, 4),
        "f1_improvement":  round(f1_after  - f1_before, 4),
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
        print("  ✅ 再学習成功: モデル性能が改善されました")
        print("     次回監視からドリフトカウンタがリセットされます")
    else:
        print("  ⚠️  性能改善が限定的です。データ量やハイパーパラメータの見直しを検討してください")
    print("=" * 60)


if __name__ == "__main__":
    main()
