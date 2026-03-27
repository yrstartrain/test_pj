"""
mlops_monitor.py
================
MLOps練習環境 - 監視スクリプト（定期実行）

実行内容:
  1. 新しい本番データをシミュレート（時間経過でドリフトあり）
  2. データドリフト検知（KS検定・PSI）
  3. モデルパフォーマンス監視
  4. システムリソース監視
  5. Markdownレポートを reports/ に保存
"""

import json, math, os, sys, time
from datetime import datetime
import numpy as np
import pandas as pd
import psutil

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

PARAMS_PATH   = os.path.join(MODEL_DIR, "model_params.json")
BASELINE_PATH = os.path.join(MODEL_DIR, "baseline_stats.json")

KS_P_THRESHOLD      = 0.05
PSI_WARN_THRESHOLD  = 0.1
PSI_ALERT_THRESHOLD = 0.2
PERF_DROP_THRESHOLD = 0.05
CPU_ALERT = 80.0; MEMORY_ALERT = 85.0; DISK_ALERT = 90.0

RUN_COUNT_FILE = os.path.join(MODEL_DIR, "run_count.json")

def get_run_count():
    if os.path.exists(RUN_COUNT_FILE):
        with open(RUN_COUNT_FILE) as f: return json.load(f).get("count", 0)
    return 0

def increment_run_count():
    count = get_run_count() + 1
    with open(RUN_COUNT_FILE, "w") as f: json.dump({"count": count}, f)
    return count

FEATURES = ["age", "usage_freq", "contract_months", "monthly_fee", "support_calls", "satisfaction_score"]

def generate_production_data(n, seed, drift_factor):
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
    prob_churn = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
    churn = (rng.uniform(0, 1, n) < prob_churn).astype(int)
    return pd.DataFrame({
        "age": age, "usage_freq": usage_freq, "contract_months": contract_months,
        "monthly_fee": monthly_fee, "support_calls": support_calls,
        "satisfaction_score": satisfaction, "churn": churn,
    })

class LogisticRegressionInference:
    def __init__(self, params):
        self.weights = np.array(params["weights"]); self.bias = params["bias"]
        self.mean_ = np.array(params["mean_"]); self.std_ = np.array(params["std_"])
    def predict_proba(self, X):
        Xs = (X - self.mean_) / self.std_
        return 1 / (1 + np.exp(-np.clip(Xs @ self.weights + self.bias, -500, 500)))
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred==1)&(y_true==1))); fp = int(np.sum((y_pred==1)&(y_true==0)))
    fn = int(np.sum((y_pred==0)&(y_true==1))); tn = int(np.sum((y_pred==0)&(y_true==0)))
    n = len(y_true)
    accuracy  = (tp+tn)/n if n>0 else 0
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def ks_2samp(a, b):
    a = np.sort(a); b = np.sort(b)
    n1, n2 = len(a), len(b)
    all_vals = np.unique(np.concatenate([a, b]))
    cdf1 = np.searchsorted(a, all_vals, side="right") / n1
    cdf2 = np.searchsorted(b, all_vals, side="right") / n2
    ks_stat = float(np.max(np.abs(cdf1 - cdf2)))
    en = math.sqrt(n1*n2/(n1+n2))
    z  = (en + 0.12 + 0.11/en) * ks_stat
    p_value = 0.0
    for k in range(1, 100):
        term = ((-1)**(k-1)) * math.exp(-2*k*k*z*z)
        p_value += term
        if abs(term) < 1e-10: break
    return ks_stat, float(min(max(2*p_value, 0.0), 1.0))

def compute_psi(expected, actual, n_bins=10):
    bins = np.percentile(expected, np.linspace(0, 100, n_bins+1))
    bins[0] -= 1e-9; bins[-1] += 1e-9; bins = np.unique(bins)
    def props(data, bins):
        counts, _ = np.histogram(data, bins=bins)
        p = counts / len(data)
        return np.where(p==0, 1e-6, p)
    ep = props(expected, bins); ap = props(actual, bins)
    return float(np.sum((ap - ep) * np.log(ap / ep)))

def get_system_metrics():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory(); disk = psutil.disk_usage("/")
    return {
        "cpu_percent": cpu, "memory_percent": mem.percent,
        "memory_used_gb": mem.used/1e9, "memory_total_gb": mem.total/1e9,
        "disk_percent": disk.percent, "disk_used_gb": disk.used/1e9, "disk_total_gb": disk.total/1e9,
    }

def lv(ok, warn=False):
    if not ok: return "⚨️ "
    if warn: return "⚠️ "
    return "✅"

def build_report(run_count, drift_factor, drift_results, perf, baseline_metrics, sys_m, prod_df, elapsed):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L += [
        "# MLOps 監視レポート", "",
        "| 項目 | 値 |", "|------|-----|",
        f"| 生成日時 | {now} |",
        f"| 実行回数 | #{run_count} |",
        f"| ドリフト係数 | {drift_factor:.3f} |",
        f"| 本番データ件数 | {len(prod_df):,} 件 |",
        f"| 離反率（本番） | {prod_df['churn'].mean()*100:.1f}% |",
        f"| 処理時間 | {elapsed:.2f} 秒 |", "",
    ]
    L += ["---", "", "## 1. データドリフト検知", "",
          "| 特徴量 | KS統計量 | p値 | KS判定 | PSI | PSI判定 |",
          "|--------|----------|-----|--------|-----|---------|"]
    any_ks = False; any_psi_a = False; any_psi_w = False
    for r in drift_results:
        ks_ok = r["ks_p"] >= KS_P_THRESHOLD
        if not ks_ok: any_ks = True
        if r["psi"] >= PSI_ALERT_THRESHOLD: psi_icon = "⚨️ "; any_psi_a = True
        elif r["psi"] >= PSI_WARN_THRESHOLD: psi_icon = "⚠️ "; any_psi_w = True
        else: psi_icon = "✅"
        L.append(f"| {r['feature']:20s} | {r['ks_stat']:.4f} | {r['ks_p']:.4f} | {lv(ks_ok)} | {r['psi']:.4f} | {psi_icon} |")
    L += ["",
          f"- **KS検定**: {'⚨️ アラート: 分布ドリフトを検知' if any_ks else '✅ 正常'}",
          f"- **PSI**: {'⚨️ アラート: PSI > 0.2' if any_psi_a else '⚠️ 警告: PSI > 0.1' if any_psi_w else '✅ 正常'}", ""]
    ab = baseline_metrics; acc_drop = ab["accuracy"]-perf["accuracy"]; f1_drop = ab["f1"]-perf["f1"]
    acc_ok = acc_drop < PERF_DROP_THRESHOLD; f1_ok = f1_drop < PERF_DROP_THRESHOLD
    L += ["---", "", "## 2. モデルパフォーマンス監視", "",
          "| 指標 | ベースライン | 現在値 | 変化 | 判定 |", "|------|-------------|--------|------|------|",
          f"| Accuracy  | {ab['accuracy']:.4f} | {perf['accuracy']:.4f} | {acc_drop:+.4f} | {lv(acc_ok)} |",
          f"| F1 Score  | {ab['f1']:.4f} | {perf['f1']:.4f} | {f1_drop:+.4f} | {lv(f1_ok)} |", ""]
    L += ["---", "", "## 3. システムリソース監視", "",
          "| リソース | 使用量 | 閘値 | 判定 |", "|---------|--------|------|------|",
          f"| CPU | {sys_m['cpu_percent']:.1f}% | {CPU_ALERT:.0f}% | {lv(sys_m['cpu_percent']<CPU_ALERT)} |",
          f"| メモリ | {sys_m['memory_percent']:.1f}% | {MEMORY_ALERT:.0f}% | {lv(sys_m['memory_percent']<MEMORY_ALERT)} |",
          f"| ディスク | {sys_m['disk_percent']:.1f}% | {DISK_ALERT:.0f}% | {lv(sys_m['disk_percent']<DISK_ALERT)} |", ""]
    any_alert = any_ks or any_psi_a or not acc_ok or not f1_ok
    overall = "⚨️ **要対応**: アラート発生" if any_alert else "⚠️ **要確認**" if any_psi_w else "✅ **すべて正常**"
    L += ["---", "", "## 4. 総合サマリ", "", f"**全体ステータス**: {overall}", "", "---",
          "", f"*このレポートは mlops_monitor.py によって自動生成されました。*"]
    return "\n".join(L)

def main():
    t_start = time.time()
    print("=" * 60)
    print("  MLOps 監視スクリプト 開始")
    print("=" * 60)
    if not os.path.exists(PARAMS_PATH):
        print("エラー: モデルファイルが見つかりません。先に mlops_setup.py を実行してください。")
        sys.exit(1)
    with open(PARAMS_PATH, encoding="utf-8") as f: model_params = json.load(f)
    with open(BASELINE_PATH, encoding="utf-8") as f: baseline_stats = json.load(f)
    model = LogisticRegressionInference(model_params)
    baseline_metrics = model_params["baseline_metrics"]
    run_count = increment_run_count()
    drift_factor = min(run_count * 0.08, 0.8)
    print(f"\n実行回数: #{run_count}  ドリフト係数: {drift_factor:.3f}")
    print("\n[1/4] 本番データ生成中...")
    prod_df = generate_production_data(n=500, seed=run_count*100, drift_factor=drift_factor)
    print(f"  件数: {len(prod_df):,}  離反率: {prod_df['churn'].mean()*100:.1f}%")
    print("\n[2/4] データドリフト検知中...")
    drift_results = []
    for feat in FEATURES:
        bs = np.array(baseline_stats[feat]["samples"])
        ps = prod_df[feat].values
        ks_stat, ks_p = ks_2samp(bs, ps)
        psi = compute_psi(bs, ps)
        drift_results.append({"feature": feat, "ks_stat": ks_stat, "ks_p": ks_p, "psi": psi})
        print(f"  {feat:22s}: KS={ks_stat:.4f} p={ks_p:.4f}  PSI={psi:.4f}")
    print("\n[3/4] モデルパフォーマンス評価中...")
    X_prod = prod_df[FEATURES].values; y_true = prod_df["churn"].values
    y_pred = model.predict(X_prod)
    perf = compute_metrics(y_true, y_pred)
    print(f"  Accuracy: {perf['accuracy']:.4f}  F1: {perf['f1']:.4f}")
    print("\n[4/4] システムリソース取得中...")
    sys_m = get_system_metrics()
    print(f"  CPU: {sys_m['cpu_percent']:.1f}%  メモリ: {sys_m['memory_percent']:.1f}%  ディスク: {sys_m['disk_percent']:.1f}%")
    elapsed = time.time() - t_start
    report_md = build_report(run_count, drift_factor, drift_results, perf, baseline_metrics, sys_m, prod_df, elapsed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [os.path.join(REPORT_DIR, f"report_{ts}.md"), os.path.join(REPORT_DIR, "latest_report.md")]:
        with open(path, "w", encoding="utf-8") as f: f.write(report_md)
    print(f"\nレポート保存完了: reports/latest_report.md")
    print("\n" + report_md)

if __name__ == "__main__":
    main()
