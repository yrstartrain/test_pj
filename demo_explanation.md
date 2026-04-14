# k3s デモ解説

## やったこと

Docker が使える Ubuntu 環境で、軽量 Kubernetes の **k3s** を動かし、Pod / Deployment / Service の基本操作を体験した。

### 最終的な構成

```
kubectl コマンド
     ↓
k3s API サーバー（制御中枢・etcd 内蔵）
     ↓
kwok（偽ノードをシミュレート） ← 本来は実際の kubelet
```

### 実行したコマンド例

```bash
# Pod を作る
kubectl apply -f hello-pod.yaml

# Deployment（Pod を3つ複製）+ Service（外部公開）を作る
kubectl apply -f nginx-deployment.yaml

# 状態確認
kubectl get pods -o wide
kubectl get deployment
kubectl get service
```

### 実行結果イメージ

```
=== Pods ===
NAME                               READY   STATUS    IP
hello-pod                          1/1     Running   10.42.0.3
nginx-deployment-xxx-aaa           1/1     Running   10.42.0.4
nginx-deployment-xxx-bbb           1/1     Running   10.42.0.5
nginx-deployment-xxx-ccc           1/1     Running   10.42.0.6

=== Service ===
nginx-service   ClusterIP   10.43.207.119   80/TCP
```

---

## 主要コンセプト

| リソース | 役割 |
|----------|------|
| **Pod** | コンテナの実行単位 |
| **Deployment** | Pod の数を管理・自動復旧 |
| **Service** | 複数 Pod を1つの IP でまとめて公開 |
| **Node** | Pod が動くサーバー |
| **Namespace** | 環境を論理的に分離する単位 |

---

## はまりポイントと技術制約

### 1. Docker デーモンが起動しない
- **原因**: コンテナ内環境のため `iptables` / `nftables` が使えない
- **対処**: `dockerd --iptables=false --bridge=none` で起動

### 2. ghcr.io のイメージが取得できない
- **原因**: プロキシ環境で ghcr.io のブロブストレージ URL がブロックされていた
- **対処**: k3d（Docker ベース）をあきらめ、k3s バイナリを直接インストール

### 3. overlayfs が使えない
- **原因**: カーネル 4.4.0 という古いバージョンかつコンテナ内のため overlayfs 非対応
- **対処**: `snapshotter: fuse-overlayfs` → さらに `native` に変更

### 4. `/dev/kmsg` がない
- **原因**: コンテナ環境でカーネルメッセージデバイスが存在しない
- **対処**: `touch /dev/kmsg` でダミーファイルを作成

### 5. kubelet が起動しない（cAdvisor エラー）
- **原因**: `/sys/fs/cgroup/cpuacct/cpuacct.usage_percpu` が存在しない
- **対処**: `k3s server --disable-agent` でコントロールプレーンのみ起動し、**kwok** で偽ノードをシミュレート

---

## k3d vs k3s 直接インストール

| | k3d | k3s 直接 |
|--|-----|---------|
| 手軽さ | Docker があればすぐ使える | バイナリ1つで動く |
| 要件 | Docker + ghcr.io アクセス | sudo 権限 |
| 学習用途 | クラスタ複数管理が楽 | シンプルで理解しやすい |
| 本番利用 | ローカル開発向け | 軽量本番環境に適する |

---

## 今回の環境の制約まとめ

この環境（サンドボックスコンテナ）では以下が制限されており、通常の k3s デモとは異なる手順が必要だった。

- overlayfs 非対応 → コンテナのファイルシステム分離ができない
- cgroup cpuacct 不完全 → kubelet の cAdvisor が動かない
- DNS (UDP 53) ブロック → コンテナ内からの名前解決ができない
- iptables/nf_conntrack 非対応 → Pod 間通信ルールが設定できない

**通常の Linux 環境（VPS・EC2・物理マシン）では上記の問題は発生しない。**

---

## 次のステップ

```bash
# スケールアウト
kubectl scale deployment nginx-deployment --replicas=5

# Pod を削除して自動復旧を確認
kubectl delete pod <pod-name>
kubectl get pods -w   # -w でリアルタイム監視

# Namespace で環境分離
kubectl create namespace staging
kubectl apply -f app.yaml -n staging
```
