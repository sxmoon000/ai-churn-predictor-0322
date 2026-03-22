"""
用户流失预测 — 电信客户 Churn Prediction
使用逻辑回归 + 随机森林预测哪些客户可能流失。
技术栈：特征工程 + 逻辑回归 + 随机森林 + ROC 曲线

知识点：
  1. 二分类问题：Churn/Not Churn
  2. 特征工程：分类变量 OneHot 编码
  3. 类别不平衡：class_weight 处理
  4. ROC-AUC：评估二分类器的排序能力
  5. 特征重要性：哪些因素最影响流失
"""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
)
import joblib

# ── 1. 生成模拟数据 ──
print("[1/3] Generating synthetic telecom data...")
np.random.seed(42)
n = 5000

X = np.column_stack([
    np.random.exponential(12, n),            # 月费 ($)
    np.random.randint(1, 60, n),             # 合同剩余(月)
    np.random.poisson(3, n),                 # 客服电话次数
    np.random.exponential(50, n),            # 月流量 (GB)
    np.random.poisson(1, n),                 # 投诉次数
    np.random.randint(0, 2, n),              # 国际漫游
    np.random.randint(0, 4, n),              # 附加服务数
    np.random.normal(300, 100, n),           # 通话时长 (分钟)
])
y = ((X[:, 0] > 60).astype(int) +          # 高价用户易流失
     (X[:, 2] > 5).astype(int) +            # 客服电话多→不满
     (X[:, 4] > 2).astype(int) +            # 投诉多
     (X[:, 1] < 10).astype(int) * 2)        # 合同快到期→高风险
y = (y >= 2).astype(int)

print(f"   样本数: {n}, 流失率: {y.mean():.1%}, 特征数: {X.shape[1]}")

# ── 2. 训练 ──
print("[2/3] Training models...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 逻辑回归
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(X_train_s, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])

# 随机森林
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train_s, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1])

print(f"   Logistic Regression AUC: {lr_auc:.3f}")
print(f"   Random Forest AUC:       {rf_auc:.3f}")

# ── 3. 特征重要性 ──
feature_names = ["月费", "合同剩余(月)", "客服电话数", "月流量(GB)", "投诉数", "国际漫游", "附加服务", "通话时长"]

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\n📊 特征重要性 (Random Forest):")
for i in indices:
    bar = "█" * int(importances[i] * 50)
    print(f"   {feature_names[i]:<15} {bar} {importances[i]:.3f}")

# 保存
Path("model").mkdir(exist_ok=True)
joblib.dump(lr, "model/lr_churn.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print(f"\n✅ 完成! LR-AUC={lr_auc:.3f}, RF-AUC={rf_auc:.3f}")
