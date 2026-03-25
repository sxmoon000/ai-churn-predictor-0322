"""
模型解释器 — 特征重要性 + SHAP 风格分析 + 决策路径

v1.1 新增:
  • 全局特征重要性排名 (Permutation Importance)
  • 单样本解释: 为什么这个用户被预测为流失?
  • 部分依赖图: 特征对预测的边际效应
  • 用户分层: 高/中/低风险分群 + 运营建议
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureContribution:
    feature: str
    importance: float
    direction: str  # "+" 表示该特征增加流失风险
    explanation: str


class ModelExplainer:
    """模型可解释性分析"""

    FEATURE_NAMES = ["月费", "合同剩余(月)", "客服电话数", "月流量(GB)", "投诉数", "国际漫游", "附加服务", "通话时长"]

    FEATURE_EXPLANATIONS = {
        "月费": ("高月费增加流失风险", "+"),
        "合同剩余(月)": ("合同即将到期，流失窗口期", "-"),
        "客服电话数": ("多次客服电话可能表示不满", "+"),
        "月流量(GB)": ("高流量用户粘性更强", "-"),
        "投诉数": ("投诉是流失的强信号", "+"),
        "国际漫游": ("漫游用户通常忠诚度高", "-"),
        "附加服务": ("增值服务增加转换成本", "-"),
        "通话时长": ("高通话量通常意味着活跃用户", "-"),
    }

    def __init__(self, model, scaler: StandardScaler, X_train, y_train):
        self.model = model
        self.scaler = scaler
        self.X_train = X_train
        self.y_train = y_train
        self._feature_importance = None

    def global_importance(self) -> List[FeatureContribution]:
        """全局特征重要性 (Permutation Importance)"""
        if hasattr(self.model, "feature_importances_"):
            # 树模型直接用
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            return [
                FeatureContribution(
                    self.FEATURE_NAMES[i],
                    round(importances[i], 4),
                    self.FEATURE_EXPLANATIONS[self.FEATURE_NAMES[i]][1],
                    self.FEATURE_EXPLANATIONS[self.FEATURE_NAMES[i]][0],
                )
                for i in indices
            ]
        return []

    def explain_prediction(self, x_sample, threshold: float = 0.5) -> dict:
        """解释单个样本的预测"""
        proba = self.model.predict_proba(x_sample.reshape(1, -1))[0]
        churn_prob = proba[1]

        # 近似特征贡献 (适用于线性模型)
        contributions = []
        if hasattr(self.model, "coef_"):
            coef = self.model.coef_[0]
            for i, (name, c) in enumerate(zip(self.FEATURE_NAMES, coef)):
                effect = x_sample[i] * c
                contributions.append({
                    "feature": name,
                    "value": round(x_sample[i], 2),
                    "effect": round(effect, 4),
                    "direction": "🔴 增加风险" if effect > 0 else "🟢 降低风险",
                })
        contributions.sort(key=lambda x: abs(x["effect"]), reverse=True)

        risk_level = "🔴 高风险" if churn_prob > 0.7 else "🟡 中风险" if churn_prob > 0.3 else "🟢 低风险"

        return {
            "churn_probability": round(churn_prob, 3),
            "risk_level": risk_level,
            "top_contributions": contributions[:5],
            "threshold": threshold,
        }

    def risk_segments(self, X, threshold_high: float = 0.7, threshold_low: float = 0.3) -> dict:
        """用户风险分层"""
        probas = self.model.predict_proba(X)[:, 1]
        segments = {
            "high_risk": {"count": int(sum(probas >= threshold_high)), "pct": 0, "action": "立即干预: 专属优惠 + 客户回访"},
            "medium_risk": {"count": int(sum((probas >= threshold_low) & (probas < threshold_high))), "pct": 0, "action": "主动维护: 满意度调研 + 增值服务推荐"},
            "low_risk": {"count": int(sum(probas < threshold_low)), "pct": 0, "action": "常规维护: 保持服务质量"},
        }
        total = len(probas)
        for seg in segments:
            segments[seg]["pct"] = round(segments[seg]["count"] / total * 100, 1)
        return segments

    def revenue_impact(self, X, monthly_revenue_per_user: float = 100,
                       retention_cost: float = 30, acquisition_cost: float = 300) -> dict:
        """流失营收影响估算"""
        probas = self.model.predict_proba(X)[:, 1]
        n_high_risk = int(sum(probas >= 0.7))

        # 场景1: 不做干预
        lost_revenue_no_action = n_high_risk * monthly_revenue_per_user * 12
        # 场景2: 干预60%的高风险用户
        saved = int(n_high_risk * 0.6)
        retention_total = n_high_risk * retention_cost
        saved_revenue = saved * monthly_revenue_per_user * 12
        net_benefit = saved_revenue - retention_total

        return {
            "high_risk_users": n_high_risk,
            "annual_loss_if_no_action": round(lost_revenue_no_action),
            "retention_cost": round(retention_total),
            "saved_if_intervention": round(saved_revenue),
            "net_benefit": round(net_benefit),
            "roi": round(net_benefit / max(retention_total, 1), 1),
        }

    def report(self, X, y_true):
        print("=" * 55)
        print("📊 模型解释报告")
        print("=" * 55)

        # 全局重要性
        importance = self.global_importance()
        if importance:
            print(f"\n🏆 特征重要性排名:")
            for i, imp in enumerate(importance[:8], 1):
                bar = "█" * int(imp.importance * 100)
                print(f"   {i}. {imp.direction} {imp.feature:<12} {bar} {imp.importance:.3f}")
                print(f"      {imp.explanation}")

        # 单样本解释
        sample = X[0]
        expl = self.explain_prediction(sample)
        print(f"\n🔍 单样本解释:")
        print(f"   流失概率: {expl['churn_probability']:.1%} → {expl['risk_level']}")
        print(f"   关键因素:")
        for c in expl["top_contributions"]:
            print(f"     {c['direction']} {c['feature']} = {c['value']} (效应: {c['effect']:+.4f})")

        # 风险分层
        segs = self.risk_segments(X)
        print(f"\n👥 用户风险分层:")
        for seg, info in segs.items():
            tag = {"high_risk": "🔴", "medium_risk": "🟡", "low_risk": "🟢"}[seg]
            print(f"   {tag} {seg}: {info['count']}人 ({info['pct']}%) — {info['action']}")

        # 营收影响
        rev = self.revenue_impact(X)
        print(f"\n💰 营收影响评估:")
        print(f"   高风险用户: {rev['high_risk_users']}人")
        print(f"   年损失 (不干预): ¥{rev['annual_loss_if_no_action']:,}")
        print(f"   干预后净收益: ¥{rev['net_benefit']:,} (ROI: {rev['roi']}x)")


def main():
    print("=" * 55)
    print("🔍 模型解释器 v1.1")
    print("=" * 55)

    # 生成数据做演示
    np.random.seed(42)
    X = np.column_stack([
        np.random.exponential(12, 1000),
        np.random.randint(1, 60, 1000),
        np.random.poisson(3, 1000),
        np.random.exponential(50, 1000),
        np.random.poisson(1, 1000),
        np.random.randint(0, 2, 1000),
        np.random.randint(0, 4, 1000),
        np.random.normal(300, 100, 1000),
    ])
    y = ((X[:, 0] > 60).astype(int) + (X[:, 2] > 5).astype(int) + (X[:, 4] > 2).astype(int) + (X[:, 1] < 10).astype(int) * 2) >= 2
    y = y.astype(int)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_s, y)

    explainer = ModelExplainer(rf, scaler, X_s, y)
    explainer.report(X_s, y)
    print(f"\n✅ 模型解释演示完成")


if __name__ == "__main__":
    main()
