# 📉 用户流失预测

> 逻辑回归 + 随机森林，AUC 评估二分类

## 🧠 知识点
- **Churn Prediction**: 预测用户是否取消服务，是机器学习在商业中的经典应用
- **ROC-AUC**: 不只看准确率——正负样本不平衡时 AUC 更可靠
- **特征重要性**: 随机森林告诉你哪个特征对预测贡献最大
- **class_weight=balanced**: 样本不均衡的自动加权方案
- **OneHot vs Ordinal**: 分类变量编码的选择

## 🚀 运行
```bash
pip install -r requirements.txt && python src/train.py
```

---

Day 6 | 2026-03-22 | [sxmoon000](https://github.com/sxmoon000)
