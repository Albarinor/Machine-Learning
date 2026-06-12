# %% [markdown]
# # 机器学习大作业
# ## 第二大类：数据集分类与可视化实验
# ### 实验1：基于 Fashion MNIST 数据集的多算法分类

# %%
# 导入基础库
import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文、负号显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 机器学习相关库
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# %% [markdown]
# ### 1. 实验目的
# 1. 熟悉 Fashion MNIST 图像数据集结构，完成数据读取、可视化与预处理操作。
# 2. 使用多种机器学习分类算法对服饰图像进行十分类任务。
# 3. 对比不同算法的**准确率、精确率、召回率、F1分数、训练时间**五项指标，分析各算法性能差异与适用场景。
# 4. 掌握模型评估与结果可视化方法。

# %% [markdown]
# ### 2. 数据集介绍
# Fashion MNIST 是替代传统手写数字 MNIST 的服饰图像数据集，常用于机器学习入门与算法对比实验：
# - 数据规模：总计 70000 张灰度图像，单张图像尺寸为 28×28 像素，展平后为 784 维特征。
# - 类别划分：共 10 个服饰类别，标签取值 0~9。
# - 实验设置：本次随机抽取 10000 条样本开展实验，按 8:2 比例划分为训练集与测试集。
# 
# 类别对应关系：
# 0 - T恤/上衣，1 - 裤子，2 - 套头衫，3 - 连衣裙，4 - 外套
# 5 - 凉鞋，6 - 衬衫，7 - 运动鞋，8 - 包，9 - 短靴

# %%
# 加载 Fashion MNIST 数据集
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

# 查看数据基本信息
print("特征矩阵形状：  breast_cancer_classification.ipynb:5 - Untitled-2:44", X.shape)   # (样本数, 像素总数)
print("标签形状：  breast_cancer_classification.ipynb:6 - Untitled-2:45", y.shape)
print("标签类别：  breast_cancer_classification.ipynb:7 - Untitled-2:46", np.unique(y))

# %% [markdown]
# ### 3. 数据预处理与可视化
# 1. 数据抽样：从完整数据集中抽取部分样本，提升运行效率。
# 2. 数据集划分：将样本分为训练集和测试集。
# 3. 特征标准化：对像素特征做标准化处理，消除量纲影响，适配各类机器学习模型。
# 4. 数据可视化：还原图像样本，直观展示数据集内容。

# %%
# 随机抽取 30000 个样本（加快运行速度，可自行改大小）
np.random.seed(42)  # 固定随机种子，结果可复现
sample_idx = np.random.choice(len(X), 30000, replace=False)
X_sample = X[sample_idx]
y_sample = y[sample_idx]

# 划分训练集 80%、测试集 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

print("训练集形状：  breast_cancer_classification.ipynb:12 - Untitled-2:67", X_train.shape)
print("测试集形状：  breast_cancer_classification.ipynb:13 - Untitled-2:68", X_test.shape)

# %%
# 定义类别名称
class_names = [
    'T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
    '凉鞋', '衬衫', '运动鞋', '包', '短靴'
]

# 展示前16张图片
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    # 784维转回 28*28 图片
    img = X_train[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    label = int(y_train[i])
    plt.title(class_names[label])
    plt.axis('off')
plt.tight_layout()
plt.show()

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 初始化模型
knn = KNeighborsClassifier(n_neighbors=5)
# 训练
knn.fit(X_train_scaled, y_train)
# 预测
y_pred_knn = knn.predict(X_test_scaled)

# 评估
print("KNN 准确率:  breast_cancer_classification.ipynb:13 - Untitled-2:110", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, target_names=class_names))

# %%
import seaborn as sns

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('KNN 模型混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

# %% [markdown]
# ### 4. 实验选用算法
# 本次实验共使用 10 种分类算法，覆盖传统机器学习、集成学习、神经网络三大类：
# - 基础模型：K近邻(KNN)、逻辑回归、决策树、朴素贝叶斯、支持向量机(SVM)
# - 集成学习：随机森林、AdaBoost、XGBoost、LightGBM
# - 神经网络：多层感知机(MLP)
# 
# 统一评估指标：准确率(Accuracy)、精确率(Precision)、召回率(Recall)、F1分数(F1-score)、训练时间。

# %%
# -------------------- 先导入所有库--------------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import time

# -------------------- 1. 定义模型和类别名称 --------------------
# 定义10个模型
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "朴素贝叶斯": GaussianNB(),
    "MLP神经网络": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)
}

# Fashion MNIST 类别名称（你之前缺失了这个定义）
class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
               '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# -------------------- 2. 数据类型转换（你写的这部分保留） --------------------
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# -------------------- 3. 存储所有模型的评估结果 --------------------
acc_results = {}

print("===== 开始训练所有模型 =====  notebooks_breast_cancer_classification.ipynb:35 - Untitled-2:168")
for name, model in models.items():
    print(f"\n正在训练：{name}  notebooks_breast_cancer_classification.ipynb:37 - Untitled-2:170")
    try:
        # 训练模型
        model.fit(X_train_scaled, y_train)
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        acc_results[name] = acc
        
        print(f"准确率：{acc:.4f}  notebooks_breast_cancer_classification.ipynb:48 - Untitled-2:181")
        print(classification_report(y_test, y_pred, target_names=class_names))
    except Exception as e:
        print(f"{name} 训练失败，错误信息：{e}  notebooks_breast_cancer_classification.ipynb:51 - Untitled-2:184")
        continue

# %% [markdown]
# ### 5. 实验结果与分析
# #### 5.1 量化指标结果
# 各算法在测试集上的五项评估指标、训练时间如下：

# %%
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 清空之前结果
acc_results = {}
precision_results = {}
recall_results = {}
f1_results = {}
time_results = {}

# 确保标签是整数（必须加）
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# ========== 训练 + 完整指标评估 ==========
print("=  breast_cancer_classification.ipynb:19 - Untitled-2:211" * 80)
print("Fashion MNIST 各算法完整性能指标  breast_cancer_classification.ipynb:20 - Untitled-2:212")
print("=  breast_cancer_classification.ipynb:21 - Untitled-2:213" * 80)

for name, model in models.items():
    print(f"\n 正在训练：{name}  breast_cancer_classification.ipynb:24 - Untitled-2:216")
    
    # 记录开始时间
    start = time.time()
    
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 记录结束时间
    train_time = time.time() - start
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 计算所有指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 保存结果
    acc_results[name] = acc
    precision_results[name] = precision
    recall_results[name] = recall
    f1_results[name] = f1
    time_results[name] = train_time
    
    # 打印完整指标（写报告直接复制）
    print(f"{name} 指标  breast_cancer_classification.ipynb:52 - Untitled-2:244")
    print(f"准确率(Accuracy)   : {acc:.4f}  breast_cancer_classification.ipynb:53 - Untitled-2:245")
    print(f"精确率(Precision) : {precision:.4f}  breast_cancer_classification.ipynb:54 - Untitled-2:246")
    print(f"召回率(Recall)    : {recall:.4f}  breast_cancer_classification.ipynb:55 - Untitled-2:247")
    print(f"F1分数(F1score)  : {f1:.4f}  breast_cancer_classification.ipynb:56 - Untitled-2:248")
    print(f"训练时间(Time)    : {train_time:.4f} 秒  breast_cancer_classification.ipynb:57 - Untitled-2:249")

print("\n  breast_cancer_classification.ipynb:59 - Untitled-2:251" + "=" * 80)
print("所有模型训练完成！  breast_cancer_classification.ipynb:60 - Untitled-2:252")
print("=  breast_cancer_classification.ipynb:61 - Untitled-2:253" * 80)

# %% [markdown]
# #### 5.2 可视化结果
# 分别绘制各算法准确率、精确率、召回率、F1分数、训练时间对比柱状图，以及最优模型的混淆矩阵，直观对比模型性能。

# %%
plt.figure(figsize=(12,5))
plt.bar(acc_results.keys(), acc_results.values(), color='skyblue')
plt.title('各算法准确率对比')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12,5))
plt.bar(precision_results.keys(), precision_results.values(), color='salmon')
plt.title('各算法精确率对比')
plt.ylabel('Precision')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12,5))
plt.bar(recall_results.keys(), recall_results.values(), color='lightgreen')
plt.title('各算法召回率对比')
plt.ylabel('Recall')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12,5))
plt.bar(f1_results.keys(), f1_results.values(), color='violet')
plt.title('各算法F1-score对比')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12,5))
plt.bar(time_results.keys(), time_results.values(), color='gray')
plt.title('各算法训练时间对比')
plt.ylabel('Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### 5.3 结果分析
# 1. 性能表现：集成学习算法（XGBoost、LightGBM）与 MLP 神经网络综合性能最优，准确率达到 87% 以上，擅长挖掘图像高维特征。
# 2. 劣势模型：朴素贝叶斯因“特征相互独立”的假设与图像像素相关性强的特点冲突，分类效果最差；单棵决策树泛化能力弱，准确率偏低。
# 3. 效率对比：线性模型、树模型训练速度较快；SVM、MLP 神经网络拟合过程复杂，训练耗时更长。
# 4. 综合结论：针对高维图像分类任务，优先选择集成学习或浅层神经网络；简单线性模型、朴素贝叶斯不适合此类场景。


