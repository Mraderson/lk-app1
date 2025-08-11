#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import time

# 设置随机种子，保证可重复性
SEED = 42
np.random.seed(SEED)

# 使用系统默认字体而不是指定特定字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 'Avant Garde', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = True
# 禁用字体警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 确保结果目录存在
def ensure_dir(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# 确保结果目录
ensure_dir('results')
ensure_dir('models')
ensure_dir('figures')

# 创建所有模型的结果目录
model_names = [
    'RandomForest', 'LightGBM', 'GradientBoosting', 'XGBoost', 
    'AdaBoost', 'DecisionTree', 'ExtraTrees', 'SVM', 
    'LinearRegression', 'Ridge', 'ANN', 'KNN'
]
for model_name in model_names:
    ensure_dir(f'results/{model_name}')

def load_data(file_path):
    """加载数据"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df, depression_threshold=10):
    """数据预处理"""
    print("Preprocessing data...")
    
    # 检查并处理缺失值
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values")
        df = df.dropna()
        print(f"After removing missing values: {df.shape[0]} rows")
    
    # 将抑郁得分二分类化（用于计算AUC等指标）
    df['Depression_Binary'] = (df['depression_score'] >= depression_threshold).astype(int)
    
    # 特征和目标变量
    X = df[['parent_child_score', 'resilience_score', 'anxiety_score', 'phone_usage_score']]
    y_reg = df['depression_score']  # 回归目标
    y_cls = df['Depression_Binary']  # 分类目标（用于计算AUC等指标）
    
    return X, y_reg, y_cls

def split_scale_data(X, y_reg, y_cls, test_size=0.2, random_state=None):
    """拆分数据并进行标准化"""
    # 拆分数据
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=test_size, random_state=random_state
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存特征名称（用于特征重要性分析）
    feature_names = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_reg_train, y_reg_test, y_cls_train, y_cls_test, feature_names, scaler

def calculate_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """计算分类评估指标"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算各项指标
    sensitivity = recall_score(y_true, y_pred)  # 敏感度 = 召回率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度
    ppv = precision_score(y_true, y_pred) if (tp + fp) > 0 else 0  # 阳性预测值 = 精确率
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
    accuracy = accuracy_score(y_true, y_pred)  # 准确率
    f1 = f1_score(y_true, y_pred)  # F1分数
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba)  # AUC
    except:
        auc = 0.5  # 如果AUC计算失败，设置为0.5
    
    return {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Accuracy': accuracy,
        'F1': f1
    }

def find_optimal_threshold(y_true, y_pred_proba):
    """找到最优的分类阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def evaluate_regression_metrics(y_true, y_pred):
    """计算回归评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def normalize_predictions(y_pred):
    """将回归预测值标准化为0-1之间，用于分类评估"""
    if np.max(y_pred) == np.min(y_pred):
        return np.zeros_like(y_pred)
    return (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred) + 1e-10)

def correlation_analysis(df, save_path='figures/correlation_heatmap.png'):
    """相关性分析"""
    # 计算特征与目标变量的相关系数
    corr = df[['parent_child_score', 'resilience_score', 'anxiety_score', 'phone_usage_score', 'depression_score']].corr()
    
    # 英文列名映射
    column_rename = {
        'parent_child_score': 'Parent-Child Relationship',
        'resilience_score': 'Resilience',
        'anxiety_score': 'Anxiety',
        'phone_usage_score': 'Phone Usage Time',
        'depression_score': 'Depression'
    }
    
    # 重命名相关矩阵的行列名
    corr_renamed = corr.rename(columns=column_rename, index=column_rename)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_renamed, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Features and Depression Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    # 打印与抑郁得分的相关系数
    print("\nCorrelation with Depression Score:")
    depression_corr = corr['depression_score'].sort_values(ascending=False)
    depression_corr.index = [column_rename.get(col, col) for col in depression_corr.index]
    print(depression_corr)
    
    return corr

def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """绘制特征重要性图"""
    # 检查模型是否支持特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 英文特征名映射
        english_feature_names = [
            'Parent-Child Relationship',
            'Resilience',
            'Anxiety',
            'Phone Usage Time'
        ]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'{model_name} Feature Importance')
        plt.bar(range(len(feature_names)), importances[indices], align='center')
        plt.xticks(range(len(feature_names)), [english_feature_names[i] for i in indices], rotation=45)
        plt.xlim([-1, len(feature_names)])
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 返回按重要性排序的特征
        feature_importance_df = pd.DataFrame({
            'Feature': [english_feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        return feature_importance_df
    
    return None

def plot_roc_curve(y_true, y_pred_proba, model_name, save_path=None):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
    return auc

def plot_all_roc_curves(all_results, save_path='figures/all_models_roc_curves.png'):
    """绘制所有模型的ROC曲线"""
    plt.figure(figsize=(12, 10))
    
    # 使用不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for i, (model_name, result) in enumerate(all_results.items()):
        model_dir = f'results/{model_name}'
        predictions_file = f'{model_dir}/predictions.csv'
        
        if os.path.exists(predictions_file):
            # 读取预测结果
            preds = pd.read_csv(predictions_file)
            if '实际抑郁状态' in preds.columns and '预测抑郁概率' in preds.columns:
                y_true = preds['实际抑郁状态']
                y_pred = preds['预测抑郁概率']
                
                # 计算ROC曲线
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred)
                
                # 绘制ROC曲线
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', color=colors[i])
    
    # 添加对角线（随机猜测）
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_model_results(model, model_name, reg_metrics, cls_metrics, y_reg_test, y_reg_pred, y_cls_test, y_proba, 
                      feature_names=None, feature_importance=None, optimal_threshold=0.5, X_train=None, X_test=None, 
                      run_shap_analysis=True, use_gpu=True):
    """保存模型结果并运行SHAP分析"""
    # 创建结果目录
    model_dir = f'results/{model_name}'
    ensure_dir(model_dir)
    
    # 保存模型
    joblib.dump(model, f'models/{model_name}_model.pkl')
    
    # 保存指标
    metrics_df = pd.DataFrame({
        'Metric': list(reg_metrics.keys()) + list(cls_metrics.keys()),
        'Value': list(reg_metrics.values()) + list(cls_metrics.values())
    })
    metrics_df.to_csv(f'{model_dir}/metrics.csv', index=False)
    
    # 保存预测值
    pred_df = pd.DataFrame({
        '实际抑郁得分': y_reg_test,
        '预测抑郁得分': y_reg_pred,
        '实际抑郁状态': y_cls_test,
        '预测抑郁概率': y_proba
    })
    pred_df.to_csv(f'{model_dir}/predictions.csv', index=False)
    
    # 保存特征重要性
    if feature_importance is not None and feature_names is not None:
        feature_importance.to_csv(f'{model_dir}/feature_importance.csv', index=False)
    
    # 绘制回归散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_reg_test, y_reg_pred, alpha=0.5)
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'k--')
    plt.xlabel('Actual Depression Score')
    plt.ylabel('Predicted Depression Score')
    plt.title(f'Regression Prediction Scatter Plot - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_dir}/regression_scatter.png', dpi=300)
    plt.close()
    
    # 绘制ROC曲线
    plot_roc_curve(y_cls_test, y_proba, model_name, save_path=f'{model_dir}/roc_curve.png')
    
    # 绘制混淆矩阵
    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_cls_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Depressed', 'Depressed'], 
                yticklabels=['Non-Depressed', 'Depressed'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'{model_dir}/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 运行SHAP分析
    if run_shap_analysis and X_train is not None and X_test is not None:
        try:
            print(f"\n开始为{model_name}模型运行SHAP可解释性分析...")
            from shap_analysis import run_shap_analysis_for_model
            
            # 运行SHAP分析
            shap_analyzer = run_shap_analysis_for_model(
                model=model,
                X_train=X_train,
                X_test=X_test,
                model_name=model_name,
                feature_names=feature_names,
                use_gpu=use_gpu,
                research_mode='fine'  # 使用精细研究模式获得高质量的可解释性分析
            )
            
            print(f"{model_name}模型的SHAP分析完成！")
            
        except Exception as e:
            print(f"运行{model_name}模型的SHAP分析时发生错误: {str(e)}")
            print("跳过SHAP分析，继续其他步骤...")
    
    # 返回结果摘要
    result_summary = {
        'model_name': model_name,
        'reg_metrics': reg_metrics,
        'cls_metrics': cls_metrics,
        'optimal_threshold': optimal_threshold
    }
    
    return result_summary