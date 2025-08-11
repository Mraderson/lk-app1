#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重新训练所有模型以确保兼容性
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# 导入高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def load_data():
    """加载数据"""
    data_path = Path('data/量表总分完整数据.csv')
    df = pd.read_csv(data_path)
    
    # 特征和目标变量
    feature_names = ['parent_child_score', 'resilience_score', 'anxiety_score', 'phone_usage_score']
    X = df[feature_names]
    y = df['depression_score']
    
    return X, y

def train_all_models():
    """训练所有模型"""
    print("🔄 Starting to retrain all models...")
    
    # 加载数据
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义所有模型
    models = {}
    
    # 基础模型
    models['LinearRegression'] = LinearRegression()
    models['Ridge'] = Ridge(alpha=1.0)
    models['DecisionTree'] = DecisionTreeRegressor(random_state=42)
    models['KNN'] = KNeighborsRegressor(n_neighbors=5)
    models['SVM'] = SVR(kernel='rbf', C=1.0)
    models['ANN'] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    # 集成模型
    models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
    models['ExtraTrees'] = ExtraTreesRegressor(n_estimators=100, random_state=42)
    models['AdaBoost'] = AdaBoostRegressor(n_estimators=100, random_state=42)
    
    # 高级模型（如果可用）
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    
    # 训练和保存模型
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    trained_models = {}
    
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 保存模型
            model_path = models_dir / f'{name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            trained_models[name] = {
                'model': model,
                'mse': mse,
                'r2': r2
            }
            
            print(f"✅ {name}: MSE={mse:.4f}, R²={r2:.4f}")
            
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    print(f"\n🎉 Training completed! Successfully trained {len(trained_models)} models")
    
    # 显示模型性能排序
    print("\n📊 Model performance ranking (by R² descending):")
    sorted_models = sorted(trained_models.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_models, 1):
        print(f"{i:2d}. {name:15s}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}")
    
    return trained_models

if __name__ == "__main__":
    train_all_models() 