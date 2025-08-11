#!/usr/bin/env python3
"""
快速创建更多机器学习模型
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    BaggingRegressor, 
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import (
    Lasso, 
    ElasticNet, 
    BayesianRidge,
    HuberRegressor,
    SGDRegressor
)
from sklearn.tree import ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载训练数据"""
    current_dir = Path(__file__).parent
    data_path = current_dir / 'data' / '量表总分完整数据.csv'
    
    if not data_path.exists():
        print("❌ 找不到数据文件")
        return None, None, None, None
    
    df = pd.read_csv(data_path)
    
    # 特征和目标变量
    features = ['parent_child_score', 'resilience_score', 'anxiety_score', 'phone_usage_score']
    target = 'depression_score'
    
    X = df[features]
    y = df[target]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def create_models():
    """创建各种新模型"""
    
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return
    
    # 标准化特征（某些模型需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义要创建的模型
    models_to_create = {
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'BayesianRidge': BayesianRidge(),
        'HuberRegressor': HuberRegressor(epsilon=1.35),
        'SGDRegressor': SGDRegressor(random_state=42, max_iter=1000),
        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'GaussianProcess': GaussianProcessRegressor(
            kernel=C(1.0) * RBF(1.0), 
            random_state=42,
            alpha=1e-6
        ),
        'ExtraTreeRegressor': ExtraTreeRegressor(random_state=42),
        'BaggingRegressor': BaggingRegressor(
            estimator=ExtraTreeRegressor(random_state=42),
            n_estimators=10,
            random_state=42
        ),
        'VotingRegressor': VotingRegressor([
            ('lr', Lasso(alpha=0.1, random_state=42)),
            ('ridge', BayesianRidge()),
            ('tree', ExtraTreeRegressor(random_state=42))
        ]),
        'StackingRegressor': StackingRegressor([
            ('lasso', Lasso(alpha=0.1, random_state=42)),
            ('bayesian', BayesianRidge()),
            ('tree', ExtraTreeRegressor(random_state=42))
        ], final_estimator=BayesianRidge())
    }
    
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    successful_models = []
    
    for model_name, model in models_to_create.items():
        try:
            print(f"🔄 Training model: {model_name}")
            
            # 根据模型选择是否使用标准化数据
            if model_name in ['MLPRegressor', 'SGDRegressor', 'GaussianProcess']:
                model.fit(X_train_scaled, y_train)
                # 测试预测
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                # 测试预测
                y_pred = model.predict(X_test)
            
            # 计算R²分数
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            
            # 保存模型
            model_path = models_dir / f'{model_name}_model.pkl'
            with open(model_path, 'wb') as f:
                # 对于需要scaler的模型，保存模型和scaler的组合
                if model_name in ['MLPRegressor', 'SGDRegressor', 'GaussianProcess']:
                    # 创建包装器类
                    class ScaledModel:
                        def __init__(self, model, scaler):
                            self.model = model
                            self.scaler = scaler
                        
                        def predict(self, X):
                            X_scaled = self.scaler.transform(X)
                            return self.model.predict(X_scaled)
                    
                    scaled_model = ScaledModel(model, scaler)
                    pickle.dump(scaled_model, f)
                else:
                    pickle.dump(model, f)
            
            print(f"✅ {model_name} 训练完成，R² = {r2:.4f}")
            successful_models.append(model_name)
            
        except Exception as e:
            print(f"❌ {model_name} 训练失败: {e}")
            continue
    
    print(f"\n🎉 成功创建了 {len(successful_models)} 个新模型:")
    for model_name in successful_models:
        print(f"  - {model_name}")
    
    return successful_models

if __name__ == "__main__":
    create_models() 