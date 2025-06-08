#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试模型加载状态
"""

import pickle
import warnings
from pathlib import Path
import pandas as pd

def test_model_loading():
    """测试所有模型的加载状态"""
    
    models_dir = Path('models')
    
    model_files = {
        'XGBoost': 'XGBoost_model.pkl',
        'LightGBM': 'LightGBM_model.pkl',
        'RandomForest': 'RandomForest_model.pkl',
        'GradientBoosting': 'GradientBoosting_model.pkl',
        'ExtraTrees': 'ExtraTrees_model.pkl',
        'SVM': 'SVM_model.pkl',
        'ANN': 'ANN_model.pkl',
        'AdaBoost': 'AdaBoost_model.pkl',
        'KNN': 'KNN_model.pkl',
        'DecisionTree': 'DecisionTree_model.pkl',
        'LinearRegression': 'LinearRegression_model.pkl',
        'Ridge': 'Ridge_model.pkl'
    }
    
    loaded_models = {}
    failed_models = []
    
    # 测试数据
    test_data = pd.DataFrame({
        '亲子量表总得分': [17],
        '韧性量表总得分': [7],
        '焦虑量表总得分': [4],
        '手机使用时间总得分': [23]
    })
    
    print("🧪 开始测试模型加载和预测...")
    print("=" * 50)
    
    for model_name, file_name in model_files.items():
        model_path = models_dir / file_name
        
        if not model_path.exists():
            print(f"❌ {model_name}: 文件不存在 ({file_name})")
            failed_models.append(model_name)
            continue
            
        try:
            # 加载模型
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # 测试预测
            prediction = model.predict(test_data)[0]
            
            loaded_models[model_name] = model
            print(f"✅ {model_name}: 加载成功，预测结果 = {prediction:.2f}")
            
        except Exception as e:
            print(f"❌ {model_name}: 加载失败 - {str(e)[:100]}...")
            failed_models.append(model_name)
    
    print("=" * 50)
    print(f"📊 总结:")
    print(f"   成功加载: {len(loaded_models)} 个模型")
    print(f"   加载失败: {len(failed_models)} 个模型")
    
    if loaded_models:
        print(f"   可用模型: {', '.join(loaded_models.keys())}")
    
    if failed_models:
        print(f"   失败模型: {', '.join(failed_models)}")
    
    return loaded_models, failed_models

if __name__ == "__main__":
    test_model_loading() 