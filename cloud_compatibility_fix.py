#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
云端环境兼容性修复脚本
修复numpy和SHAP库的兼容性问题
"""

import sys
import warnings

def apply_numpy_compatibility_patch():
    """应用numpy兼容性补丁"""
    try:
        import numpy as np
        
        # 检查并添加缺失的别名
        if not hasattr(np, 'int'):
            np.int = int
            print("✅ 添加 np.int 别名")
        
        if not hasattr(np, 'float'):
            np.float = float
            print("✅ 添加 np.float 别名")
            
        if not hasattr(np, 'bool'):
            np.bool = bool
            print("✅ 添加 np.bool 别名")
            
        if not hasattr(np, 'complex'):
            np.complex = complex
            print("✅ 添加 np.complex 别名")
            
        print("✅ NumPy兼容性补丁应用成功")
        return True
        
    except Exception as e:
        print(f"❌ NumPy兼容性补丁失败: {e}")
        return False

def test_shap_compatibility():
    """测试SHAP库兼容性"""
    try:
        import shap
        print("✅ SHAP库导入成功")
        
        # 测试简单的SHAP功能
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # 创建简单测试数据
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [2, 4, 6, 8]
        })
        y = [3, 6, 9, 12]
        
        # 训练简单模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 测试SHAP LinearExplainer
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X.iloc[[0]])
        
        print("✅ SHAP LinearExplainer 测试成功")
        return True
        
    except Exception as e:
        print(f"❌ SHAP兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始云端兼容性修复...")
    
    # 应用numpy补丁
    numpy_ok = apply_numpy_compatibility_patch()
    
    # 测试SHAP兼容性
    shap_ok = test_shap_compatibility()
    
    if numpy_ok and shap_ok:
        print("🎉 云端兼容性修复完成！")
        return True
    else:
        print("⚠️ 部分兼容性问题仍存在")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 