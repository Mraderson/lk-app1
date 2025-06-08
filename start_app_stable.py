#!/usr/bin/env python3
"""
稳定的抑郁量表预测应用启动脚本
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    required_imports = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'shap': 'shap'
    }
    
    missing = []
    for import_name, package_name in required_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    # 检查pickle（内置模块）
    try:
        import pickle
    except ImportError:
        missing.append('pickle')
    
    if missing:
        print(f"❌ 缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def test_models():
    """测试模型是否可用"""
    try:
        import pickle
        import warnings
        import pandas as pd
        
        models_dir = Path('models')
        working_models = []
        
        model_files = {
            'XGBoost': 'XGBoost_model.pkl',
            'LightGBM': 'LightGBM_model.pkl', 
            'KNN': 'KNN_model.pkl',
            'LinearRegression': 'LinearRegression_model.pkl',
            'Ridge': 'Ridge_model.pkl'
        }
        
        test_data = pd.DataFrame({
            '亲子量表总得分': [17],
            '韧性量表总得分': [7],
            '焦虑量表总得分': [4],
            '手机使用时间总得分': [23]
        })
        
        for model_name, file_name in model_files.items():
            model_path = models_dir / file_name
            if model_path.exists():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        model.predict(test_data)
                        working_models.append(model_name)
                except:
                    continue
        
        if working_models:
            print(f"✅ 可用模型: {', '.join(working_models)}")
            return True
        else:
            print("❌ 没有可用的模型")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def start_streamlit():
    """启动Streamlit应用"""
    print("🚀 启动抑郁量表预测应用...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        # 启动streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🌐 应用将在 http://localhost:8501 运行")
        print("按 Ctrl+C 停止应用")
        
        process = subprocess.Popen(cmd, env=env)
        
        # 等待用户中断
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 正在停止应用...")
            process.terminate()
            process.wait()
            print("✅ 应用已停止")
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("🧠 抑郁量表得分预测应用")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 测试模型
    if not test_models():
        print("⚠️  部分模型不可用，但应用仍可运行")
    
    # 启动应用
    start_streamlit()

if __name__ == "__main__":
    main() 