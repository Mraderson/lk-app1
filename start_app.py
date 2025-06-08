#!/usr/bin/env python3
"""
抑郁量表得分预测应用启动脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # 设置当前目录
    app_dir = Path(__file__).parent
    os.chdir(app_dir)
    
    print("🚀 启动抑郁量表得分预测应用...")
    print(f"📁 应用目录: {app_dir}")
    
    # 检查必要文件
    required_files = [
        'streamlit_app.py',
        'models/XGBoost_model.pkl',
        'models/LightGBM_model.pkl',
        'data/量表总分完整数据.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)
    
    print("✅ 所有必要文件检查完毕")
    
    # 启动应用
    try:
        print("🌟 正在启动Streamlit应用...")
        print("📱 应用将在浏览器中打开: http://localhost:8501")
        print("🛑 按 Ctrl+C 停止应用")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', '127.0.0.1'
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 应用已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 