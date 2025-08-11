#!/usr/bin/env python3
"""
Depression Scale Score Prediction Application Startup Script
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # 设置当前目录
    app_dir = Path(__file__).parent
    os.chdir(app_dir)
    
    print("🚀 Starting Depression Scale Score Prediction Application...")
    print(f"📁 Application directory: {app_dir}")
    
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
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)
    
    print("✅ All required files checked")
    
    # 启动应用
    try:
        print("🌟 Starting Streamlit application...")
        print("📱 Application will open in browser: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop application")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', '127.0.0.1'
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 