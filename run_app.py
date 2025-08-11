#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit应用启动脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """启动Streamlit应用"""
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    app_file = current_dir / "streamlit_app.py"
    
    if not app_file.exists():
        print("❌ 找不到streamlit_app.py文件")
        return
    
    print("🚀 Starting Depression Scale Score Prediction Application...")
    print(f"📁 应用文件: {app_file}")
    print("🌐 应用将在浏览器中自动打开")
    print("⏹️  按 Ctrl+C 停止应用")
    print("-" * 50)
    
    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n✅ 应用已停止")
    except Exception as e:
        print(f"❌ 启动应用时出错: {e}")

if __name__ == "__main__":
    main() 