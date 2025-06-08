#!/usr/bin/env python3
"""
SHAP安装脚本 - 用于云端环境
"""
import subprocess
import sys
import os

def install_shap():
    """安装SHAP包，使用多种降级策略"""
    
    # 策略1: 尝试预编译的wheel
    try:
        print("📦 尝试安装预编译的SHAP...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--only-binary=all", "shap==0.41.0"
        ])
        print("✅ SHAP安装成功 (预编译)")
        return True
    except:
        pass
    
    # 策略2: 尝试较旧版本
    try:
        print("📦 尝试安装旧版本SHAP...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "shap==0.40.0", "--no-cache-dir"
        ])
        print("✅ SHAP安装成功 (v0.40.0)")
        return True
    except:
        pass
    
    # 策略3: 最小化安装
    try:
        print("📦 尝试最小化SHAP安装...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "shap==0.39.0", "--no-deps"
        ])
        print("✅ SHAP安装成功 (最小化)")
        return True
    except:
        pass
    
    print("❌ SHAP安装失败，应用将在无SHAP模式下运行")
    return False

if __name__ == "__main__":
    install_shap() 