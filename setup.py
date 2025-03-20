#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH Statistical Analysis Framework Setup
用於安裝和設置ETH統計分析框架所需的依賴和環境
"""

import os
import sys
import subprocess
import platform
from setuptools import setup, find_packages

# 定義安裝依賴
REQUIRED_PACKAGES = [
    'pandas>=1.3.0',
    'numpy>=1.20.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'python-binance>=1.0.15',
    'tqdm>=4.62.0',
    'requests>=2.26.0',
    'python-dateutil>=2.8.2',
    'scipy>=1.7.0',
    'statsmodels>=0.13.0',
    'rich>=10.0.0',
    'xgboost>=1.5.0'
]

# 為開發者準備的額外依賴
EXTRA_PACKAGES = {
    'dev': [
        'jupyter>=1.0.0',
        'pylint>=2.8.0',
        'black>=21.5b2',
        'pytest>=6.2.5',
    ]
}

def check_python_version():
    """檢查Python版本是否符合要求"""
    min_version = (3, 8)
    if sys.version_info < min_version:
        sys.exit(f"錯誤: 需要Python {min_version[0]}.{min_version[1]}或更高版本，"
                f"當前版本是 {sys.version_info.major}.{sys.version_info.minor}")

def setup_directories():
    """設置所需的目錄結構"""
    # 確保主要目錄存在
    directories = [
        'main/historical_data',
        'main/historical_data/cache',
        'results',
        'results/event_study',
        'results/clustering',
        'results/hypothesis_testing'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 已創建目錄: {directory}")

def print_setup_summary():
    """打印安裝摘要信息"""
    print("\n" + "="*50)
    print("ETH 統計分析框架設置完成！")
    print("="*50)
    print("\n運行以下命令開始使用:")
    print("python main/main.py --run event_study --timeframe 1m --days 30 --drop_threshold -0.01")
    print("\n查看更多選項:")
    print("python main/main.py")
    print("\n" + "="*50)

def main():
    """主安裝函數"""
    # 檢查Python版本
    check_python_version()
    
    # 設置必要的目錄
    setup_directories()
    
    # 安裝依賴
    setup(
        name="eth_statistical",
        version="1.0.0",
        description="ETH Statistical Analysis Framework",
        author="mouyasushi",
        packages=find_packages(),
        install_requires=REQUIRED_PACKAGES,
        extras_require=EXTRA_PACKAGES,
        python_requires=">=3.8",
    )
    
    # 打印摘要信息
    print_setup_summary()

if __name__ == "__main__":
    main() 