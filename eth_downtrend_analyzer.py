#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH下跌期間穩定幣種分析器
此腳本分析ETH下跌期間表現穩定的加密貨幣，以及它們在ETH反轉時的表現。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from binance.client import Client
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import argparse
from crypto_correlation_analyzer import CryptoCorrelationAnalyzer

warnings.filterwarnings('ignore')

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='分析ETH下跌期間表現穩定的加密貨幣')
    
    # 這邊修改 threshold 
    parser.add_argument('--days', type=int, default=30,
                        help='分析的天數範圍 (默認: 90)')
    
    parser.add_argument('--downtrend', type=float, default=-0.05,
                        help='ETH下跌的閾值，例如 -0.05 表示下跌5% (默認: -0.05)')
    
    parser.add_argument('--rebound', type=float, default=0.03,
                        help='ETH反彈的閾值，例如 0.03 表示反彈3% (默認: 0.03)')
    
    parser.add_argument('--window', type=int, default=5,
                        help='滑動窗口大小，用於識別趨勢 (默認: 5)')
    
    parser.add_argument('--top', type=int, default=10,
                        help='顯示表現最好的前N個加密貨幣 (默認: 10)')
    
    parser.add_argument('--api_key', type=str, default=None,
                        help='Binance API密鑰 (可選)')
    
    parser.add_argument('--api_secret', type=str, default=None,
                        help='Binance API密碼 (可選)')
    
    parser.add_argument('--max_klines', type=int, default=200,
                        help='每個時間週期獲取的最大K線數量 (默認: 200)')
    
    parser.add_argument('--use_cache', action='store_true',
                        help='使用緩存數據 (如果可用)')
    
    return parser.parse_args()

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 設置日期範圍
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    print(f"分析ETH下跌期間的穩定幣種")
    print(f"時間範圍: {start_date} 到 {end_date}")
    print(f"ETH下跌閾值: {args.downtrend * 100}%")
    print(f"ETH反彈閾值: {args.rebound * 100}%")
    print(f"滑動窗口大小: {args.window}")
    print(f"顯示前 {args.top} 個表現最好的幣種")
    
    # 初始化分析器
    analyzer = CryptoCorrelationAnalyzer(
        api_key=args.api_key,
        api_secret=args.api_secret,
        max_klines=args.max_klines
    )
    
    # 執行標準相關性分析以獲取基礎數據
    print("\n執行基礎相關性分析...")
    results = analyzer.analyze_all_timeframes(start_date, end_date, top_n=args.top, use_cache=args.use_cache)
    
    # 分析ETH下跌期間的穩定幣種
    print("\n分析ETH下跌期間的穩定幣種...")
    stable_coins, best_rebounders, best_combined = analyzer.analyze_eth_downtrend_resilience(
        results, 
        start_date, 
        end_date, 
        downtrend_threshold=args.downtrend,
        rebound_threshold=args.rebound,
        window_size=args.window,
        top_n=args.top
    )
    
    if stable_coins is None:
        print("在指定時間範圍內沒有找到符合條件的ETH下跌期間")
        return
    
    print("\n分析完成！")
    print(f"結果已保存到 'eth_market_cycle_analysis.csv'")
    print(f"圖表已保存為 'eth_downtrend_stable_coins.png', 'eth_rebound_best_coins.png', 'eth_cycle_best_coins.png'")

if __name__ == "__main__":
    main() 