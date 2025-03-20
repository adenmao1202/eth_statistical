#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hypothesis Testing Module
For testing statistical significance of altcoin returns before and after ETH price drops
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# 統計學相關模塊導入嘗試多種可能的導入方式
try:
    from scipy.stats.multitest import multipletests
except ImportError:
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None
        print("警告: 無法導入 multipletests 函數，將不進行多重檢驗校正")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_fetcher import DataFetcher
from utils import calculate_abnormal_returns, calculate_effect_size


class HypothesisTesting:
    """Statistical hypothesis testing for altcoin returns before/after ETH price drops"""
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the hypothesis testing analyzer
        
        Args:
            data_fetcher (DataFetcher): Data fetcher instance
        """
        self.data_fetcher = data_fetcher
        self.reference_symbol = config.DEFAULT_REFERENCE_SYMBOL
        
        # Analysis results
        self.event_periods = []
        self.pre_event_returns = {}
        self.post_event_returns = {}
        self.test_results = pd.DataFrame()
        
        # Parameters
        self.pre_event_window = config.PRE_EVENT_WINDOW
        self.post_event_window = config.POST_EVENT_WINDOW
        self.significance_level = config.SIGNIFICANCE_LEVEL
        
        # 输出目录
        self.output_dir = "results/hypothesis_testing/default"
    
    def identify_eth_drop_events(self, eth_data: pd.DataFrame, 
                                drop_threshold: float = config.MIN_DROP_PCT,
                                window_size: int = config.DETECTION_WINDOW_SIZE) -> List[Dict[str, Any]]:
        """
        Identify significant ETH price drop events
        
        Args:
            eth_data (pd.DataFrame): ETH price data
            drop_threshold (float): Threshold for identifying significant drops (negative value)
            window_size (int): Window size for calculating price changes
            
        Returns:
            List[Dict[str, Any]]: List of event periods with start/end times and metrics
        """
        if 'close' not in eth_data.columns:
            raise ValueError("Data must contain 'close' column")
        
        print(f"Analyzing {len(eth_data)} data points for ETH price drops")
        print(f"Date range: {eth_data.index.min()} to {eth_data.index.max()}")
        print(f"Drop threshold: {drop_threshold}, Window size: {window_size}")
        
        # Calculate percentage changes within the detection window
        eth_data['pct_change'] = eth_data['close'].pct_change(window_size)
        
        # Identify potential event periods where ETH drops at least drop_threshold % in window_size periods
        event_periods = []
        
        for idx, row in eth_data.iterrows():
            if pd.isna(row['pct_change']):
                continue
                
            if row['pct_change'] <= drop_threshold:
                # Found a drop event
                event_time = idx
                
                # Define event window boundaries
                pre_event_start = event_time - timedelta(minutes=self.pre_event_window)
                post_event_end = event_time + timedelta(minutes=self.post_event_window)
                
                # Calculate event metrics
                event_data = eth_data[(eth_data.index >= pre_event_start) & (eth_data.index <= post_event_end)]
                if not event_data.empty:
                    pre_event_price = eth_data.loc[pre_event_start:event_time]['close'].iloc[0]
                    event_price = eth_data.loc[event_time]['close']
                    post_event_price = eth_data.loc[event_time:post_event_end]['close'].iloc[-1]
                    drop_pct = row['pct_change'] * 100  # Convert to percentage
                    
                    # Add to event list
                    event_periods.append({
                        'event_time': event_time,
                        'pre_event_start': pre_event_start,
                        'post_event_end': post_event_end,
                        'pre_event_price': pre_event_price,
                        'event_price': event_price,
                        'post_event_price': post_event_price,
                        'drop_pct': drop_pct
                    })
        
        # Filter out overlapping events
        filtered_events = self._filter_overlapping_events(event_periods)
        
        print(f"Identified {len(filtered_events)} ETH drop events")
        
        # Sort by drop percentage (most severe first)
        filtered_events.sort(key=lambda x: x['drop_pct'])
        
        return filtered_events
    
    def _filter_overlapping_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out overlapping events, keeping the most severe drops
        
        Args:
            events (List[Dict[str, Any]]): List of event periods
            
        Returns:
            List[Dict[str, Any]]: Filtered list of events
        """
        if not events:
            return []
            
        # Sort events by drop percentage (ascending, most severe drops first)
        sorted_events = sorted(events, key=lambda x: x['drop_pct'])
        
        filtered_events = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            overlapping = False
            for filtered_event in filtered_events:
                # Check if this event overlaps with any already filtered event
                if ((event['pre_event_start'] <= filtered_event['post_event_end'] and 
                     event['post_event_end'] >= filtered_event['pre_event_start']) or
                    (filtered_event['pre_event_start'] <= event['post_event_end'] and 
                     filtered_event['post_event_end'] >= event['pre_event_start'])):
                    overlapping = True
                    break
            
            if not overlapping:
                filtered_events.append(event)
        
        return filtered_events
    
    def calculate_returns(self, coin_data: Dict[str, pd.DataFrame], 
                         event_periods: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Calculate pre-event and post-event returns for all coins
        
        Args:
            coin_data (Dict[str, pd.DataFrame]): Dictionary of coin data
            event_periods (List[Dict[str, Any]]): List of event periods
            
        Returns:
            Tuple[Dict[str, List[float]], Dict[str, List[float]]]: Pre-event and post-event returns
        """
        pre_event_returns = {}
        post_event_returns = {}
        
        for symbol, df in coin_data.items():
            pre_event_returns[symbol] = []
            post_event_returns[symbol] = []
            
            for event in event_periods:
                event_time = event['event_time']
                pre_start = event['pre_event_start']
                post_end = event['post_event_end']
                
                # Get data for pre-event and post-event windows
                pre_event_data = df[(df.index >= pre_start) & (df.index <= event_time)]
                post_event_data = df[(df.index >= event_time) & (df.index <= post_end)]
                
                # Calculate returns if enough data is available
                if len(pre_event_data) >= 2 and len(post_event_data) >= 2:
                    # Pre-event return
                    pre_return = (pre_event_data['close'].iloc[-1] / pre_event_data['close'].iloc[0] - 1) * 100
                    pre_event_returns[symbol].append(pre_return)
                    
                    # Post-event return
                    post_return = (post_event_data['close'].iloc[-1] / post_event_data['close'].iloc[0] - 1) * 100
                    post_event_returns[symbol].append(post_return)
        
        return pre_event_returns, post_event_returns
    
    def conduct_statistical_tests(self, pre_event_returns: Dict[str, List[float]], 
                                post_event_returns: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Conduct statistical tests on pre and post event returns
        
        Args:
            pre_event_returns (Dict[str, List[float]]): Pre-event returns
            post_event_returns (Dict[str, List[float]]): Post-event returns
            
        Returns:
            pd.DataFrame: DataFrame with test results
        """
        results = []
        
        for symbol in pre_event_returns.keys():
            if symbol not in post_event_returns:
                continue
                
            pre_returns = pre_event_returns[symbol]
            post_returns = post_event_returns[symbol]
            
            # 最小样本量要求
            min_sample_size = 5
            if len(pre_returns) < min_sample_size or len(post_returns) < min_sample_size:
                continue
            
            # 异常值处理 - 使用百分位数截断极端值
            lower_percentile = 1
            upper_percentile = 99
            
            pre_lower = np.percentile(pre_returns, lower_percentile)
            pre_upper = np.percentile(pre_returns, upper_percentile)
            post_lower = np.percentile(post_returns, lower_percentile)
            post_upper = np.percentile(post_returns, upper_percentile)
            
            pre_returns_filtered = [x for x in pre_returns if pre_lower <= x <= pre_upper]
            post_returns_filtered = [x for x in post_returns if post_lower <= x <= post_upper]
            
            # 至少需要一定数量的有效观测值
            if len(pre_returns_filtered) < min_sample_size or len(post_returns_filtered) < min_sample_size:
                continue
                
            # 计算基本统计量
            pre_mean = np.mean(pre_returns_filtered)
            post_mean = np.mean(post_returns_filtered)
            pre_std = np.std(pre_returns_filtered)
            post_std = np.std(post_returns_filtered)
            mean_diff = post_mean - pre_mean
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(post_returns_filtered, pre_returns_filtered)
            
            # 计算效应大小 (Cohen's d)
            # 对于配对样本，使用标准化平均差异
            pooled_std = np.sqrt((pre_std**2 + post_std**2) / 2)
            if pooled_std != 0:  # 避免除以零
                effect_size, effect_interpretation = calculate_effect_size(mean_diff, pooled_std)
            else:
                effect_size, effect_interpretation = 0, "無法計算"
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                w_stat, w_p_value = stats.wilcoxon(post_returns_filtered, pre_returns_filtered)
            except:
                w_stat, w_p_value = np.nan, np.nan
            
            # Kolmogorov-Smirnov test for distribution difference
            try:
                ks_stat, ks_p_value = stats.ks_2samp(post_returns_filtered, pre_returns_filtered)
            except:
                ks_stat, ks_p_value = np.nan, np.nan
            
            # Store results
            results.append({
                'symbol': symbol,
                'sample_size': min(len(pre_returns), len(post_returns)),
                'pre_mean': pre_mean,
                'post_mean': post_mean,
                'pre_std': pre_std,
                'post_std': post_std,
                'mean_diff': mean_diff,
                't_stat': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'effect_interpretation': effect_interpretation,
                'significant': p_value < self.significance_level,
                'wilcoxon_stat': w_stat,
                'wilcoxon_p': w_p_value,
                'ks_stat': ks_stat,
                'ks_p': ks_p_value
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Apply multiple testing correction (Benjamini-Hochberg)
            if multipletests is not None:
                _, corrected_p_values, _, _ = multipletests(results_df['p_value'], alpha=self.significance_level, method='fdr_bh')
                results_df['corrected_p_value'] = corrected_p_values
                results_df['significant_corrected'] = results_df['corrected_p_value'] < self.significance_level
            else:
                # 如果multipletests不可用，則使用原始p值
                results_df['corrected_p_value'] = results_df['p_value']
                results_df['significant_corrected'] = results_df['significant']
            
            # Sort by absolute mean difference
            results_df['abs_mean_diff'] = results_df['mean_diff'].abs()
            results_df.sort_values('abs_mean_diff', ascending=False, inplace=True)
            results_df.drop('abs_mean_diff', axis=1, inplace=True)
        
        return results_df
    
    def visualize_results(self, results_df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Visualize hypothesis testing results
        
        Args:
            results_df (pd.DataFrame): DataFrame with test results
            top_n (int): Number of top results to display
        """
        if results_df.empty:
            print("No data to visualize")
            return
            
        # Get top coins by absolute mean difference
        results_df['abs_mean_diff'] = results_df['mean_diff'].abs()
        top_results = results_df.sort_values('abs_mean_diff', ascending=False).head(top_n)
        results_df.drop('abs_mean_diff', axis=1, inplace=True)
        
        # Create bar plot
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_results['symbol'], top_results['mean_diff'], 
                     color=[('green' if x else 'gray') for x in top_results['significant_corrected']])
        
        # Add red line at y=0
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Symbol')
        plt.ylabel('Mean Return Difference (Post - Pre) %')
        plt.title(f'Top {top_n} Coins by Return Difference After ETH Price Drops')
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color='green'), plt.Rectangle((0,0),1,1, color='gray')]
        labels = ['Statistically Significant', 'Not Significant']
        plt.legend(handles, labels)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/return_difference_top.png', dpi=300)
        plt.close()
        
        # Create scatter plot of pre vs post returns
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='pre_mean', y='post_mean', hue='significant_corrected', 
                       size='sample_size', alpha=0.7, data=results_df)
        
        # Add diagonal line (y=x)
        min_val = min(results_df['pre_mean'].min(), results_df['post_mean'].min())
        max_val = max(results_df['pre_mean'].max(), results_df['post_mean'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # Add labels for selected coins
        for _, row in top_results.head(10).iterrows():
            plt.annotate(row['symbol'], (row['pre_mean'], row['post_mean']), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.xlabel('Pre-Event Mean Return (%)')
        plt.ylabel('Post-Event Mean Return (%)')
        plt.title('Pre-Event vs Post-Event Mean Returns')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pre_post_returns_scatter.png', dpi=300)
        plt.close()
    
    def create_distribution_plots(self, pre_event_returns: Dict[str, List[float]], 
                                post_event_returns: Dict[str, List[float]], 
                                symbols: List[str]) -> None:
        """
        Create distribution plots comparing pre and post event returns
        
        Args:
            pre_event_returns (Dict[str, List[float]]): Pre-event returns for each coin
            post_event_returns (Dict[str, List[float]]): Post-event returns for each coin
            symbols (List[str]): List of symbols to plot
        """
        for symbol in symbols:
            if symbol not in pre_event_returns or symbol not in post_event_returns:
                continue
                
            pre_returns = pre_event_returns[symbol]
            post_returns = post_event_returns[symbol]
            
            if len(pre_returns) < 5 or len(post_returns) < 5:
                continue
                
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot distributions
            sns.histplot(pre_returns, kde=True, ax=ax1, color='blue', label='Pre-Event')
            sns.histplot(post_returns, kde=True, ax=ax2, color='red', label='Post-Event')
            
            # Add statistics to the plot
            pre_mean = np.mean(pre_returns)
            post_mean = np.mean(post_returns)
            pre_std = np.std(pre_returns)
            post_std = np.std(post_returns)
            
            ax1.axvline(pre_mean, color='black', linestyle='--')
            ax2.axvline(post_mean, color='black', linestyle='--')
            
            ax1.set_title(f'Pre-Event Returns for {symbol}\nMean: {pre_mean:.2f}%, Std: {pre_std:.2f}%')
            ax2.set_title(f'Post-Event Returns for {symbol}\nMean: {post_mean:.2f}%, Std: {post_std:.2f}%')
            
            ax1.set_xlabel('Return (%)')
            ax2.set_xlabel('Return (%)')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/distribution_{symbol}.png', dpi=300)
            plt.close()
    
    def summarize_results(self, results_df: pd.DataFrame) -> None:
        """
        Summarize hypothesis testing results
        
        Args:
            results_df (pd.DataFrame): DataFrame with test results
        """
        if results_df.empty:
            print("No results to summarize")
            return
            
        # Count significant results
        significant = results_df[results_df['significant_corrected']]
        
        print("\n=== Hypothesis Testing Results Summary ===")
        print(f"Total coins analyzed: {len(results_df)}")
        print(f"Coins with significant change: {len(significant)} ({len(significant)/len(results_df)*100:.1f}%)")
        
        # Count positive and negative changes
        positive_sig = significant[significant['mean_diff'] > 0]
        negative_sig = significant[significant['mean_diff'] < 0]
        
        print(f"Coins with significant positive change: {len(positive_sig)} ({len(positive_sig)/len(significant)*100:.1f}% of significant)")
        print(f"Coins with significant negative change: {len(negative_sig)} ({len(negative_sig)/len(significant)*100:.1f}% of significant)")
        
        # Top 5 positive changes
        if not positive_sig.empty:
            print("\n--- Top 5 Positive Changes ---")
            top_positive = positive_sig.sort_values('mean_diff', ascending=False).head(5)
            for _, row in top_positive.iterrows():
                print(f"{row['symbol']}: {row['mean_diff']:.2f}% (p={row['corrected_p_value']:.4f})")
        
        # Top 5 negative changes
        if not negative_sig.empty:
            print("\n--- Top 5 Negative Changes ---")
            top_negative = negative_sig.sort_values('mean_diff', ascending=True).head(5)
            for _, row in top_negative.iterrows():
                print(f"{row['symbol']}: {row['mean_diff']:.2f}% (p={row['corrected_p_value']:.4f})")
    
    def run_hypothesis_testing(self, days: int = 30, timeframe: str = '1m', 
                             drop_threshold: float = config.MIN_DROP_PCT, 
                             window_size: int = config.DETECTION_WINDOW_SIZE,
                             top_n: int = config.DEFAULT_TOP_N, 
                             end_date: Optional[str] = None,
                             use_cache: bool = True,
                             pre_event_window: int = None,
                             post_event_window: int = None,
                             progress_callback: Optional[Callable] = None,
                             output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        運行假設檢驗分析

        Args:
            days: 分析天數
            timeframe: 時間框架('1m', '5m', '15m', '1h', '4h')
            drop_threshold: 認定ETH價格顯著下跌的閾值 (負數，例如-0.01表示1%的下跌)
            window_size: 檢測窗口大小
            top_n: 分析排名前N的幣種 
            end_date: 結束日期，格式為'YYYY-MM-DD'
            use_cache: 是否使用緩存數據
            pre_event_window: 事件前窗口（分鐘）
            post_event_window: 事件後窗口（分鐘）
            progress_callback: 進度回調函數
            output_dir: 輸出目錄

        Returns:
            pd.DataFrame: DataFrame with test results
        """
        # 更新事件窗口大小(如果提供)
        if pre_event_window is not None:
            self.pre_event_window = pre_event_window
        if post_event_window is not None:
            self.post_event_window = post_event_window
            
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
            
        print(f"Running hypothesis testing for {days} days with {timeframe} data")
        
        # Calculate dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Add extra days to allow for event windows
        extra_days = max(2, (self.pre_event_window + self.post_event_window) // (60 * 24) + 1)
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days+extra_days)).strftime('%Y-%m-%d')
        
        print(f"Analysis period: {start_date} to {end_date}")
        print(f"Pre-event window: {self.pre_event_window} minutes, Post-event window: {self.post_event_window} minutes")
        
        # Get ETH data
        if progress_callback:
            progress_callback(0.15, "正在獲取ETH歷史數據...")
            
        eth_data = self.data_fetcher.fetch_historical_data(
            symbol=self.reference_symbol, 
            interval=timeframe, 
            days=days,
            end_date=end_date,
            use_cache=use_cache
        )
        
        if eth_data.empty:
            print(f"Failed to get data for {self.reference_symbol}")
            return pd.DataFrame()
        
        # Identify ETH drop events
        if progress_callback:
            progress_callback(0.25, "正在識別ETH下跌事件...")
            
        self.event_periods = self.identify_eth_drop_events(eth_data, drop_threshold, window_size)
        
        if not self.event_periods:
            print("No significant ETH drop events identified")
            return pd.DataFrame()
        
        # Get data for top volume symbols
        if progress_callback:
            progress_callback(0.35, "正在獲取加密貨幣數據...")
            
        coin_data = self.data_fetcher.get_all_data(
            timeframe=timeframe,
            days=days,
            end_date=end_date,
            use_cache=use_cache,
            include_reference=True  # 确保包含ETH
        )
        
        # Calculate returns
        if progress_callback:
            progress_callback(0.50, "正在計算事件前後收益率...")
            
        self.pre_event_returns, self.post_event_returns = self.calculate_returns(coin_data, self.event_periods)
        
        # Conduct statistical tests
        if progress_callback:
            progress_callback(0.70, "正在進行統計檢驗...")
            
        results_df = self.conduct_statistical_tests(self.pre_event_returns, self.post_event_returns)
        
        # Save results
        results_file = f"{self.output_dir}/hypothesis_testing_results.csv"
        results_df.to_csv(results_file)
        print(f"Saved results to {results_file}")
        
        # Visualize results
        if progress_callback:
            progress_callback(0.85, "正在生成可視化圖表...")
            
        self.visualize_results(results_df, top_n)
        
        # Calculate summary
        if progress_callback:
            progress_callback(0.95, "正在生成結果摘要...")
            
        self.summarize_results(results_df)
        
        # Create distribution plots for notable symbols
        if not results_df.empty:
            significant = results_df[results_df['p_value'] < self.significance_level]
            if not significant.empty:
                # Create distribution plots for top 5 highest absolute mean difference
                top_symbols = significant.reindex(significant['mean_diff'].abs().sort_values(ascending=False).index).head(5)['symbol'].tolist()
                self.create_distribution_plots(self.pre_event_returns, self.post_event_returns, top_symbols)
        
        if progress_callback:
            progress_callback(1.0, "分析完成！")
            
        return results_df 