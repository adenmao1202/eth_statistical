#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event Study Module
Responsible for analyzing the impact of ETH price drops on other cryptocurrencies
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import time
import json
import math
from matplotlib.ticker import MaxNLocator

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_fetcher import DataFetcher
from utils import calculate_abnormal_returns, calculate_effect_size
import traceback


class EventStudy:
    """Analyzer for cryptocurrency performance during ETH price drops"""
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the event study analyzer
        
        Args:
            data_fetcher (DataFetcher): Data fetcher instance
        """
        self.data_fetcher = data_fetcher
        self.reference_symbol = config.DEFAULT_REFERENCE_SYMBOL
        
        # Cache for analysis results
        self.event_periods = []
        self.abnormal_returns = {}
        self.event_window_minutes = config.EVENT_WINDOW_MINUTES
        
        # 输出目录
        self.output_dir = "results/event_study/default"
    
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
                start_time = idx
                
                # Define event window
                pre_event_window = timedelta(minutes=self.event_window_minutes)
                post_event_window = timedelta(minutes=self.event_window_minutes)
                
                # Calculate event window boundaries
                event_start = start_time - pre_event_window
                event_end = start_time + post_event_window
                
                # Calculate event metrics
                event_data = eth_data[(eth_data.index >= event_start) & (eth_data.index <= event_end)]
                if not event_data.empty:
                    event_start_price = event_data.iloc[0]['close']
                    event_end_price = event_data.iloc[-1]['close']
                    drop_price = eth_data.loc[start_time]['close']
                    
                    # 確保 pct_change 是浮點數並轉換為百分比
                    try:
                        pct_change_value = float(row['pct_change']) if isinstance(row['pct_change'], (str, np.ndarray)) else row['pct_change']
                        drop_pct = pct_change_value * 100  # 轉換為百分比
                    except (ValueError, TypeError) as e:
                        print(f"警告：處理 {start_time} 的 pct_change 值時出錯：{e}，使用預設值")
                        drop_pct = -1.0  # 使用預設值
                    
                    # Add to event list - 確保所有時間戳都是 pandas.Timestamp 對象
                    event_periods.append({
                        'event_time': pd.Timestamp(start_time),
                        'start': pd.Timestamp(event_start),
                        'end': pd.Timestamp(event_end),
                        'start_price': event_start_price,
                        'drop_price': drop_price,
                        'end_price': event_end_price,
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
                # 確保時間戳比較是正確的
                try:
                    # 檢查這個事件是否與任何已過濾事件重疊
                    event_start = pd.Timestamp(event['start'])
                    event_end = pd.Timestamp(event['end'])
                    filtered_start = pd.Timestamp(filtered_event['start'])
                    filtered_end = pd.Timestamp(filtered_event['end'])
                    
                    if ((event_start <= filtered_end and event_end >= filtered_start) or
                        (filtered_start <= event_end and filtered_end >= event_start)):
                        overlapping = True
                        break
                except Exception as e:
                    print(f"Error comparing events: {e}")
                    # 保守起見，如果比較失敗則視為重疊
                    overlapping = True
                    break
            
            if not overlapping:
                filtered_events.append(event)
        
        return filtered_events
    
    def calculate_abnormal_returns(self, coin_data: Dict[str, pd.DataFrame], 
                                  event_periods: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Calculate abnormal returns for all coins during event periods
        
        Args:
            coin_data (Dict[str, pd.DataFrame]): Dictionary of coin data
            event_periods (List[Dict[str, Any]]): List of event periods
            
        Returns:
            Dict[str, List[float]]: Dictionary of abnormal returns for each coin
        """
        abnormal_returns = {}
        
        for symbol, df in coin_data.items():
            if symbol != self.reference_symbol:  # Skip ETH
                abnormal_returns[symbol] = []
                
                for period in event_periods:
                    try:
                        start_time = period['start']
                        end_time = period['end']
                        event_time = period['event_time']
                        
                        # 確保時間戳是正確的類型
                        if isinstance(df.index, pd.RangeIndex):
                            # 如果索引是 RangeIndex，我們需要用位置索引而不是時間戳
                            # 在這種情況下，我們可以通過時間戳查找最接近的索引位置
                            closest_start = df.index[0]
                            closest_event = df.index[len(df) // 2]  # 簡單假設事件在中間
                            closest_end = df.index[-1]
                            
                            # 獲取對應數據
                            pre_event_data = df.iloc[:len(df) // 2]
                            post_event_data = df.iloc[len(df) // 2:]
                        else:
                            # 正常情況下使用時間戳索引
                            # 確保時間戳是 pandas.Timestamp 類型
                            if not isinstance(start_time, pd.Timestamp):
                                start_time = pd.Timestamp(start_time)
                            if not isinstance(event_time, pd.Timestamp):
                                event_time = pd.Timestamp(event_time)
                            if not isinstance(end_time, pd.Timestamp):
                                end_time = pd.Timestamp(end_time)
                                
                            # 獲取事件前後的數據
                            pre_event_data = df[(df.index >= start_time) & (df.index < event_time)]
                            post_event_data = df[(df.index >= event_time) & (df.index <= end_time)]
                        
                        # 計算異常收益
                        if not pre_event_data.empty and not post_event_data.empty:
                            # Calculate expected return based on pre-event data
                            pre_event_return = (pre_event_data.iloc[-1]['close'] / pre_event_data.iloc[0]['close'] - 1)
                            
                            # Calculate actual return in post-event period
                            post_event_return = (post_event_data.iloc[-1]['close'] / post_event_data.iloc[0]['close'] - 1)
                            
                            # Calculate abnormal return
                            abnormal_return = (post_event_return - pre_event_return) * 100  # Convert to percentage
                            
                            abnormal_returns[symbol].append(abnormal_return)
                    except Exception as e:
                        print(f"Error calculating abnormal returns for {symbol}: {e}")
                        continue
        
        return abnormal_returns
    
    def calculate_statistical_significance(self, abnormal_returns: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Calculate statistical significance of abnormal returns
        
        Args:
            abnormal_returns (Dict[str, List[float]]): Dictionary of abnormal returns
            
        Returns:
            pd.DataFrame: DataFrame with statistical significance metrics
        """
        results = []
        
        for symbol, returns in abnormal_returns.items():
            try:
                # 確保返回是數值列表且長度足夠
                if not returns or len(returns) <= 1:
                    print(f"跳過 {symbol}：樣本量不足 (n={len(returns) if returns else 0})")
                    continue
                
                # 確保所有返回都是浮點數
                numeric_returns = []
                for ret in returns:
                    try:
                        numeric_returns.append(float(ret))
                    except (ValueError, TypeError):
                        # 跳過非數值項
                        pass
                
                if len(numeric_returns) <= 1:
                    print(f"跳過 {symbol}：有效數值樣本量不足 (n={len(numeric_returns)})")
                    continue
                
                mean_ar = np.mean(numeric_returns)
                std_ar = np.std(numeric_returns)
                t_stat, p_value = stats.ttest_1samp(numeric_returns, 0)
                
                # 計算效應大小和解釋
                effect_size, effect_interpretation = calculate_effect_size(mean_ar, std_ar)
                
                results.append({
                    'symbol': symbol,
                    'mean_abnormal_return': float(mean_ar),
                    'std_dev': float(std_ar),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < config.SIGNIFICANCE_LEVEL),
                    'sample_size': int(len(numeric_returns)),
                    'effect_size': float(effect_size),
                    'effect_interpretation': effect_interpretation
                })
            except Exception as e:
                print(f"處理 {symbol} 的統計顯著性時出錯：{str(e)}")
        
        # Convert to DataFrame and sort by mean abnormal return
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.sort_values('mean_abnormal_return', ascending=False, inplace=True)
        
        return results_df
    
    def visualize_event_impact(self, stats_df: pd.DataFrame, top_n: int = 10) -> None:
        """
        Visualize the impact of ETH price drops on other coins
        
        Args:
            stats_df (pd.DataFrame): DataFrame with statistical significance metrics
            top_n (int): Number of top coins to display
        """
        if stats_df.empty:
            print("No data to visualize")
            return
            
        # Get top and bottom performers
        top_performers = stats_df.head(top_n)
        bottom_performers = stats_df.tail(top_n)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot top performers
        sns.barplot(x='mean_abnormal_return', y='symbol', data=top_performers, 
                   hue='significant', palette=['lightgray', 'green'], ax=ax1)
        ax1.set_title(f'Top {top_n} Performers After ETH Price Drops')
        ax1.set_xlabel('Mean Abnormal Return (%)')
        ax1.set_ylabel('Symbol')
        
        # Plot bottom performers
        sns.barplot(x='mean_abnormal_return', y='symbol', data=bottom_performers, 
                   hue='significant', palette=['lightgray', 'red'], ax=ax2)
        ax2.set_title(f'Bottom {top_n} Performers After ETH Price Drops')
        ax2.set_xlabel('Mean Abnormal Return (%)')
        ax2.set_ylabel('Symbol')
        
        plt.tight_layout()
        
        # 保存到指定目录
        plt.savefig(f"{self.output_dir}/eth_drop_impact.png", dpi=300)
        plt.close()
    
    def create_interactive_visualization(self, stats_df: pd.DataFrame, abnormal_returns: Dict[str, List[float]], event_periods: List[Dict[str, Any]]) -> None:
        """
        Create interactive visualization of coin performance during ETH price drops
        
        Args:
            stats_df (pd.DataFrame): DataFrame with statistical significance metrics
            abnormal_returns (Dict[str, List[float]]): Dictionary of abnormal returns
            event_periods (List[Dict[str, Any]]): List of event periods
        """
        if stats_df.empty:
            print("No data to visualize")
            return
        
        # Create interactive scatter plot
        fig = px.scatter(
            stats_df,
            x='mean_abnormal_return',
            y='t_statistic',
            color='significant',
            size='sample_size',
            hover_name='symbol',
            labels={
                'mean_abnormal_return': 'Mean Abnormal Return (%)',
                't_statistic': 't-statistic',
                'significant': 'Statistically Significant',
                'sample_size': 'Sample Size'
            },
            title='ETH Price Drop Impact Analysis'
        )
        
        # Add horizontal line at t=0
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        # Add vertical line at abnormal return=0
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            legend_title_text='Significant',
            xaxis_title='Mean Abnormal Return (%)',
            yaxis_title='t-statistic',
            hovermode='closest'
        )
        
        # 保存到指定目录
        fig.write_html(f"{self.output_dir}/eth_drop_impact_interactive.html")
    
    def run_event_study(self, days: int = 30, timeframe: str = '1m',
                      drop_threshold: float = config.MIN_DROP_PCT,
                      window_size: int = config.DEFAULT_WINDOW_SIZE,
                      top_n: int = 10, end_date: Optional[str] = None,
                      use_cache: bool = True,
                      progress_callback: Optional[Callable] = None,
                      output_dir: str = "results") -> pd.DataFrame:
        """
        Run full event study analysis
        
        Args:
            days (int): Number of days to analyze
            timeframe (str): Timeframe to fetch data
            drop_threshold (float): Threshold for identifying price drops
            window_size (int): Window size for smoothing prices
            top_n (int): Number of top coins to display
            end_date (str, optional): End date in YYYY-MM-DD format
            use_cache (bool): Whether to use cached data
            progress_callback (Callable, optional): Callback function for updating progress
            output_dir (str): Directory to save results
            
        Returns:
            pd.DataFrame: DataFrame with event study results
        """
        self.output_dir = output_dir
        
        # Initialize progress callback
        def update_progress(description: str, percent: float) -> None:
            if progress_callback:
                try:
                    # 确保percent是数字 (0-1范围)
                    percent_float = float(percent) / 100.0 if percent > 1 else float(percent)
                    progress_callback(percent_float, description)
                except Exception as e:
                    print(f"更新进度时出错: {e}")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set end date if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        end_date_str = end_date
        
        # Store parameters
        self.event_params = {
            'days': days,
            'timeframe': timeframe,
            'drop_threshold': drop_threshold,
            'window_size': window_size,
            'end_date': end_date,
            'use_cache': use_cache
        }
        
        # Initialize for results
        abnormal_returns = {}
        event_periods = []
        
        # Fetch ETH data
        try:
            update_progress("获取ETH历史数据", 0.1)
            eth_data = self.data_fetcher.fetch_historical_data(
                symbol=self.reference_symbol,
                interval=timeframe,
                days=days,
                end_date=end_date_str,
                use_cache=use_cache
            )
            
            if eth_data.empty:
                print(f"Error: Could not fetch data for {self.reference_symbol}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching ETH data: {e}")
            return pd.DataFrame()
            
        # Update progress
        update_progress("识别ETH价格下跌事件", 0.2)
        
        # Identify ETH drop events
        try:
            event_periods = self.identify_eth_drop_events(
                eth_data=eth_data,
                drop_threshold=drop_threshold,
                window_size=window_size
            )
            
            if not event_periods:
                print(f"No significant ETH price drop events found with threshold {drop_threshold}")
                return pd.DataFrame()
                
            # Store event periods for future reference
            self.event_periods = event_periods
        except Exception as e:
            print(f"Error identifying ETH drop events: {e}")
            return pd.DataFrame()
        
        # Update progress
        update_progress("获取其他币种数据", 0.4)
        
        # Fetch data for top volume coins
        try:
            coin_data = self.data_fetcher.get_all_data(
                timeframe=timeframe,
                days=days,
                end_date=end_date_str,
                use_cache=use_cache
            )
        except Exception as e:
            print(f"Error fetching coin data: {e}")
            return pd.DataFrame()
        
        # Update progress
        update_progress("计算异常收益", 0.6)
        
        # Calculate abnormal returns
        try:
            abnormal_returns = self.calculate_abnormal_returns(coin_data, event_periods)
            
            # Store abnormal returns for future reference
            self.abnormal_returns = abnormal_returns
        except Exception as e:
            print(f"Error calculating abnormal returns: {e}")
            traceback.print_exc()
            return pd.DataFrame()
        
        # Update progress
        update_progress("计算统计显著性", 0.75)
        
        # Calculate statistical significance
        try:
            stats_df = self.calculate_statistical_significance(abnormal_returns)
        except Exception as e:
            print(f"Error calculating statistical significance: {e}")
            return pd.DataFrame()
        
        # Update progress
        update_progress("生成可视化图表", 0.9)
            
        # Generate visualizations
        if not stats_df.empty:
            try:
                self.visualize_event_impact(stats_df, top_n)
                self.create_interactive_visualization(stats_df, abnormal_returns, event_periods)
            except Exception as e:
                print(f"Error generating visualizations: {e}")
        
        # Update progress
        update_progress("分析完成", 1.0)
        
        return stats_df 