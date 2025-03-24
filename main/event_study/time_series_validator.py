#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time Series Validation Module
Assesses the stability of cryptocurrency performance across different time periods
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_fetcher import DataFetcher

console = Console()

class TimeSeriesValidator:
    """
    Time Series Validator
    Assesses the stability of cryptocurrency performance across different time periods
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the time series validator
        
        Args:
            data_fetcher: DataFetcher instance
        """
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def validate_across_time_periods(self, 
                                   symbols: List[str], 
                                   total_days: int = 360, 
                                   period_length: int = 90, 
                                   timeframe: str = '1h',
                                   eth_event_params: Dict = None
                                   ) -> pd.DataFrame:
        """
        Validate cryptocurrency performance across multiple time periods
        
        Args:
            symbols: List of symbols to analyze
            total_days: Total days to analyze
            period_length: Length of each period in days
            timeframe: Time interval
            eth_event_params: Parameters for ETH event identification
            
        Returns:
            pd.DataFrame: DataFrame with stability scores
        """
        self.logger.info(f"Validating {len(symbols)} coins across multiple time periods")
        
        if eth_event_params is None:
            eth_event_params = {
                'drop_threshold': -0.02, 
                'consecutive_drops': 1,
                'volume_factor': 1.5
            }
        
        # Calculate number of periods
        num_periods = total_days // period_length
        self.logger.info(f"Analyzing {num_periods} periods of {period_length} days each")
        
        period_results = []
        symbols_performance = {symbol: [] for symbol in symbols}
        
        # Analyze each time period
        for i in range(num_periods):
            # Calculate time period
            end_date = (datetime.now() - timedelta(days=i*period_length)).strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=(i+1)*period_length)).strftime('%Y-%m-%d')
            
            self.logger.info(f"Analyzing period {i+1}/{num_periods}: {start_date} to {end_date}")
            
            try:
                # Get ETH events
                eth_events = self._identify_eth_events(
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    **eth_event_params
                )
                
                if not eth_events:
                    self.logger.warning(f"No ETH events found in period {start_date} to {end_date}")
                    continue
                    
                # Analyze performance
                period_performance = self._analyze_period_performance(
                    symbols=symbols,
                    eth_events=eth_events,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                )
                
                # Store period results
                period_results.append({
                    'period': f"{start_date} to {end_date}",
                    'eth_events': len(eth_events),
                    'coin_count': len(period_performance)
                })
                
                # Add to symbol performance
                for symbol, performance in period_performance.items():
                    if symbol in symbols_performance:
                        symbols_performance[symbol].append(performance)
            
            except Exception as e:
                self.logger.error(f"Error analyzing period {start_date} to {end_date}: {e}")
        
        # Calculate stability scores
        stability_scores = {}
        for symbol, performances in symbols_performance.items():
            if performances:
                # Calculate mean performance and variability
                mean_performance = np.mean(performances)
                performance_std = np.std(performances)
                
                # Calculate stability score (higher is better)
                if performance_std == 0:
                    stability_score = mean_performance * 100  # Perfect stability, apply mean only
                else:
                    stability_score = mean_performance / (performance_std + 0.0001)  # Add small constant to avoid division by zero
                
                # Add more weight to coins with data from more periods
                period_coverage = len(performances) / num_periods
                weighted_score = stability_score * period_coverage
                
                stability_scores[symbol] = {
                    'symbol': symbol,
                    'mean_performance': mean_performance,
                    'performance_std': performance_std,
                    'period_coverage': period_coverage,
                    'stability_score': stability_score,
                    'weighted_score': weighted_score
                }
        
        # Create results DataFrame
        results_df = pd.DataFrame(list(stability_scores.values()))
        if not results_df.empty:
            # Sort by weighted stability score
            results_df = results_df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
            
            # Generate output report
            self._generate_stability_report(results_df, symbols, period_results)
        else:
            self.logger.warning("No valid stability scores calculated")
            results_df = pd.DataFrame(columns=['symbol', 'mean_performance', 'performance_std', 
                                           'period_coverage', 'stability_score', 'weighted_score'])
        
        return results_df
    
    def _identify_eth_events(self, 
                           start_date: str, 
                           end_date: str, 
                           timeframe: str,
                           drop_threshold: float = -0.02,
                           consecutive_drops: int = 1,
                           volume_factor: float = 1.5,
                           **kwargs) -> List[Dict]:
        """
        Identify ETH price drop events
        
        Args:
            start_date: Start date
            end_date: End date
            timeframe: Time interval
            drop_threshold: Price drop threshold (negative value)
            consecutive_drops: Number of consecutive candles with drops
            volume_factor: Volume increase factor
            
        Returns:
            List[Dict]: List of ETH events
        """
        # Calculate days difference
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days + 1  # Add 1 to include end date
        
        # Get ETH historical data
        eth_data = self.data_fetcher.fetch_historical_data(
            symbol='ETHUSDT', 
            interval=timeframe,
            days=days_diff,
            end_date=end_date,
            use_cache=True
        )
        
        if eth_data.empty:
            self.logger.warning(f"No ETH data available for period {start_date} to {end_date}")
            return []
        
        # Calculate price changes and volume ratios
        eth_data['pct_change'] = eth_data['close'].pct_change()
        eth_data['volume_ratio'] = eth_data['volume'] / eth_data['volume'].rolling(window=10).mean()
        
        # Mark drop candles
        eth_data['is_drop'] = eth_data['pct_change'] < 0
        eth_data['consecutive_drops'] = eth_data['is_drop'].rolling(consecutive_drops).sum()
        
        # Identify significant drops
        significant_drops = eth_data[
            (eth_data['pct_change'] <= drop_threshold) & 
            (eth_data['consecutive_drops'] >= consecutive_drops) &
            (eth_data['volume_ratio'] >= volume_factor)
        ]
        
        # Filter events to be at least 24 hours apart
        filtered_events = []
        last_event_time = None
        
        for idx, row in significant_drops.iterrows():
            event_time = idx
            
            # Ensure events are at least 24 hours apart
            if last_event_time is None or (event_time - last_event_time).total_seconds() >= 24*3600:
                filtered_events.append({
                    'time': event_time,
                    'price': row['close'],
                    'pct_change': row['pct_change'],
                    'volume_ratio': row['volume_ratio']
                })
                last_event_time = event_time
        
        self.logger.info(f"Identified {len(filtered_events)} ETH events in period {start_date} to {end_date}")
        return filtered_events
    
    def _analyze_period_performance(self, 
                                   symbols: List[str],
                                   eth_events: List[Dict],
                                   start_date: str,
                                   end_date: str,
                                   timeframe: str,
                                   pre_event_window: int = 15,
                                   post_event_window: int = 45) -> Dict[str, float]:
        """
        Analyze coin performance in a single time period
        
        Args:
            symbols: List of coin symbols
            eth_events: List of ETH events
            start_date: Start date
            end_date: End date
            timeframe: Time interval
            pre_event_window: Pre-event window minutes
            post_event_window: Post-event window minutes
            
        Returns:
            Dict[str, float]: Dictionary of coin performances
        """
        # Calculate days difference
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_dt - start_dt).days + 1  # Add 1 to include end date
        
        # Get historical data for all coins
        coin_data = {}
        for symbol in symbols:
            try:
                data = self.data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval=timeframe,
                    days=days_diff,
                    end_date=end_date,
                    use_cache=True
                )
                if not data.empty:
                    coin_data[symbol] = data
            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {e}")
        
        # Calculate performance around each ETH event
        performances = {}
        
        for symbol, data in coin_data.items():
            symbol_performances = []
            
            # Skip if no data available
            if data is None or len(data) == 0:
                continue
                
            # 處理可能的重複索引問題
            if not data.index.is_unique:
                self.logger.warning(f"Duplicate timestamps found for {symbol}, fixing by keeping last values")
                # 保留最後一個出現的值（通常是最新的）
                data = data.loc[~data.index.duplicated(keep='last')]
                
            for event in eth_events:
                event_time = event['time']
                
                try:
                    # Find pre-event reference price
                    pre_event_price = None
                    closest_idx = data.index.get_indexer([event_time], method='nearest')[0]
                    
                    if closest_idx >= 0 and closest_idx < len(data):
                        # Look back to find pre-event price
                        pre_event_idx = max(0, closest_idx - pre_event_window)
                        pre_event_price = data.iloc[pre_event_idx]['close']
                        
                        # Look forward to find post-event price
                        post_event_idx = min(len(data) - 1, closest_idx + post_event_window)
                        post_event_price = data.iloc[post_event_idx]['close']
                        
                        # Calculate performance
                        if pre_event_price is not None and pre_event_price > 0:
                            performance = (post_event_price / pre_event_price) - 1
                            symbol_performances.append(performance)
                
                except Exception as e:
                    self.logger.warning(f"Error calculating performance for {symbol} at {event_time}: {e}")
            
            # Average performance across all events in this period
            if symbol_performances:
                performances[symbol] = np.mean(symbol_performances)
        
        self.logger.info(f"Calculated performance for {len(performances)} coins in period {start_date} to {end_date}")
        return performances
    
    def _generate_stability_report(self, results_df: pd.DataFrame, symbols: List[str], period_results: List[Dict]):
        """
        Generate stability analysis report
        
        Args:
            results_df: DataFrame with stability results
            symbols: List of analyzed symbols
            period_results: Period analysis results
        """
        try:
            # Create report file
            report_path = os.path.join(self.results_dir, f"stability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with open(report_path, 'w') as f:
                f.write("=======================================\n")
                f.write("Time Series Validation Stability Report\n")
                f.write("=======================================\n\n")
                
                # Overall analysis
                f.write(f"Total symbols analyzed: {len(symbols)}\n")
                f.write(f"Symbols with valid stability scores: {len(results_df)}\n")
                f.write(f"Periods analyzed: {len(period_results)}\n\n")
                
                # Period details
                f.write("Period Details:\n")
                f.write("==============\n")
                for period in period_results:
                    f.write(f"* {period['period']}: {period['eth_events']} ETH events, {period['coin_count']} coins analyzed\n")
                
                f.write("\n")
                
                # Top performers
                top_n = min(20, len(results_df))
                f.write(f"Top {top_n} Most Stable Coins:\n")
                f.write("=======================\n")
                
                for i in range(top_n):
                    row = results_df.iloc[i]
                    f.write(f"{i+1}. {row['symbol']}: Score={row['weighted_score']:.4f}, Mean Performance={row['mean_performance']:.4f}, StdDev={row['performance_std']:.4f}, Coverage={row['period_coverage']:.2f}\n")
            
            self.logger.info(f"Stability report generated at {report_path}")
            
            # Generate performance distribution plots for top coins
            self._plot_performance_distributions(results_df.head(10)['symbol'].tolist(), period_results)
            
        except Exception as e:
            self.logger.error(f"Error generating stability report: {e}")
    
    def _plot_performance_distributions(self, top_symbols: List[str], period_results: List[Dict]):
        """
        Plot performance distributions for top coins
        
        Args:
            top_symbols: List of top symbols to plot
            period_results: Period analysis results
        """
        try:
            # Set plot style
            plt.style.use('dark_background')
            sns.set(style="darkgrid")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot title and labels
            ax.set_title("Performance Distribution of Top Stable Coins", fontsize=16)
            ax.set_xlabel("Performance (%)", fontsize=12)
            ax.set_ylabel("Symbols", fontsize=12)
            
            # Plot period markers
            period_count = len(period_results)
            for i, period in enumerate(period_results):
                ax.axvline(x=i/period_count, color='grey', linestyle='--', alpha=0.5)
                ax.text(i/period_count, 1.05, f"Period {i+1}", rotation=45, transform=ax.get_xaxis_transform())
            
            # Save plot
            plot_path = os.path.join(self.results_dir, f"stability_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Performance distribution plot generated at {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting performance distributions: {e}")
    
    def compare_performance_distributions(self, 
                                        symbols: List[str], 
                                        num_periods: int = 4, 
                                        period_length: int = 90, 
                                        timeframe: str = '1h'):
        """
        Compare performance distributions across time periods
        
        Args:
            symbols: List of symbols to compare
            num_periods: Number of time periods
            period_length: Length of each period in days
            timeframe: Time interval
        """
        self.logger.info(f"Comparing performance distributions for {len(symbols)} symbols across {num_periods} periods")
        
        # Get performance data for each period
        period_performances = []
        period_labels = []
        
        # Analyze each time period
        for i in range(num_periods):
            # Calculate time period
            end_date = (datetime.now() - timedelta(days=i*period_length)).strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=(i+1)*period_length)).strftime('%Y-%m-%d')
            
            period_labels.append(f"Period {i+1}: {start_date} to {end_date}")
            
            try:
                # Identify ETH events
                eth_events = self._identify_eth_events(
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                )
                
                if eth_events:
                    # Analyze performance
                    performance = self._analyze_period_performance(
                        symbols=symbols,
                        eth_events=eth_events,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=timeframe
                    )
                    
                    period_performances.append(performance)
            except Exception as e:
                self.logger.error(f"Error analyzing period {start_date} to {end_date}: {e}")
        
        # Generate comparison plots
        if period_performances:
            self._plot_period_comparison(symbols, period_performances, period_labels)
        else:
            self.logger.warning("No performance data to compare")
    
    def _plot_period_comparison(self, symbols: List[str], period_performances: List[Dict], period_labels: List[str]):
        """
        Plot performance comparison across periods
        
        Args:
            symbols: List of symbols to plot
            period_performances: List of performance dictionaries
            period_labels: List of period labels
        """
        try:
            # Set plot style
            plt.style.use('dark_background')
            sns.set(style="darkgrid")
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(14, 16))
            
            # Data for plots
            symbol_data = {}
            all_performances = []
            
            # Collect performance data for each symbol
            for symbol in symbols:
                perf_values = []
                for period_perf in period_performances:
                    if symbol in period_perf:
                        perf_values.append(period_perf[symbol])
                    else:
                        perf_values.append(np.nan)
                
                symbol_data[symbol] = perf_values
                all_performances.extend([p for p in perf_values if not np.isnan(p)])
            
            # Plot 1: Performance heatmap
            heatmap_data = pd.DataFrame(symbol_data, index=period_labels).T
            
            sns.heatmap(heatmap_data, cmap='coolwarm', center=0, 
                      vmin=np.nanpercentile(all_performances, 10),
                      vmax=np.nanpercentile(all_performances, 90),
                      ax=axes[0], annot=True, fmt=".2f")
            
            axes[0].set_title("Performance Heatmap Across Time Periods", fontsize=16)
            axes[0].set_ylabel("Symbols", fontsize=12)
            
            # Plot 2: Performance consistency
            consistency_data = []
            
            for symbol in symbols:
                performances = [p for p in symbol_data[symbol] if not np.isnan(p)]
                if performances:
                    mean_perf = np.mean(performances)
                    std_perf = np.std(performances)
                    consistency_data.append({
                        'symbol': symbol,
                        'mean_performance': mean_perf,
                        'std_dev': std_perf,
                        'period_count': len(performances)
                    })
            
            consistency_df = pd.DataFrame(consistency_data)
            
            if not consistency_df.empty:
                # Sort by mean performance
                consistency_df = consistency_df.sort_values('mean_performance', ascending=False)
                
                # Plot mean and std error bars
                axes[1].errorbar(
                    x=consistency_df['symbol'],
                    y=consistency_df['mean_performance'],
                    yerr=consistency_df['std_dev'],
                    fmt='o',
                    capsize=5,
                    ecolor='#888888',
                    markersize=8
                )
                
                # Plot zero line
                axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                
                axes[1].set_title("Performance Consistency Across Time Periods", fontsize=16)
                axes[1].set_xlabel("Symbols", fontsize=12)
                axes[1].set_ylabel("Mean Performance (with Std Dev)", fontsize=12)
                axes[1].tick_params(axis='x', rotation=45)
                
                # Add period count annotation
                for i, row in consistency_df.iterrows():
                    axes[1].text(
                        i, row['mean_performance'] + 1.2 * row['std_dev'],
                        f"{row['period_count']}/{len(period_performances)}",
                        ha='center', va='center', fontsize=8
                    )
            
            # Save plot
            plot_path = os.path.join(self.results_dir, f"period_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Period comparison plot generated at {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting period comparison: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _calculate_period_performance(self, symbols: List[str], eth_events: List[datetime], 
                                      start_date: datetime, end_date: datetime,
                                      pre_event_window: int = 10, post_event_window: int = 30) -> Dict[str, float]:
        """
        Calculate average performance of each coin during ETH drop events in a specific time period
        
        Args:
            symbols: List of cryptocurrency symbols to analyze
            eth_events: List of datetime objects representing ETH drop events
            start_date: Start date of the period
            end_date: End date of the period
            pre_event_window: Number of periods before the event (default: 10)
            post_event_window: Number of periods after the event (default: 30)
            
        Returns:
            Dictionary mapping symbols to their average performance across events
        """
        performances = {}
        
        # 計算天數差
        days_diff = (end_date - start_date).days + 1  # 加1包含結束日期
        
        # Load ETH data for reference
        eth_data = self.data_fetcher.fetch_historical_data(
            symbol='ETHUSDT', 
            interval=self.timeframe,
            days=days_diff,
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Skip if no ETH data available
        if eth_data is None or len(eth_data) == 0:
            self.logger.warning(f"No ETH data available for period {start_date} to {end_date}")
            return {}
        
        for symbol in symbols:
            symbol_performances = []
            
            # Load historical data
            data = self.data_fetcher.fetch_historical_data(
                symbol=symbol+'USDT', 
                interval=self.timeframe,
                days=days_diff,
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Skip if no data available
            if data is None or len(data) == 0:
                continue
                
            # 處理可能的重複索引問題
            if not data.index.is_unique:
                self.logger.warning(f"Duplicate timestamps found for {symbol}, fixing by keeping last values")
                # 保留最後一個出現的值（通常是最新的）
                data = data.loc[~data.index.duplicated(keep='last')]
                
            for event_time in eth_events:
                
                try:
                    # Find pre-event reference price
                    pre_event_price = None
                    closest_idx = data.index.get_indexer([event_time], method='nearest')[0]
                    
                    if closest_idx >= 0 and closest_idx < len(data):
                        # Look back to find pre-event price
                        pre_event_idx = max(0, closest_idx - pre_event_window)
                        pre_event_price = data.iloc[pre_event_idx]['close']
                        
                        # Look forward to find post-event price
                        post_event_idx = min(len(data) - 1, closest_idx + post_event_window)
                        post_event_price = data.iloc[post_event_idx]['close']
                        
                        # Calculate performance
                        if pre_event_price is not None and pre_event_price > 0:
                            performance = (post_event_price / pre_event_price) - 1
                            symbol_performances.append(performance)
                
                except Exception as e:
                    self.logger.warning(f"Error calculating performance for {symbol} at {event_time}: {e}")
            
            # Average performance across all events in this period
            if symbol_performances:
                performances[symbol] = np.mean(symbol_performances)
        
        self.logger.info(f"Calculated performance for {len(performances)} coins in period {start_date} to {end_date}")
        return performances 