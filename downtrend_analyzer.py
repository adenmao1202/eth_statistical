#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Downtrend Analyzer Module
Responsible for analyzing cryptocurrency performance during ETH downtrends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import config
from data_fetcher import DataFetcher
from analyzer import CorrelationAnalyzer


class DowntrendAnalyzer:
    """Analyzer for cryptocurrency performance during ETH downtrends"""
    
    def __init__(self, correlation_analyzer: CorrelationAnalyzer):
        """
        Initialize the downtrend analyzer
        
        Args:
            correlation_analyzer (CorrelationAnalyzer): Correlation analyzer instance
        """
        self.correlation_analyzer = correlation_analyzer
        self.data_fetcher = correlation_analyzer.data_fetcher
        self.reference_symbol = correlation_analyzer.reference_symbol
        
        # Cache for analysis results
        self.downtrend_periods = []
        self.stable_coins = None
        self.best_rebounders = None
        self.combined_score = None
    
    def identify_downtrend_periods(self, data: pd.DataFrame, 
                                  drop_threshold: float = config.DEFAULT_DROP_THRESHOLD,
                                  window_size: int = config.DEFAULT_WINDOW_SIZE) -> List[Dict[str, Any]]:
        """
        Identify significant downtrend periods in the reference symbol price data
        
        Args:
            data (pd.DataFrame): Price data for the reference symbol
            drop_threshold (float): Threshold for identifying significant drops (negative value)
            window_size (int): Window size for calculating price changes
            
        Returns:
            List[Dict[str, Any]]: List of downtrend periods with start/end times and metrics
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        print(f"Analyzing {len(data)} data points for downtrends")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Drop threshold: {drop_threshold}, Window size: {window_size}")
        
        # Calculate rolling percentage change
        data['pct_change'] = data['close'].pct_change(window_size)
        
        # Print statistics about percentage changes
        min_pct = data['pct_change'].min()
        max_pct = data['pct_change'].max()
        mean_pct = data['pct_change'].mean()
        std_pct = data['pct_change'].std()
        
        print(f"Percentage change statistics:")
        print(f"  Min: {min_pct:.4f}, Max: {max_pct:.4f}")
        print(f"  Mean: {mean_pct:.4f}, Std: {std_pct:.4f}")
        
        # Count potential downtrend points
        potential_downtrends = (data['pct_change'] <= drop_threshold).sum()
        print(f"Found {potential_downtrends} potential downtrend points (pct_change <= {drop_threshold})")
        
        # Identify potential downtrend periods
        downtrend_periods = []
        in_downtrend = False
        start_idx = None
        
        for idx, row in data.iterrows():
            if pd.isna(row['pct_change']):
                continue
                
            if not in_downtrend and row['pct_change'] <= drop_threshold:
                # Start of downtrend
                in_downtrend = True
                start_idx = idx
                # Find the price window_size periods ago
                window_ago_idx = idx - pd.Timedelta(minutes=window_size)
                start_price = data.loc[window_ago_idx]['close'] if window_ago_idx in data.index else data.iloc[0]['close']
            elif in_downtrend and row['pct_change'] > drop_threshold:
                # End of downtrend
                in_downtrend = False
                end_idx = idx
                
                # Calculate metrics for this period
                period_data = data.loc[start_idx:end_idx]
                lowest_price = period_data['close'].min()
                lowest_idx = period_data['close'].idxmin()
                
                # Calculate drop percentage
                drop_pct = ((lowest_price / start_price) - 1) * 100
                
                # Add period to list
                downtrend_periods.append({
                    'start': start_idx,
                    'end': end_idx,
                    'start_price': start_price,
                    'end_price': data.loc[end_idx]['close'],
                    'lowest_price': lowest_price,
                    'lowest_time': lowest_idx,
                    'drop_pct': drop_pct,
                    'duration_minutes': (end_idx - start_idx).total_seconds() / 60
                })
        
        # Check if we're still in a downtrend at the end of the data
        if in_downtrend:
            end_idx = data.index[-1]
            period_data = data.loc[start_idx:end_idx]
            lowest_price = period_data['close'].min()
            lowest_idx = period_data['close'].idxmin()
            
            # Calculate drop percentage
            drop_pct = ((lowest_price / start_price) - 1) * 100
            
            # Add period to list
            downtrend_periods.append({
                'start': start_idx,
                'end': end_idx,
                'start_price': start_price,
                'end_price': data.loc[end_idx]['close'],
                'lowest_price': lowest_price,
                'lowest_time': lowest_idx,
                'drop_pct': drop_pct,
                'duration_minutes': (end_idx - start_idx).total_seconds() / 60
            })
        
        print(f"Identified {len(downtrend_periods)} downtrend periods")
        
        # Print details of each downtrend period
        for i, period in enumerate(downtrend_periods):
            print(f"Downtrend {i+1}:")
            print(f"  Start: {period['start']}, End: {period['end']}")
            print(f"  Duration: {period['duration_minutes']:.1f} minutes")
            print(f"  Drop: {period['drop_pct']:.2f}%")
        
        # Sort by drop percentage (most severe first)
        downtrend_periods.sort(key=lambda x: x['drop_pct'])
        
        # Cache the result
        self.downtrend_periods = downtrend_periods
        
        return downtrend_periods
    
    def find_stable_coins(self, data_dict: Dict[str, pd.DataFrame], 
                         downtrend_periods: List[Dict[str, Any]], 
                         top_n: int = config.DEFAULT_TOP_N) -> pd.Series:
        """
        Find coins that remain stable during ETH downtrends
        
        Args:
            data_dict (dict): Dictionary with symbols as keys and DataFrames as values
            downtrend_periods (list): List of downtrend periods
            top_n (int): Number of top stable coins to return
            
        Returns:
            pd.Series: Series of stability scores with symbols as index
        """
        if not downtrend_periods:
            print("No downtrend periods provided, returning empty result")
            return pd.Series(dtype=float)
            
        # Initialize dictionary to store stability scores
        stability_scores = {}
        
        # Iterate through all symbols
        for symbol, df in data_dict.items():
            if symbol == self.reference_symbol:
                continue
                
            # Skip symbols with insufficient data
            if df.empty or len(df) < 10:
                continue
                
            # Calculate stability score for each downtrend period
            period_scores = []
            
            for period in downtrend_periods:
                start = period['start']
                end = period['end']
                
                # Get data for this period
                try:
                    period_data = df.loc[start:end]
                    
                    # Skip if insufficient data
                    if len(period_data) < 2:
                        continue
                        
                    # Calculate percentage change during this period
                    start_price = period_data.iloc[0]['close']
                    end_price = period_data.iloc[-1]['close']
                    pct_change = ((end_price / start_price) - 1) * 100
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility = period_data['close'].pct_change().std() * 100
                    
                    # Calculate stability score (higher is better)
                    # We want coins that go up or stay flat during ETH downtrends
                    # and have low volatility
                    stability = pct_change - (volatility * 0.5)
                    
                    period_scores.append(stability)
                except Exception as e:
                    print(f"Error calculating stability for {symbol}: {e}")
                    continue
            
            # Calculate average stability score across all periods
            if period_scores:
                avg_stability = sum(period_scores) / len(period_scores)
                stability_scores[symbol] = avg_stability
        
        # Convert to Series and sort
        stability_series = pd.Series(stability_scores)
        stability_series = stability_series.sort_values(ascending=False)
        
        # Store for later use
        self.stable_coins = stability_series
        
        # Return top N
        return stability_series.head(top_n)
    
    def find_best_rebounders(self, data_dict: Dict[str, pd.DataFrame], 
                             downtrend_periods: List[Dict[str, Any]], 
                             window_size: int = config.DEFAULT_WINDOW_SIZE,
                             top_n: int = config.DEFAULT_TOP_N) -> pd.Series:
        """
        Find coins that rebound best after ETH downtrends
        
        Args:
            data_dict (dict): Dictionary with symbols as keys and DataFrames as values
            downtrend_periods (list): List of downtrend periods
            window_size (int): Window size for calculating rebound
            top_n (int): Number of top rebounders to return
            
        Returns:
            pd.Series: Series of rebound scores with symbols as index
        """
        if not downtrend_periods:
            print("No downtrend periods provided, returning empty result")
            return pd.Series(dtype=float)
            
        # Initialize dictionary to store rebound scores
        rebound_scores = {}
        
        # Iterate through all symbols
        for symbol, df in data_dict.items():
            if symbol == self.reference_symbol:
                continue
                
            # Skip symbols with insufficient data
            if df.empty or len(df) < 10:
                continue
                
            # Calculate rebound score for each downtrend period
            period_scores = []
            
            for period in downtrend_periods:
                end_time = period['end']
                
                # Calculate rebound window end time
                rebound_end = end_time + pd.Timedelta(minutes=window_size)
                
                # Get data for rebound period
                try:
                    if end_time in df.index and rebound_end in df.index:
                        end_price = df.loc[end_time]['close']
                        rebound_price = df.loc[rebound_end]['close']
                        
                        # Calculate rebound percentage
                        rebound_pct = ((rebound_price / end_price) - 1) * 100
                        period_scores.append(rebound_pct)
                except Exception as e:
                    print(f"Error calculating rebound for {symbol}: {e}")
                    continue
            
            # Calculate average rebound score across all periods
            if period_scores:
                avg_rebound = sum(period_scores) / len(period_scores)
                rebound_scores[symbol] = avg_rebound
        
        # Convert to Series and sort
        rebound_series = pd.Series(rebound_scores)
        rebound_series = rebound_series.sort_values(ascending=False)
        
        # Store for later use
        self.best_rebounders = rebound_series
        
        # Return top N
        return rebound_series.head(top_n)
    
    def calculate_combined_score(self, stable_coins: pd.Series, rebounders: pd.Series, 
                               stability_weight: float = 0.5, top_n: int = config.DEFAULT_TOP_N) -> pd.Series:
        """
        Calculate combined score based on stability during downtrends and rebound after downtrends
        
        Args:
            stable_coins (pd.Series): Series of stability scores
            rebounders (pd.Series): Series of rebound scores
            stability_weight (float): Weight for stability score (0-1)
            top_n (int): Number of top coins to return
            
        Returns:
            pd.Series: Series of combined scores with symbols as index
        """
        if stable_coins.empty or rebounders.empty:
            print("Empty stability or rebound scores, returning empty result")
            return pd.Series(dtype=float)
            
        # Get all symbols
        all_symbols = set(stable_coins.index) | set(rebounders.index)
        
        # Initialize combined scores
        combined_scores = {}
        
        for symbol in all_symbols:
            # Get scores (default to 0 if not present)
            stability = stable_coins.get(symbol, 0)
            rebound = rebounders.get(symbol, 0)
            
            # Normalize scores to 0-1 range
            norm_stability = (stability - stable_coins.min()) / (stable_coins.max() - stable_coins.min()) if len(stable_coins) > 1 else 0.5
            norm_rebound = (rebound - rebounders.min()) / (rebounders.max() - rebounders.min()) if len(rebounders) > 1 else 0.5
            
            # Calculate weighted score
            combined = (stability_weight * norm_stability) + ((1 - stability_weight) * norm_rebound)
            combined_scores[symbol] = combined
        
        # Convert to Series and sort
        combined_series = pd.Series(combined_scores)
        combined_series = combined_series.sort_values(ascending=False)
        
        # Store for later use
        self.combined_score = combined_series
        
        # Return top N
        return combined_series.head(top_n)
    
    def visualize_downtrends(self, data: pd.DataFrame, downtrend_periods: List[Dict[str, Any]]) -> None:
        """
        Visualize identified downtrend periods
        
        Args:
            data (pd.DataFrame): Price data for the reference symbol
            downtrend_periods (list): List of downtrend periods
        """
        plt.figure(figsize=(15, 8))
        
        # Plot price
        plt.plot(data.index, data['close'], label=self.reference_symbol, color='blue')
        
        # Highlight downtrend periods
        for i, period in enumerate(downtrend_periods):
            start = period['start']
            end = period['end']
            lowest = period['lowest_time']
            
            # Shade downtrend area
            plt.axvspan(start, end, alpha=0.2, color='red')
            
            # Mark lowest point
            plt.scatter(lowest, period['lowest_price'], color='red', s=100, zorder=5, marker='v')
            
            # Add text annotation
            plt.annotate(f"Drop {i+1}: {period['drop_pct']:.1f}%", 
                        xy=(lowest, period['lowest_price']),
                        xytext=(10, -30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.title(f'{self.reference_symbol} Price with Identified Downtrends', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('eth_downtrends.png', dpi=300)
        plt.show()
    
    def visualize_stable_coins(self, data_dict: Dict[str, pd.DataFrame], 
                              stable_coins: pd.Series, 
                              downtrend_periods: List[Dict[str, Any]],
                              top_n: int = 5) -> None:
        """
        Visualize performance of stable coins during downtrends
        
        Args:
            data_dict (dict): Dictionary with symbols as keys and DataFrames as values
            stable_coins (pd.Series): Series of stability scores
            downtrend_periods (list): List of downtrend periods
            top_n (int): Number of top stable coins to display
        """
        # Get top stable coins
        top_stable = stable_coins.head(top_n)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Normalize reference symbol price
        ref_data = data_dict[self.reference_symbol].copy()
        ref_data['normalized'] = ref_data['close'] / ref_data['close'].iloc[0] * 100
        
        # Plot reference symbol
        plt.plot(ref_data.index, ref_data['normalized'], label=f"{self.reference_symbol} (Reference)", linewidth=2, color='black')
        
        # Plot each stable coin
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_stable)))
        for i, (symbol, score) in enumerate(top_stable.items()):
            if symbol in data_dict:
                coin_data = data_dict[symbol].copy()
                coin_data['normalized'] = coin_data['close'] / coin_data['close'].iloc[0] * 100
                plt.plot(coin_data.index, coin_data['normalized'], 
                        label=f"{symbol} ({score:.2f}%)", 
                        linewidth=1.5, color=colors[i])
        
        # Highlight downtrend periods
        for i, period in enumerate(downtrend_periods):
            plt.axvspan(period['start'], period['end'], alpha=0.2, color='red')
            
            # Add text annotation
            mid_point = period['start'] + (period['end'] - period['start']) / 2
            y_pos = ref_data['normalized'].max() * (0.95 - 0.05 * i)
            plt.annotate(f"Drop {i+1}: {period['drop_pct']:.1f}%", 
                        xy=(mid_point, y_pos),
                        xytext=(mid_point, y_pos),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                        ha='center')
        
        plt.title(f'Top {top_n} Stable Coins During {self.reference_symbol} Downtrends', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig('eth_stable_coins.png', dpi=300)
        plt.show()
    
    def visualize_best_rebounders(self, data_dict: Dict[str, pd.DataFrame], 
                                 rebounders: pd.Series, 
                                 downtrend_periods: List[Dict[str, Any]],
                                 top_n: int = 5) -> None:
        """
        Visualize performance of best rebounding coins after downtrends
        
        Args:
            data_dict (dict): Dictionary with symbols as keys and DataFrames as values
            rebounders (pd.Series): Series of rebound scores
            downtrend_periods (list): List of downtrend periods
            top_n (int): Number of top rebounders to display
        """
        # Get top rebounders
        top_rebounders = rebounders.head(top_n)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Normalize reference symbol price
        ref_data = data_dict[self.reference_symbol].copy()
        ref_data['normalized'] = ref_data['close'] / ref_data['close'].iloc[0] * 100
        
        # Plot reference symbol
        plt.plot(ref_data.index, ref_data['normalized'], label=f"{self.reference_symbol} (Reference)", linewidth=2, color='black')
        
        # Plot each rebounder
        colors = plt.cm.plasma(np.linspace(0, 1, len(top_rebounders)))
        for i, (symbol, score) in enumerate(top_rebounders.items()):
            if symbol in data_dict:
                coin_data = data_dict[symbol].copy()
                coin_data['normalized'] = coin_data['close'] / coin_data['close'].iloc[0] * 100
                plt.plot(coin_data.index, coin_data['normalized'], 
                        label=f"{symbol} ({score:.2f}%)", 
                        linewidth=1.5, color=colors[i])
        
        # Highlight downtrend periods and rebound windows
        for i, period in enumerate(downtrend_periods):
            # Highlight downtrend
            plt.axvspan(period['start'], period['end'], alpha=0.2, color='red')
            
            # Highlight rebound window
            rebound_end = period['end'] + timedelta(minutes=config.DEFAULT_WINDOW_SIZE)
            plt.axvspan(period['end'], rebound_end, alpha=0.2, color='green')
            
            # Add text annotation
            mid_point = period['start'] + (period['end'] - period['start']) / 2
            y_pos = ref_data['normalized'].max() * (0.95 - 0.05 * i)
            plt.annotate(f"Drop {i+1}: {period['drop_pct']:.1f}%", 
                        xy=(mid_point, y_pos),
                        xytext=(mid_point, y_pos),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                        ha='center')
        
        plt.title(f'Top {top_n} Rebounding Coins After {self.reference_symbol} Downtrends', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig('eth_rebound_best_coins.png', dpi=300)
        plt.show()
    
    def analyze_downtrends(self, timeframe: str, start_date: str, end_date: Optional[str] = None,
                          drop_threshold: float = config.DEFAULT_DROP_THRESHOLD,
                          window_size: int = config.DEFAULT_WINDOW_SIZE,
                          top_n: int = config.DEFAULT_TOP_N,
                          use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze cryptocurrency performance during ETH downtrends
        
        Args:
            timeframe (str): Kline timeframe (e.g., '1m', '1h')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            drop_threshold (float): Threshold for identifying significant drops (negative value)
            window_size (int): Window size for calculating price changes
            top_n (int): Number of top results to display
            use_cache (bool): Whether to use cached data
            
        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        print(f"Analyzing {self.reference_symbol} downtrends on {timeframe} timeframe")
        
        # Get data for all symbols
        all_symbols = self.correlation_analyzer.data_fetcher.get_all_futures_symbols()
        usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        if self.reference_symbol not in usdt_symbols:
            usdt_symbols.append(self.reference_symbol)
        
        # Prioritize reference symbol
        if self.reference_symbol in usdt_symbols:
            usdt_symbols.remove(self.reference_symbol)
            prioritized_symbols = [self.reference_symbol] + usdt_symbols
        else:
            prioritized_symbols = usdt_symbols
        
        # Get data for all symbols
        data_dict = self.correlation_analyzer.data_fetcher.fetch_data_for_all_symbols(
            prioritized_symbols, timeframe, start_date, end_date, use_cache
        )
        
        # Check if we have reference symbol data
        if self.reference_symbol not in data_dict or data_dict[self.reference_symbol].empty:
            raise ValueError(f"No {self.reference_symbol} data available for {timeframe}")
        
        # Identify downtrend periods
        ref_data = data_dict[self.reference_symbol]
        downtrend_periods = self.identify_downtrend_periods(
            ref_data, drop_threshold, window_size
        )
        
        print(f"Identified {len(downtrend_periods)} significant downtrend periods")
        
        # If no downtrend periods found, return empty results
        if not downtrend_periods:
            return {
                'downtrend_periods': [],
                'stable_coins': pd.Series(),
                'best_rebounders': pd.Series(),
                'combined_score': pd.Series(),
                'scatter_data': pd.DataFrame(),
                'data_dict': data_dict
            }
        
        # Find stable coins during downtrends
        stable_coins = self.find_stable_coins(data_dict, downtrend_periods, top_n)
        
        # Find best rebounders after downtrends
        rebounders = self.find_best_rebounders(data_dict, downtrend_periods, window_size, top_n)
        
        # Calculate combined score
        combined = self.calculate_combined_score(self.stable_coins, self.best_rebounders, 0.5, top_n)
        
        # Visualize results
        self.visualize_downtrends(ref_data, downtrend_periods)
        self.visualize_stable_coins(data_dict, stable_coins, downtrend_periods)
        self.visualize_best_rebounders(data_dict, rebounders, downtrend_periods)
        
        # Create scatter plot
        scatter_data = self.correlation_analyzer.create_scatter_plot(
            self.stable_coins, self.best_rebounders, top_n
        )
        
        # Return results
        return {
            'downtrend_periods': downtrend_periods,
            'stable_coins': stable_coins,
            'best_rebounders': rebounders,
            'combined_score': combined,
            'scatter_data': scatter_data,
            'data_dict': data_dict
        } 