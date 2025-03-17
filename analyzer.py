#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis Module
Responsible for cryptocurrency correlation analysis and visualization
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

import config
from data_fetcher import DataFetcher


class CorrelationAnalyzer:
    """Correlation analyzer responsible for analyzing price correlations between cryptocurrencies"""
    
    def __init__(self, data_fetcher: DataFetcher, reference_symbol: str = config.DEFAULT_REFERENCE_SYMBOL):
        """
        Initialize the correlation analyzer
        
        Args:
            data_fetcher (DataFetcher): Data fetcher instance
            reference_symbol (str, optional): Reference trading pair symbol
        """
        self.data_fetcher = data_fetcher
        self.reference_symbol = reference_symbol
        
        # Cache last analysis results
        self._last_stable_coins = None
        self._last_rebounders = None
        self._last_combined = None
    
    @staticmethod
    def _calculate_correlation_numpy(x: np.ndarray, y: np.ndarray) -> float:
        """Optimized correlation calculation using numpy"""
        # Direct numpy implementation of Pearson correlation
        x_norm = x - np.mean(x)
        y_norm = y - np.mean(y)
        return np.sum(x_norm * y_norm) / (np.sqrt(np.sum(x_norm**2)) * np.sqrt(np.sum(y_norm**2)))

    def calculate_correlation(self, data_dict: Dict[str, pd.DataFrame], 
                             reference_symbol: Optional[str] = None) -> pd.Series:
        """Vectorized correlation calculation using numpy arrays"""
        if reference_symbol is None:
            reference_symbol = self.reference_symbol
            
        if reference_symbol not in data_dict:
            raise ValueError(f"Reference symbol {reference_symbol} not found in data")

        # Convert all closing prices to numpy arrays for faster calculation
        ref_prices = data_dict[reference_symbol]['close'].values
        correlations = {}

        # Use more efficient method to calculate correlations in parallel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 16)) as executor:
            # Create list of tuples (symbol, prices) for parallel processing
            symbol_prices = []
            for symbol, df in data_dict.items():
                prices = df['close'].values
                if len(prices) == len(ref_prices):
                    symbol_prices.append((symbol, prices))
            
            # Submit all correlation calculations in batch
            futures_to_symbols = {}
            for symbol, prices in symbol_prices:
                future = executor.submit(self._calculate_correlation_numpy, ref_prices, prices)
                futures_to_symbols[future] = symbol

            # Process results
            for future in as_completed(futures_to_symbols):
                symbol = futures_to_symbols[future]
                try:
                    correlations[symbol] = future.result()
                except Exception as e:
                    print(f"Error calculating correlation for {symbol}: {e}")
                    correlations[symbol] = np.nan

        return pd.Series(correlations).sort_values(ascending=False)
    
    def visualize_correlation(self, correlation: pd.Series, timeframe: str, top_n: int = config.DEFAULT_TOP_N) -> None:
        """
        Visualize correlations between reference symbol and top N trading pairs
        
        Args:
            correlation (pd.Series): Series containing correlation values
            timeframe (str): Kline timeframe
            top_n (int, optional): Number of highest correlated trading pairs to display
        """
        # Get top N highest correlated trading pairs (excluding reference symbol itself)
        top_corr = correlation.drop(self.reference_symbol).head(top_n)
        bottom_corr = correlation.drop(self.reference_symbol).tail(top_n)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot highest correlations
        sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax1, palette='viridis')
        ax1.set_title(f'Top {top_n} Positive Correlations with {self.reference_symbol} ({timeframe})')
        ax1.set_xlabel('Correlation')
        ax1.set_ylabel('Trading Pair')
        
        # Plot lowest correlations
        sns.barplot(x=bottom_corr.values, y=bottom_corr.index, ax=ax2, palette='viridis')
        ax2.set_title(f'Top {top_n} Negative Correlations with {self.reference_symbol} ({timeframe})')
        ax2.set_xlabel('Correlation')
        ax2.set_ylabel('Trading Pair')
        
        plt.tight_layout()
        plt.savefig(f'correlation_{timeframe}.png', dpi=300)
        plt.show()
    
    def analyze_all_timeframes(self, start_date: str, end_date: Optional[str] = None, 
                              top_n: int = config.DEFAULT_TOP_N, 
                              use_cache: bool = True) -> Dict[str, pd.Series]:
        """Analyze all timeframes sequentially to avoid asyncio issues"""
        # Get trading pairs only once
        all_symbols = self.data_fetcher.get_all_futures_symbols()
        usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        if self.reference_symbol not in usdt_symbols:
            usdt_symbols.append(self.reference_symbol)
        print(f"Found {len(usdt_symbols)} USDT futures trading pairs")
        
        # Prioritize reference symbol to ensure it's always processed
        if self.reference_symbol in usdt_symbols:
            usdt_symbols.remove(self.reference_symbol)
            prioritized_symbols = [self.reference_symbol] + usdt_symbols
        else:
            prioritized_symbols = usdt_symbols
        
        results = {}
        # Process each timeframe sequentially - safer than nested async loops
        for timeframe_name, timeframe_code in config.TIMEFRAMES.items():
            print(f"\nAnalyzing {timeframe_name} timeframe")
            
            # Reset rate limit counters between timeframes to ensure fresh start
            self.data_fetcher.rate_limiter.weight_used = 0
            self.data_fetcher.rate_limiter.weight_reset_time = datetime.now().timestamp() + 60
            self.data_fetcher.rate_limiter.requests_this_second = 0
            self.data_fetcher.rate_limiter.second_start_time = datetime.now().timestamp()
            
            # Get data for all symbols
            data = self.data_fetcher.fetch_data_for_all_symbols(prioritized_symbols, timeframe_name, start_date, end_date, use_cache)
            
            # Check if we have reference symbol data, which is required for correlation analysis
            if self.reference_symbol not in data or data[self.reference_symbol].empty:
                print(f"Warning: No {self.reference_symbol} data available for {timeframe_name}. Skipping correlation analysis.")
                continue
                
            # Calculate correlations
            correlation = self.calculate_correlation(data, self.reference_symbol)
            results[timeframe_name] = correlation
            
            # Save correlations to CSV
            correlation.to_csv(f'correlation_{timeframe_name}.csv')
            
            # Visualize correlations
            self.visualize_correlation(correlation, timeframe_name, top_n)
            
            # Add delay between timeframes to avoid rate limit issues
            if timeframe_name != list(config.TIMEFRAMES.keys())[-1]:  # If not the last timeframe
                print(f"Completed analysis for {timeframe_name}. Waiting for next timeframe...")
                import time
                time.sleep(2)  # Short delay between timeframes
        
        return results
    
    def create_correlation_heatmap(self, results: Dict[str, pd.Series], top_n: int = config.DEFAULT_TOP_N) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create correlation heatmap across all timeframes
        
        Args:
            results (dict): Dictionary with timeframes as keys and correlation Series as values
            top_n (int, optional): Number of highest correlated trading pairs to display
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing positive and negative correlation heatmap data
        """
        # Get top N highest correlated trading pairs across all timeframes
        all_corr = pd.DataFrame(results)
        
        # Sort by average correlation
        all_corr['avg'] = all_corr.mean(axis=1)
        
        # Get highest positively correlated symbols
        top_symbols = all_corr.sort_values('avg', ascending=False).drop(self.reference_symbol).head(top_n).index.tolist()
        
        # Get highest negatively correlated symbols
        bottom_symbols = all_corr.sort_values('avg', ascending=True).drop(self.reference_symbol).head(top_n).index.tolist()
        
        # Create positive correlation heatmap data
        heatmap_data_positive = all_corr.loc[top_symbols].drop(columns=['avg'])
        
        # Create negative correlation heatmap data
        heatmap_data_negative = all_corr.loc[bottom_symbols].drop(columns=['avg'])
        
        # Create positive correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data_positive, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f'Top {top_n} Positive Correlations with {self.reference_symbol} (Across Timeframes)')
        plt.tight_layout()
        plt.savefig('correlation_heatmap_positive.png', dpi=300)
        plt.show()
        
        # Create negative correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data_negative, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Top {top_n} Negative Correlations with {self.reference_symbol} (Across Timeframes)')
        plt.tight_layout()
        plt.savefig('correlation_heatmap_negative.png', dpi=300)
        plt.show()
        
        # Create combined heatmap (original functionality)
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data_positive, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f'Top {top_n} Correlations with {self.reference_symbol} (Across Timeframes)')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300)
        plt.show()
        
        return heatmap_data_positive, heatmap_data_negative

    def visualize_price_movements(self, data_dict: Dict[str, pd.DataFrame], 
                                 downtrend_periods: Optional[List[Dict[str, Any]]] = None, 
                                 timeframe: str = '1h', top_n: int = 5) -> None:
        """
        Visualize price movements of ETH and top stable coins during downtrends
        
        Args:
            data_dict (dict): Dictionary with symbols as keys and DataFrames as values
            downtrend_periods (list, optional): List of downtrend periods
            timeframe (str, optional): Timeframe for visualization
            top_n (int, optional): Number of top stable coins to display
        """
        if self.reference_symbol not in data_dict:
            print(f"Error: {self.reference_symbol} data not found")
            return
            
        # Normalize all prices as percentage change from first point
        ref_data = data_dict[self.reference_symbol].copy()
        ref_data['normalized'] = ref_data['close'] / ref_data['close'].iloc[0] * 100
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot reference symbol price
        plt.plot(ref_data.index, ref_data['normalized'], label=self.reference_symbol, linewidth=2, color='black')
        
        # If we have stable coins available, get the top ones
        if hasattr(self, '_last_stable_coins') and self._last_stable_coins is not None:
            stable_coins = self._last_stable_coins.head(top_n).index.tolist()
            
            # Plot each stable coin
            colors = plt.cm.viridis(np.linspace(0, 1, len(stable_coins)))
            for i, symbol in enumerate(stable_coins):
                if symbol in data_dict:
                    coin_data = data_dict[symbol].copy()
                    coin_data['normalized'] = coin_data['close'] / coin_data['close'].iloc[0] * 100
                    plt.plot(coin_data.index, coin_data['normalized'], label=symbol, linewidth=1.5, color=colors[i])
        
        # Mark downtrend periods if provided
        if downtrend_periods:
            for i, period in enumerate(downtrend_periods):
                start = period['start']
                end = period['end']
                lowest_point = ref_data[ref_data['close'] == period['lowest_price']].index[0]
                
                # Add shading for downtrend area
                plt.axvspan(start, end, alpha=0.2, color='red')
                
                # Mark lowest point
                plt.scatter(lowest_point, ref_data.loc[lowest_point, 'normalized'], 
                           color='red', s=100, zorder=5, marker='v')
                
                # Add text annotation for the period
                mid_point = start + (end - start) / 2
                y_pos = ref_data['normalized'].max() * (0.95 - 0.05 * i)
                plt.annotate(f"Period {i+1}: {period['drop_pct']:.1f}%", 
                            xy=(mid_point, y_pos),
                            xytext=(mid_point, y_pos),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                            ha='center')
        
        plt.title(f'Price Movement Comparison ({timeframe} Timeframe)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f'price_movements_{timeframe}.png', dpi=300)
        plt.show()
        
    def create_scatter_plot(self, stable_coins: pd.Series, best_rebounders: pd.Series, 
                           top_n: int = config.DEFAULT_TOP_N) -> pd.DataFrame:
        """
        Create scatter plot of downtrend performance vs rebound performance
        
        Args:
            stable_coins (pd.Series): Series containing downtrend performance values
            best_rebounders (pd.Series): Series containing rebound performance values
            top_n (int, optional): Number of top coins to highlight
            
        Returns:
            pd.DataFrame: DataFrame for scatter plot
        """
        # Create DataFrame for scatter plot
        scatter_data = pd.DataFrame({
            'Downtrend Performance (%)': stable_coins,
            'Rebound Performance (%)': best_rebounders
        })
        
        # Calculate combined score
        scatter_data['Combined Score'] = scatter_data['Downtrend Performance (%)'] + scatter_data['Rebound Performance (%)']
        
        # Sort by combined score
        scatter_data = scatter_data.sort_values('Combined Score', ascending=False)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot all points
        plt.scatter(scatter_data['Downtrend Performance (%)'], 
                   scatter_data['Rebound Performance (%)'], 
                   alpha=0.5, s=50, color='gray')
        
        # Highlight top N coins
        top_coins = scatter_data.head(top_n)
        plt.scatter(top_coins['Downtrend Performance (%)'], 
                   top_coins['Rebound Performance (%)'], 
                   alpha=1.0, s=100, color='green')
        
        # Add labels for top coins
        for idx, row in top_coins.iterrows():
            plt.annotate(idx, 
                        (row['Downtrend Performance (%)'], row['Rebound Performance (%)']),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
        
        # Add quadrant lines
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add quadrant labels
        plt.annotate('Best Combined\n(Stable + Strong Rebound)', xy=(5, 5), xycoords='data',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.8))
        
        plt.annotate('Good Rebound but\nDrops During Downtrend', xy=(-5, 5), xycoords='data',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        plt.annotate('Stable but Poor Rebound', xy=(5, -5), xycoords='data',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.8))
        
        plt.annotate('Poor Combined Performance\n(Drops + Weak Rebound)', xy=(-5, -5), xycoords='data',
                    bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8))
        
        plt.title('Cryptocurrency Performance: Downtrend vs Rebound', fontsize=16)
        plt.xlabel('Downtrend Performance (%)', fontsize=12)
        plt.ylabel('Rebound Performance (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('performance_scatter_plot.png', dpi=300)
        plt.show()
        
        return scatter_data 