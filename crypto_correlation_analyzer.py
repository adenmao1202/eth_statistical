#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryptocurrency Correlation Analyzer
This script analyzes the price correlation between ETH and other cryptocurrencies
using Binance historical data across different timeframes (1min, 5min, 15min, 1h, 4h).
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from binance.client import Client
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import asyncio
from collections import defaultdict

warnings.filterwarnings('ignore')

class CryptoCorrelationAnalyzer:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, max_klines: int = 200):
        """
        Initialize the Crypto Correlation Analyzer with optimized settings.
        
        Args:
            api_key (str, optional): Binance API key. Defaults to None.
            api_secret (str, optional): Binance API secret. Defaults to None.
            max_klines (int, optional): Maximum number of K-lines to fetch. Defaults to 200.
                Should be between 100-300 for optimal performance.
        """
        self.client = Client(api_key, api_secret)
        self.timeframes = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR
        }
        self.eth_symbol = 'ETHUSDT'
        self.data_dir = 'historical_data'
        self.max_klines = max(100, min(300, max_klines))  # Ensure max_klines is between 100-300
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize cache for data operations
        self._symbol_cache = {}
        self._correlation_cache = {}
        
        # Configure numpy for optimal performance
        np.set_printoptions(precision=8, suppress=True)
        
        # Set up thread pool for parallel operations
        self._max_workers = os.cpu_count()
        
        # Configure pandas for better performance
        pd.set_option('compute.use_numexpr', True)
        
    def get_all_futures_symbols(self):
        """
        Get all available futures symbols from Binance.
        
        Returns:
            list: List of futures symbols.
        """
        exchange_info = self.client.futures_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        return symbols
    
    def download_historical_data(self, symbol, timeframe, start_date, end_date=None):
        """
        Download historical kline data for a symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            timeframe (str): Timeframe for the klines (e.g., '1m', '1h').
            start_date (str): Start date in format 'YYYY-MM-DD'.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to current date.
            
        Returns:
            pandas.DataFrame: DataFrame with historical price data.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate appropriate date range based on max_klines and timeframe
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Adjust start date based on timeframe and max_klines
        # This is a rough estimation to get approximately max_klines candles
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240
        }
        
        # Calculate minutes needed for max_klines
        minutes_needed = self.max_klines * timeframe_minutes[timeframe]
        
        # Calculate adjusted start date
        adjusted_start_dt = end_dt - timedelta(minutes=minutes_needed)
        original_start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Use the later of the two dates (original or adjusted)
        start_dt = max(original_start_dt, adjusted_start_dt)
        adjusted_start_date = start_dt.strftime('%Y-%m-%d')
        
        print(f"Fetching ~{self.max_klines} candles for {symbol} at {timeframe} timeframe")
        print(f"Adjusted date range: {adjusted_start_date} to {end_date}")
            
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        # Get the data from Binance with limit parameter
        klines = self.client.futures_historical_klines(
            symbol=symbol,
            interval=self.timeframes[timeframe],
            start_str=start_ts,
            end_str=end_ts,
            limit=self.max_klines  # Limit the number of candles
        )
        
        # Create a DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def save_data_to_csv(self, df, symbol, timeframe):
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pandas.DataFrame): DataFrame to save.
            symbol (str): Trading pair symbol.
            timeframe (str): Timeframe for the klines.
        """
        filename = f"{self.data_dir}/{symbol}_{timeframe}.csv"
        df.to_csv(filename)
        print(f"Saved {symbol} {timeframe} data to {filename}")
    
    def load_data_from_csv(self, symbol, timeframe):
        """
        Load DataFrame from CSV file.
        
        Args:
            symbol (str): Trading pair symbol.
            timeframe (str): Timeframe for the klines.
            
        Returns:
            pandas.DataFrame: DataFrame with historical price data.
        """
        filename = f"{self.data_dir}/{symbol}_{timeframe}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        return None
    
    async def _fetch_single_symbol(self, symbol: str, timeframe: str, start_date: str, end_date: Optional[str], use_cache: bool) -> Tuple[str, Optional[pd.DataFrame]]:
        """Fetch data for a single symbol asynchronously"""
        if use_cache:
            df = self.load_data_from_csv(symbol, timeframe)
            if df is not None:
                return symbol, df

        try:
            # Use synchronous API call since we're already in an async context
            df = self.download_historical_data(symbol, timeframe, start_date, end_date)
            if not df.empty:
                self.save_data_to_csv(df, symbol, timeframe)
                return symbol, df
        except Exception as e:
            print(f"Error fetching data for {symbol} at {timeframe}: {e}")
        return symbol, None

    async def fetch_data_for_all_symbols_async(self, symbols: List[str], timeframe: str, start_date: str, 
                                           end_date: Optional[str] = None, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols in parallel using asyncio"""
        data = {}
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._fetch_single_symbol(symbol, timeframe, start_date, end_date, use_cache))
            tasks.append(task)
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Fetching {timeframe} data"):
            symbol, df = await coro
            if df is not None:
                data[symbol] = df
                
        return data
        
    def fetch_data_for_all_symbols(self, symbols: List[str], timeframe: str, start_date: str, 
                                end_date: Optional[str] = None, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Synchronous wrapper for fetch_data_for_all_symbols_async"""
        # Use ThreadPoolExecutor for parallel processing - safer than mixing async/sync contexts
        data = {}
        with ThreadPoolExecutor(max_workers=min(32, len(symbols))) as executor:
            futures = {}
            for symbol in symbols:
                if use_cache:
                    df = self.load_data_from_csv(symbol, timeframe)
                    if df is not None:
                        data[symbol] = df
                        continue
                        
                futures[executor.submit(self.download_historical_data, symbol, timeframe, start_date, end_date)] = symbol
                
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {timeframe} data"):
                symbol = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        data[symbol] = df
                        self.save_data_to_csv(df, symbol, timeframe)
                except Exception as e:
                    print(f"Error fetching data for {symbol} at {timeframe}: {e}")
                    
        return data
    
    @staticmethod
    def _calculate_correlation_numpy(x: np.ndarray, y: np.ndarray) -> float:
        """Optimized correlation calculation using numpy"""
        # Direct numpy implementation of Pearson correlation
        x_norm = x - np.mean(x)
        y_norm = y - np.mean(y)
        return np.sum(x_norm * y_norm) / (np.sqrt(np.sum(x_norm**2)) * np.sqrt(np.sum(y_norm**2)))

    def calculate_correlation(self, data_dict: Dict[str, pd.DataFrame], reference_symbol: str = 'ETHUSDT') -> pd.Series:
        """Vectorized correlation calculation using numpy arrays"""
        if reference_symbol not in data_dict:
            raise ValueError(f"Reference symbol {reference_symbol} not found in data")

        # Convert all close prices to numpy arrays for faster computation
        ref_prices = data_dict[reference_symbol]['close'].values
        correlations = {}

        # Calculate correlations in parallel with a more efficient approach
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 16)) as executor:
            # Create a list of tuples (symbol, prices) for parallel processing
            symbol_prices = []
            for symbol, df in data_dict.items():
                prices = df['close'].values
                if len(prices) == len(ref_prices):
                    symbol_prices.append((symbol, prices))
            
            # Submit all correlation calculations as a batch
            futures_to_symbols = {}
            for symbol, prices in symbol_prices:
                future = executor.submit(self._calculate_correlation_numpy, ref_prices, prices)
                futures_to_symbols[future] = symbol

            # Process results as they complete
            for future in as_completed(futures_to_symbols):
                symbol = futures_to_symbols[future]
                try:
                    correlations[symbol] = future.result()
                except Exception as e:
                    print(f"Error calculating correlation for {symbol}: {e}")
                    correlations[symbol] = np.nan

        return pd.Series(correlations).sort_values(ascending=False)
    
    def visualize_correlation(self, correlation, timeframe, top_n=20):
        """
        Visualize correlation between reference symbol and top N symbols.
        
        Args:
            correlation (pandas.Series): Series with correlation values.
            timeframe (str): Timeframe for the klines.
            top_n (int, optional): Number of top correlated symbols to show. Defaults to 20.
        """
        # Get top N correlated symbols (excluding the reference symbol itself)
        top_corr = correlation.drop(self.eth_symbol).head(top_n)
        bottom_corr = correlation.drop(self.eth_symbol).tail(top_n)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot top correlations
        sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax1, palette='viridis')
        ax1.set_title(f'Top {top_n} Positive Correlations with {self.eth_symbol} ({timeframe})')
        ax1.set_xlabel('Correlation')
        ax1.set_ylabel('Symbol')
        
        # Plot bottom correlations
        sns.barplot(x=bottom_corr.values, y=bottom_corr.index, ax=ax2, palette='viridis')
        ax2.set_title(f'Top {top_n} Negative Correlations with {self.eth_symbol} ({timeframe})')
        ax2.set_xlabel('Correlation')
        ax2.set_ylabel('Symbol')
        
        plt.tight_layout()
        plt.savefig(f'correlation_{timeframe}.png', dpi=300)
        plt.show()
    
    async def _analyze_timeframe(self, timeframe_name: str, timeframe_code: str, usdt_symbols: List[str],
                                start_date: str, end_date: Optional[str], top_n: int, use_cache: bool) -> Tuple[str, pd.Series]:
        """Analyze a single timeframe asynchronously"""
        print(f"\nAnalyzing {timeframe_name} timeframe")
        # Use the async version directly since we're already in an async context
        data = await self.fetch_data_for_all_symbols_async(usdt_symbols, timeframe_name, start_date, end_date, use_cache)
        correlation = self.calculate_correlation(data, self.eth_symbol)
        correlation.to_csv(f'correlation_{timeframe_name}.csv')
        self.visualize_correlation(correlation, timeframe_name, top_n)
        return timeframe_name, correlation

    def analyze_all_timeframes(self, start_date: str, end_date: Optional[str] = None, top_n: int = 20, use_cache: bool = True) -> Dict[str, pd.Series]:
        """Analyze all timeframes sequentially to avoid asyncio issues"""
        # Get symbols only once
        all_symbols = self.get_all_futures_symbols()
        usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        if self.eth_symbol not in usdt_symbols:
            usdt_symbols.append(self.eth_symbol)
        print(f"Found {len(usdt_symbols)} USDT futures symbols")
        
        results = {}
        # Process each timeframe sequentially - safer than nested async loops
        for timeframe_name, timeframe_code in self.timeframes.items():
            print(f"\nAnalyzing {timeframe_name} timeframe")
            # Fetch data for all symbols
            data = self.fetch_data_for_all_symbols(usdt_symbols, timeframe_name, start_date, end_date, use_cache)
            # Calculate correlation
            correlation = self.calculate_correlation(data, self.eth_symbol)
            results[timeframe_name] = correlation
            # Save correlation to CSV
            correlation.to_csv(f'correlation_{timeframe_name}.csv')
            # Visualize correlation
            self.visualize_correlation(correlation, timeframe_name, top_n)
        
        return results
    
    def create_correlation_heatmap(self, results, top_n=20):
        """
        Create a heatmap of correlations across all timeframes.
        
        Args:
            results (dict): Dictionary with timeframe as key and correlation Series as value.
            top_n (int, optional): Number of top correlated symbols to show. Defaults to 20.
        """
        # Get top N correlated symbols across all timeframes
        all_corr = pd.DataFrame(results)
        
        # Sort by average correlation
        all_corr['avg'] = all_corr.mean(axis=1)
        
        # Get top positive correlated symbols
        top_symbols = all_corr.sort_values('avg', ascending=False).drop(self.eth_symbol).head(top_n).index.tolist()
        
        # Get top negative correlated symbols
        bottom_symbols = all_corr.sort_values('avg', ascending=True).drop(self.eth_symbol).head(top_n).index.tolist()
        
        # Create positive correlation heatmap data
        heatmap_data_positive = all_corr.loc[top_symbols].drop(columns=['avg'])
        
        # Create negative correlation heatmap data
        heatmap_data_negative = all_corr.loc[bottom_symbols].drop(columns=['avg'])
        
        # Create positive correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data_positive, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f'Top {top_n} Positively Correlated Symbols with {self.eth_symbol} Across Timeframes')
        plt.tight_layout()
        plt.savefig('correlation_heatmap_positive.png', dpi=300)
        plt.show()
        
        # Create negative correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data_negative, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Top {top_n} Negatively Correlated Symbols with {self.eth_symbol} Across Timeframes')
        plt.tight_layout()
        plt.savefig('correlation_heatmap_negative.png', dpi=300)
        plt.show()
        
        # Create combined heatmap (original functionality)
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data_positive, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f'Top {top_n} Correlated Symbols with {self.eth_symbol} Across Timeframes')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300)
        plt.show()
        
        return heatmap_data_positive, heatmap_data_negative

    def analyze_eth_downtrend_resilience(self, results, start_date, end_date, downtrend_threshold=-0.05, rebound_threshold=0.03, window_size=5, top_n=10):
    
        print(f"\nAnalyzing cryptocurrencies that remain stable during ETH downtrend...")
        
        # 1. Fetch historical data for ETH (using a 1-hour timeframe to capture more details)
        eth_data = self.download_historical_data(self.eth_symbol, '1h', start_date, end_date)
        
        # 2. Identify ETH downtrend periods
        eth_data['pct_change'] = eth_data['close'].pct_change(window_size)
        
        # Identify periods when ETH experienced a significant drop
        downtrend_periods = []
        current_downtrend = None
        
        for idx, row in eth_data.iterrows():
            if row['pct_change'] <= downtrend_threshold and current_downtrend is None:
                # Start a new downtrend period
                current_downtrend = {'start': idx, 'prices': [row['close']]}
            elif row['pct_change'] <= downtrend_threshold and current_downtrend is not None:
                # Continue the current downtrend period
                current_downtrend['prices'].append(row['close'])
            elif row['pct_change'] >= rebound_threshold and current_downtrend is not None:
                # End the downtrend period as ETH begins to rebound
                current_downtrend['end'] = idx
                current_downtrend['end_price'] = row['close']
                current_downtrend['start_price'] = current_downtrend['prices'][0]
                current_downtrend['lowest_price'] = min(current_downtrend['prices'])
                current_downtrend['drop_pct'] = (current_downtrend['lowest_price'] / current_downtrend['start_price'] - 1) * 100
                current_downtrend['rebound_pct'] = (current_downtrend['end_price'] / current_downtrend['lowest_price'] - 1) * 100
                
                downtrend_periods.append(current_downtrend)
                current_downtrend = None
        
        # If the last downtrend period hasn't ended
        if current_downtrend is not None:
            current_downtrend['end'] = eth_data.index[-1]
            current_downtrend['end_price'] = eth_data['close'].iloc[-1]
            current_downtrend['start_price'] = current_downtrend['prices'][0]
            current_downtrend['lowest_price'] = min(current_downtrend['prices'])
            current_downtrend['drop_pct'] = (current_downtrend['lowest_price'] / current_downtrend['start_price'] - 1) * 100
            current_downtrend['rebound_pct'] = (current_downtrend['end_price'] / current_downtrend['lowest_price'] - 1) * 100
            
            downtrend_periods.append(current_downtrend)
        
        print(f"Found {len(downtrend_periods)} ETH downtrend periods")
        
        if len(downtrend_periods) == 0:
            print("No ETH downtrend periods meeting the criteria were found within the specified date range")
            return None, None, None
        
        # 3. Analyze negatively correlated assets during each downtrend period
        # Get all USDT trading pairs
        all_symbols = self.get_all_futures_symbols()
        usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        
        # Store the performance of each coin during the downtrend period
        downtrend_performance = {}
        rebound_performance = {}
        
        for period in downtrend_periods:
            period_start = period['start']
            period_end = period['end']
            period_lowest = eth_data[eth_data['close'] == period['lowest_price']].index[0]
            
            print(f"\nAnalyzing downtrend period: {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}")
            print(f"ETH drop: {period['drop_pct']:.2f}%, Rebound: {period['rebound_pct']:.2f}%")
            
            # Analyze performance for each coin
            for symbol in tqdm(usdt_symbols, desc="Analyzing coin performance"):
                if symbol == self.eth_symbol:
                    continue
                
                try:
                    # Fetch historical data for the coin
                    coin_data = self.download_historical_data(symbol, '1h', 
                                                            period_start.strftime('%Y-%m-%d'), 
                                                            period_end.strftime('%Y-%m-%d'))
                    
                    if coin_data.empty or len(coin_data) < 2:
                        continue
                    
                    # Calculate performance during the downtrend period
                    try:
                        # Get the prices at the corresponding time points
                        start_price = coin_data.loc[:period_start].iloc[-1]['close']
                        lowest_time_price = coin_data.loc[:period_lowest].iloc[-1]['close']
                        end_price = coin_data.loc[:period_end].iloc[-1]['close']
                        
                        # Calculate the percentage change during the downtrend period
                        downtrend_pct = (lowest_time_price / start_price - 1) * 100
                        
                        # Calculate the percentage change during the ETH rebound period
                        rebound_pct = (end_price / lowest_time_price - 1) * 100
                        
                        # Store the results
                        if symbol not in downtrend_performance:
                            downtrend_performance[symbol] = []
                        
                        if symbol not in rebound_performance:
                            rebound_performance[symbol] = []
                        
                        downtrend_performance[symbol].append(downtrend_pct)
                        rebound_performance[symbol].append(rebound_pct)
                        
                    except (KeyError, IndexError) as e:
                        # Time point mismatch, skip this coin
                        continue
                        
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    continue
        
        # 4. Calculate average performance
        avg_downtrend = {}
        avg_rebound = {}
        
        for symbol in downtrend_performance:
            if len(downtrend_performance[symbol]) > 0:
                avg_downtrend[symbol] = sum(downtrend_performance[symbol]) / len(downtrend_performance[symbol])
            
        for symbol in rebound_performance:
            if len(rebound_performance[symbol]) > 0:
                avg_rebound[symbol] = sum(rebound_performance[symbol]) / len(rebound_performance[symbol])
        
        # 5. Identify the coins with the most stable performance during ETH downtrend (i.e., the smallest drop or even an increase)
        stable_coins = pd.Series(avg_downtrend).sort_values(ascending=False)
        
        # 6. Identify the coins with the best performance during ETH rebound
        best_rebounders = pd.Series(avg_rebound).sort_values(ascending=False)
        
        # 7. Identify the coins that are both stable and have a good rebound
        combined_score = {}
        
        for symbol in stable_coins.index:
            if symbol in best_rebounders.index:
                # Combined score = downtrend performance + rebound performance
                combined_score[symbol] = stable_coins[symbol] + best_rebounders[symbol]
        
        best_combined = pd.Series(combined_score).sort_values(ascending=False)
        
        # 8. Create a results DataFrame
        result_df = pd.DataFrame({
            'Downtrend Performance (%)': stable_coins,
            'Rebound Performance (%)': best_rebounders,
            'Combined Score': best_combined
        })
        
        # 9. Visualize the results
        # 9.1 Top N coins with the best downtrend performance
        plt.figure(figsize=(12, 8))
        sns.barplot(x=stable_coins.head(top_n).values, y=stable_coins.head(top_n).index, palette='viridis')
        plt.title(f'Top {top_n} Cryptocurrencies with the Most Stable Performance during ETH Downtrend')
        plt.xlabel('Price Change Percentage (%)')
        plt.ylabel('Coin')
        plt.tight_layout()
        plt.savefig('eth_downtrend_stable_coins.png', dpi=300)
        plt.show()
        
        # 9.2 Top N coins with the best performance during ETH rebound
        plt.figure(figsize=(12, 8))
        sns.barplot(x=best_rebounders.head(top_n).values, y=best_rebounders.head(top_n).index, palette='viridis')
        plt.title(f'Top {top_n} Cryptocurrencies with the Best Performance during ETH Rebound')
        plt.xlabel('Price Change Percentage (%)')
        plt.ylabel('Coin')
        plt.tight_layout()
        plt.savefig('eth_rebound_best_coins.png', dpi=300)
        plt.show()
        
        # 9.3 Top N coins with the best combined performance during the ETH market cycle
        plt.figure(figsize=(12, 8))
        sns.barplot(x=best_combined.head(top_n).values, y=best_combined.head(top_n).index, palette='viridis')
        plt.title(f'Top {top_n} Cryptocurrencies with the Best Combined Performance in the ETH Market Cycle')
        plt.xlabel('Combined Score')
        plt.ylabel('Coin')
        plt.tight_layout()
        plt.savefig('eth_cycle_best_coins.png', dpi=300)
        plt.show()
        
        # 10. Save the results to a CSV file
        result_df.to_csv('eth_market_cycle_analysis.csv')
        
        print("\nAnalysis complete!")
        print(f"Top {top_n} cryptocurrencies with the most stable performance during ETH downtrend:")
        print(stable_coins.head(top_n))
        
        print(f"\nTop {top_n} cryptocurrencies with the best performance during ETH rebound:")
        print(best_rebounders.head(top_n))
        
        print(f"\nTop {top_n} cryptocurrencies with the best combined performance in the ETH market cycle:")
        print(best_combined.head(top_n))
        
        return stable_coins, best_rebounders, best_combined


def main():
    # Set your Binance API credentials (optional for historical data)
    api_key = None  # Replace with your API key if needed
    api_secret = None  # Replace with your API secret if needed
    
    # Initialize the analyzer with max_klines parameter (100-300 range)
    max_klines = 200  # You can adjust this value between 100-300
    analyzer = CryptoCorrelationAnalyzer(api_key, api_secret, max_klines=max_klines)
    
    # Set date range (last 30 days by default)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Analyzing data from {start_date} to {end_date} with max {max_klines} K-lines per timeframe")
    
    # 選擇分析模式
    print("\n請選擇分析模式:")
    print("1. 標準相關性分析")
    print("2. ETH下跌期間穩定幣種分析")
    print("3. 兩種分析都執行")
    
    choice = input("請輸入選項 (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        # Analyze all timeframes
        results = analyzer.analyze_all_timeframes(start_date, end_date, top_n=20, use_cache=True)  # use Cache
        
        # Create correlation heatmap
        heatmap_data_positive, heatmap_data_negative = analyzer.create_correlation_heatmap(results, top_n=20)
        
        print("標準相關性分析完成!")
        print(f"Positive correlation heatmap saved as 'correlation_heatmap_positive.png'")
        print(f"Negative correlation heatmap saved as 'correlation_heatmap_negative.png'")
        print(f"Original correlation heatmap saved as 'correlation_heatmap.png'")
    
    if choice in ['2', '3']:
        # 設置更長的時間範圍來分析ETH下跌期間
        long_start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        print(f"\n分析ETH下跌期間的穩定幣種 (時間範圍: {long_start_date} 到 {end_date})")
        
        # 如果已經執行了標準分析，使用其結果；否則執行一個快速分析
        if choice == '3':
            # 使用已有的結果
            stable_coins, best_rebounders, best_combined = analyzer.analyze_eth_downtrend_resilience(
                results, long_start_date, end_date, 
                downtrend_threshold=-0.05,  # ETH下跌5%視為下跌期
                rebound_threshold=0.03,     # ETH反彈3%視為反彈期
                window_size=5,              # 5小時窗口
                top_n=10                    # 顯示前10名
            )
        else:
            # 執行快速分析獲取結果
            quick_results = analyzer.analyze_all_timeframes(long_start_date, end_date, top_n=20, use_cache=True)
            stable_coins, best_rebounders, best_combined = analyzer.analyze_eth_downtrend_resilience(
                quick_results, long_start_date, end_date, 
                downtrend_threshold=-0.05,
                rebound_threshold=0.03,
                window_size=5,
                top_n=10
            )
        
        print("ETH下跌期間穩定幣種分析完成!")
        print("結果已保存到 'eth_market_cycle_analysis.csv'")
    
    print("\n所有分析完成!")


if __name__ == "__main__":
    main()
