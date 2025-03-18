# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Cryptocurrency Correlation Analyzer
# This script analyzes the price correlation between ETH and other cryptocurrencies
# using Binance historical data across different timeframes (1min, 5min, 15min, 1h, 4h).
# """

# import os
# import time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from binance.client import Client
# from datetime import datetime, timedelta
# from tqdm import tqdm
# import json
# import warnings
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from functools import lru_cache
# from typing import Dict, List, Optional, Tuple
# import asyncio
# from collections import defaultdict
# import pickle
# import hashlib
# import random
# import requests
# from requests.exceptions import RequestException
# import argparse

# warnings.filterwarnings('ignore')

# class CryptoCorrelationAnalyzer:
#     def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, max_klines: int = 200,
#                  request_delay: float = 1.0, max_workers: int = 5, use_proxies: bool = False, 
#                  proxies: List[str] = None, cache_expiry: int = 86400):
#         """
#         Initialize the Crypto Correlation Analyzer with optimized settings.
        
#         Args:
#             api_key (str, optional): Binance API key. Defaults to None.
#             api_secret (str, optional): Binance API secret. Defaults to None.
#             max_klines (int, optional): Maximum number of K-lines to fetch. Defaults to 200.
#                 Should be between 100-300 for optimal performance.
#             request_delay (float, optional): Delay between API requests in seconds. Defaults to 1.0.
#             max_workers (int, optional): Maximum number of concurrent workers. Defaults to 5.
#             use_proxies (bool, optional): Whether to use proxy rotation. Defaults to False.
#             proxies (List[str], optional): List of proxy URLs. Defaults to None.
#             cache_expiry (int, optional): Cache expiry time in seconds. Defaults to 86400 (1 day).
#         """
#         self.client = Client(api_key, api_secret)
#         self.timeframes = {
#             '1m': Client.KLINE_INTERVAL_1MINUTE,
#             '5m': Client.KLINE_INTERVAL_5MINUTE,
#             '15m': Client.KLINE_INTERVAL_15MINUTE,
#             '1h': Client.KLINE_INTERVAL_1HOUR,
#             '4h': Client.KLINE_INTERVAL_4HOUR
#         }
#         self.eth_symbol = 'ETHUSDT'
#         self.data_dir = 'historical_data'
#         self.cache_dir = f'{self.data_dir}/cache'
#         self.max_klines = max(100, min(300, max_klines))  # Ensure max_klines is between 100-300
#         os.makedirs(self.data_dir, exist_ok=True)
#         os.makedirs(self.cache_dir, exist_ok=True)
        
#         # API request settings
#         self.request_delay = request_delay
#         self.max_workers = max_workers
#         self.use_proxies = use_proxies
#         self.proxies = proxies or []
#         self.cache_expiry = cache_expiry
#         self.last_request_time = 0
        
#         # Rate limit tracking
#         self.weight_used = 0
#         self.weight_reset_time = time.time() + 60  # Reset weight counter every minute
#         self.MAX_WEIGHT_PER_MINUTE = 2400  # Binance Futures API weight limit per minute
#         self.MAX_REQUESTS_PER_SECOND = 20  # Binance API max requests per second
#         self.requests_this_second = 0
#         self.second_start_time = time.time()
        
#         # Initialize cache for data operations
#         self._symbol_cache = {}
#         self._correlation_cache = {}
        
#         # Configure numpy for optimal performance
#         np.set_printoptions(precision=8, suppress=True)
        
#         # Configure pandas for better performance
#         pd.set_option('compute.use_numexpr', True)
        
#     def _get_proxy(self):
#         """Get a random proxy from the proxy list."""
#         if not self.use_proxies or not self.proxies:
#             return None
#         return random.choice(self.proxies)
    
#     def _wait_for_rate_limit(self, weight=1):
#         """
#         Wait to respect rate limits based on weight and request frequency.
        
#         Args:
#             weight (int): The weight of the request. Defaults to 1.
#         """
#         current_time = time.time()
        
#         # Check if we need to reset the weight counter
#         if current_time >= self.weight_reset_time:
#             self.weight_used = 0
#             self.weight_reset_time = current_time + 60
        
#         # Check if we need to reset the requests per second counter
#         if current_time - self.second_start_time >= 1:
#             self.requests_this_second = 0
#             self.second_start_time = current_time
        
#         # Check if we're approaching the weight limit
#         if self.weight_used + weight >= self.MAX_WEIGHT_PER_MINUTE:
#             # Wait until the next minute starts
#             sleep_time = self.weight_reset_time - current_time
#             if sleep_time > 0:
#                 print(f"Approaching weight limit ({self.weight_used}/{self.MAX_WEIGHT_PER_MINUTE}). Waiting {sleep_time:.2f} seconds...")
#                 time.sleep(sleep_time)
#                 # Reset counters after waiting
#                 self.weight_used = 0
#                 self.weight_reset_time = time.time() + 60
#                 self.requests_this_second = 0
#                 self.second_start_time = time.time()
#                 current_time = time.time()
        
#         # Check if we're approaching the requests per second limit
#         if self.requests_this_second >= self.MAX_REQUESTS_PER_SECOND:
#             # Wait until the next second starts
#             sleep_time = 1 - (current_time - self.second_start_time)
#             if sleep_time > 0:
#                 time.sleep(sleep_time)
#                 self.requests_this_second = 0
#                 self.second_start_time = time.time()
#                 current_time = time.time()
        
#         # Enforce minimum delay between requests
#         elapsed = current_time - self.last_request_time
#         if elapsed < self.request_delay:
#             sleep_time = self.request_delay - elapsed
#             time.sleep(sleep_time)
        
#         # Update counters
#         self.last_request_time = time.time()
#         self.weight_used += weight
#         self.requests_this_second += 1
    
#     def _get_cache_key(self, symbol, timeframe, start_str, end_str):
#         """Generate a unique cache key for the data request."""
#         key_data = f"{symbol}_{timeframe}_{start_str}_{end_str}_{self.max_klines}"
#         return hashlib.md5(key_data.encode()).hexdigest()
    
#     def _save_to_cache(self, cache_key, data):
#         """Save data to cache file."""
#         cache_file = f"{self.cache_dir}/{cache_key}.pkl"
#         cache_meta = f"{self.cache_dir}/{cache_key}.meta"
        
#         # Save data
#         with open(cache_file, 'wb') as f:
#             pickle.dump(data, f)
        
#         # Save metadata
#         meta = {
#             'timestamp': time.time(),
#             'symbol': data.index.name if hasattr(data, 'index') and data.index.name else None
#         }
#         with open(cache_meta, 'w') as f:
#             json.dump(meta, f)
    
#     def _load_from_cache(self, cache_key):
#         """Load data from cache if it exists and is not expired."""
#         cache_file = f"{self.cache_dir}/{cache_key}.pkl"
#         cache_meta = f"{self.cache_dir}/{cache_key}.meta"
        
#         if not (os.path.exists(cache_file) and os.path.exists(cache_meta)):
#             return None
        
#         # Check if cache is expired
#         with open(cache_meta, 'r') as f:
#             meta = json.load(f)
        
#         if time.time() - meta['timestamp'] > self.cache_expiry:
#             return None  # Cache expired
        
#         # Load data from cache
#         with open(cache_file, 'rb') as f:
#             return pickle.load(f)
        
#     def get_all_futures_symbols(self):
#         """
#         Get all available futures symbols from Binance.
        
#         Returns:
#             list: List of futures symbols.
#         """
#         try:
#             self._wait_for_rate_limit(weight=10)  # Higher weight for exchange info
#             exchange_info = self.client.futures_exchange_info()
#             symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
#             return symbols
#         except Exception as e:
#             print(f"Error getting futures symbols: {e}")
#             # Return a limited set of popular symbols as fallback
#             return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
#                     'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT']
    
#     def download_historical_data(self, symbol, timeframe, start_date, end_date=None, retry_count=3):
#         """
#         Download historical kline data for a symbol and timeframe.
        
#         Args:
#             symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
#             timeframe (str): Timeframe for the klines (e.g., '1m', '1h').
#             start_date (str): Start date in format 'YYYY-MM-DD'.
#             end_date (str, optional): End date in format 'YYYY-MM-DD'.
#             retry_count (int, optional): Number of retries on failure. Defaults to 3.
            
#         Returns:
#             pandas.DataFrame: DataFrame with historical price data.
#         """
#         if end_date is None:
#             end_date = datetime.now().strftime('%Y-%m-%d')
        
#         # Calculate appropriate date range based on max_klines and timeframe
#         end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
#         # Adjust start date based on timeframe and max_klines
#         # This is a rough estimation to get approximately max_klines candles
#         timeframe_minutes = {
#             '1m': 1,
#             '5m': 5,
#             '15m': 15,
#             '1h': 60,
#             '4h': 240
#         }
        
#         # Calculate minutes needed for max_klines
#         minutes_needed = self.max_klines * timeframe_minutes[timeframe]
        
#         # Calculate adjusted start date
#         adjusted_start_dt = end_dt - timedelta(minutes=minutes_needed)
#         original_start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
#         # Use the later of the two dates (original or adjusted)
#         start_dt = max(original_start_dt, adjusted_start_dt)
#         adjusted_start_date = start_dt.strftime('%Y-%m-%d')
        
#         print(f"Fetching ~{self.max_klines} candles for {symbol} at {timeframe} timeframe")
#         print(f"Adjusted date range: {adjusted_start_date} to {end_date}")
            
#         start_ts = int(start_dt.timestamp() * 1000)
#         end_ts = int(end_dt.timestamp() * 1000)
        
#         # Check cache first
#         cache_key = self._get_cache_key(symbol, timeframe, start_ts, end_ts)
#         cached_data = self._load_from_cache(cache_key)
#         if cached_data is not None:
#             print(f"Using cached data for {symbol} ({timeframe})")
#             return cached_data
        
#         # Calculate request weight based on limit parameter
#         request_weight = 1  # Default weight
#         if self.max_klines > 100:
#             if self.max_klines <= 500:
#                 request_weight = 2
#             elif self.max_klines <= 1000:
#                 request_weight = 5
#             else:
#                 request_weight = 10
        
#         # Try to get data with retries
#         for attempt in range(retry_count):
#             try:
#                 # Wait for rate limit with appropriate weight
#                 self._wait_for_rate_limit(weight=request_weight)
                
#                 # Get the data from Binance with limit parameter
#                 klines = self.client.futures_historical_klines(
#                     symbol=symbol,
#                     interval=self.timeframes[timeframe],
#                     start_str=start_ts,
#                     end_str=end_ts,
#                     limit=self.max_klines  # Limit the number of candles
#                 )
                
#                 # Create a DataFrame
#                 df = pd.DataFrame(klines, columns=[
#                     'timestamp', 'open', 'high', 'low', 'close', 'volume',
#                     'close_time', 'quote_asset_volume', 'number_of_trades',
#                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
#                 ])
                
#                 # Convert timestamp to datetime
#                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
#                 # Convert price columns to float
#                 for col in ['open', 'high', 'low', 'close', 'volume']:
#                     df[col] = df[col].astype(float)
                    
#                 # Set timestamp as index
#                 df.set_index('timestamp', inplace=True)
                
#                 # Save to cache
#                 self._save_to_cache(cache_key, df)
                
#                 return df
                
#             except Exception as e:
#                 print(f"Error fetching data for {symbol} at {timeframe} (attempt {attempt+1}/{retry_count}): {e}")
                
#                 # Exponential backoff
#                 if attempt < retry_count - 1:
#                     backoff_time = (2 ** attempt) * self.request_delay
#                     print(f"Retrying in {backoff_time:.2f} seconds...")
#                     time.sleep(backoff_time)
        
#         # If all retries failed, return empty DataFrame
#         print(f"Failed to fetch data for {symbol} at {timeframe} after {retry_count} attempts")
#         return pd.DataFrame()
    
#     def save_data_to_csv(self, df, symbol, timeframe):
#         """
#         Save DataFrame to CSV file.
        
#         Args:
#             df (pandas.DataFrame): DataFrame to save.
#             symbol (str): Trading pair symbol.
#             timeframe (str): Timeframe for the klines.
#         """
#         filename = f"{self.data_dir}/{symbol}_{timeframe}.csv"
#         df.to_csv(filename)
#         print(f"Saved {symbol} {timeframe} data to {filename}")
    
#     def load_data_from_csv(self, symbol, timeframe):
#         """
#         Load DataFrame from CSV file.
        
#         Args:
#             symbol (str): Trading pair symbol.
#             timeframe (str): Timeframe for the klines.
            
#         Returns:
#             pandas.DataFrame: DataFrame with historical price data.
#         """
#         filename = f"{self.data_dir}/{symbol}_{timeframe}.csv"
#         if os.path.exists(filename):
#             df = pd.read_csv(filename)
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
#             df.set_index('timestamp', inplace=True)
#             return df
#         return None
    
#     async def _fetch_single_symbol(self, symbol: str, timeframe: str, start_date: str, end_date: Optional[str], use_cache: bool) -> Tuple[str, Optional[pd.DataFrame]]:
#         """Fetch data for a single symbol asynchronously"""
#         if use_cache:
#             df = self.load_data_from_csv(symbol, timeframe)
#             if df is not None:
#                 return symbol, df

#         try:
#             # Use synchronous API call since we're already in an async context
#             df = self.download_historical_data(symbol, timeframe, start_date, end_date)
#             if not df.empty:
#                 self.save_data_to_csv(df, symbol, timeframe)
#                 return symbol, df
#         except Exception as e:
#             print(f"Error fetching data for {symbol} at {timeframe}: {e}")
#         return symbol, None

#     def get_optimal_batch_size(self, remaining_symbols: int, timeframe: str) -> int:
#         """
#         Calculate the optimal batch size based on current rate limit usage and timeframe.
        
#         Args:
#             remaining_symbols (int): Number of symbols remaining to process
#             timeframe (str): Timeframe being processed
            
#         Returns:
#             int: Optimal batch size
#         """
#         # Calculate remaining weight capacity
#         remaining_weight = self.MAX_WEIGHT_PER_MINUTE - self.weight_used
        
#         # Estimate weight per request based on timeframe and max_klines
#         request_weight = 1  # Default weight
#         if self.max_klines > 100:
#             if self.max_klines <= 500:
#                 request_weight = 2
#             elif self.max_klines <= 1000:
#                 request_weight = 5
#             else:
#                 request_weight = 10
                
#         # Calculate how many requests we can make with remaining weight
#         max_possible_requests = max(1, remaining_weight // request_weight)
        
#         # Consider time remaining in the current minute
#         time_remaining = max(0.1, self.weight_reset_time - time.time())
#         requests_per_second = self.MAX_REQUESTS_PER_SECOND
        
#         # Calculate how many requests we can make in the remaining time
#         max_time_based_requests = int(time_remaining * requests_per_second * 0.8)  # 80% to be safe
        
#         # Take the minimum of weight-based and time-based limits
#         max_requests = min(max_possible_requests, max_time_based_requests)
        
#         # Consider our max_workers setting
#         max_workers_limit = min(self.max_workers, remaining_symbols)
        
#         # Calculate final batch size
#         optimal_batch_size = min(max_requests, max_workers_limit, remaining_symbols)
        
#         # Ensure batch size is at least 1 but not more than 20 (to avoid overwhelming the API)
#         return max(1, min(20, optimal_batch_size))

#     async def fetch_data_for_all_symbols_async(self, symbols: List[str], timeframe: str, start_date: str, 
#                                            end_date: Optional[str] = None, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
#         """Fetch historical data for all symbols in parallel using asyncio"""
#         data = {}
#         remaining_symbols = symbols.copy()
        
#         while remaining_symbols:
#             # Calculate optimal batch size based on current rate limit usage
#             batch_size = self.get_optimal_batch_size(len(remaining_symbols), timeframe)
            
#             # Get the next batch of symbols
#             batch = remaining_symbols[:batch_size]
#             remaining_symbols = remaining_symbols[batch_size:]
            
#             print(f"Processing batch of {len(batch)} symbols (remaining: {len(remaining_symbols)})")
            
#             # Process the batch
#             batch_tasks = []
#             for symbol in batch:
#                 if use_cache:
#                     df = self.load_data_from_csv(symbol, timeframe)
#                     if df is not None:
#                         data[symbol] = df
#                         continue
                
#                 task = asyncio.create_task(self._fetch_single_symbol(symbol, timeframe, start_date, end_date, use_cache))
#                 batch_tasks.append(task)
            
#             if batch_tasks:
#                 for coro in tqdm(asyncio.as_completed(batch_tasks), total=len(batch_tasks), desc=f"Fetching {timeframe} data (batch)"):
#                     symbol, df = await coro
#                     if df is not None:
#                         data[symbol] = df
            
#             # Add a small delay between batches to avoid hitting rate limits
#             if remaining_symbols:
#                 await asyncio.sleep(0.5)
                
#         return data
        
#     def fetch_data_for_all_symbols(self, symbols: List[str], timeframe: str, start_date: str, 
#                                 end_date: Optional[str] = None, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
#         """Synchronous wrapper for fetch_data_for_all_symbols_async"""
#         # Use ThreadPoolExecutor for parallel processing - safer than mixing async/sync contexts
#         data = {}
#         remaining_symbols = symbols.copy()
        
#         while remaining_symbols:
#             # Calculate optimal batch size based on current rate limit usage
#             batch_size = self.get_optimal_batch_size(len(remaining_symbols), timeframe)
            
#             # Get the next batch of symbols
#             batch = remaining_symbols[:batch_size]
#             remaining_symbols = remaining_symbols[batch_size:]
            
#             print(f"Processing batch of {len(batch)} symbols (remaining: {len(remaining_symbols)})")
            
#             with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
#                 futures = {}
#                 for symbol in batch:
#                     if use_cache:
#                         df = self.load_data_from_csv(symbol, timeframe)
#                         if df is not None:
#                             data[symbol] = df
#                             continue
                            
#                     futures[executor.submit(self.download_historical_data, symbol, timeframe, start_date, end_date)] = symbol
                
#                 if futures:
#                     for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {timeframe} data (batch)"):
#                         symbol = futures[future]
#                         try:
#                             df = future.result()
#                             if not df.empty:
#                                 data[symbol] = df
#                                 self.save_data_to_csv(df, symbol, timeframe)
#                         except Exception as e:
#                             print(f"Error fetching data for {symbol} at {timeframe}: {e}")
            
#             # Add a small delay between batches to avoid hitting rate limits
#             if remaining_symbols:
#                 print(f"Batch complete. Moving to next batch...")
#                 time.sleep(0.5)  # Small delay between batches
                    
#         return data
    
#     @staticmethod
#     def _calculate_correlation_numpy(x: np.ndarray, y: np.ndarray) -> float:
#         """Optimized correlation calculation using numpy"""
#         # Direct numpy implementation of Pearson correlation
#         x_norm = x - np.mean(x)
#         y_norm = y - np.mean(y)
#         return np.sum(x_norm * y_norm) / (np.sqrt(np.sum(x_norm**2)) * np.sqrt(np.sum(y_norm**2)))

#     def calculate_correlation(self, data_dict: Dict[str, pd.DataFrame], reference_symbol: str = 'ETHUSDT') -> pd.Series:
#         """Vectorized correlation calculation using numpy arrays"""
#         if reference_symbol not in data_dict:
#             raise ValueError(f"Reference symbol {reference_symbol} not found in data")

#         # Convert all close prices to numpy arrays for faster computation
#         ref_prices = data_dict[reference_symbol]['close'].values
#         correlations = {}

#         # Calculate correlations in parallel with a more efficient approach
#         with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 16)) as executor:
#             # Create a list of tuples (symbol, prices) for parallel processing
#             symbol_prices = []
#             for symbol, df in data_dict.items():
#                 prices = df['close'].values
#                 if len(prices) == len(ref_prices):
#                     symbol_prices.append((symbol, prices))
            
#             # Submit all correlation calculations as a batch
#             futures_to_symbols = {}
#             for symbol, prices in symbol_prices:
#                 future = executor.submit(self._calculate_correlation_numpy, ref_prices, prices)
#                 futures_to_symbols[future] = symbol

#             # Process results as they complete
#             for future in as_completed(futures_to_symbols):
#                 symbol = futures_to_symbols[future]
#                 try:
#                     correlations[symbol] = future.result()
#                 except Exception as e:
#                     print(f"Error calculating correlation for {symbol}: {e}")
#                     correlations[symbol] = np.nan

#         return pd.Series(correlations).sort_values(ascending=False)
    
#     def visualize_correlation(self, correlation, timeframe, top_n=20):
#         """
#         Visualize correlation between reference symbol and top N symbols.
        
#         Args:
#             correlation (pandas.Series): Series with correlation values.
#             timeframe (str): Timeframe for the klines.
#             top_n (int, optional): Number of top correlated symbols to show. Defaults to 20.
#         """
#         # Get top N correlated symbols (excluding the reference symbol itself)
#         top_corr = correlation.drop(self.eth_symbol).head(top_n)
#         bottom_corr = correlation.drop(self.eth_symbol).tail(top_n)
        
#         # Create figure with two subplots
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
#         # Plot top correlations
#         sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax1, palette='viridis')
#         ax1.set_title(f'Top {top_n} Positive Correlations with {self.eth_symbol} ({timeframe})')
#         ax1.set_xlabel('Correlation')
#         ax1.set_ylabel('Symbol')
        
#         # Plot bottom correlations
#         sns.barplot(x=bottom_corr.values, y=bottom_corr.index, ax=ax2, palette='viridis')
#         ax2.set_title(f'Top {top_n} Negative Correlations with {self.eth_symbol} ({timeframe})')
#         ax2.set_xlabel('Correlation')
#         ax2.set_ylabel('Symbol')
        
#         plt.tight_layout()
#         plt.savefig(f'correlation_{timeframe}.png', dpi=300)
#         plt.show()
    
#     async def _analyze_timeframe(self, timeframe_name: str, timeframe_code: str, usdt_symbols: List[str],
#                                 start_date: str, end_date: Optional[str], top_n: int, use_cache: bool) -> Tuple[str, pd.Series]:
#         """Analyze a single timeframe asynchronously"""
#         print(f"\nAnalyzing {timeframe_name} timeframe")
#         # Use the async version directly since we're already in an async context
#         data = await self.fetch_data_for_all_symbols_async(usdt_symbols, timeframe_name, start_date, end_date, use_cache)
#         correlation = self.calculate_correlation(data, self.eth_symbol)
#         correlation.to_csv(f'correlation_{timeframe_name}.csv')
#         self.visualize_correlation(correlation, timeframe_name, top_n)
#         return timeframe_name, correlation

#     def analyze_all_timeframes(self, start_date: str, end_date: Optional[str] = None, top_n: int = 20, use_cache: bool = True) -> Dict[str, pd.Series]:
#         """Analyze all timeframes sequentially to avoid asyncio issues"""
#         # Get symbols only once
#         all_symbols = self.get_all_futures_symbols()
#         usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
#         if self.eth_symbol not in usdt_symbols:
#             usdt_symbols.append(self.eth_symbol)
#         print(f"Found {len(usdt_symbols)} USDT futures symbols")
        
#         # Prioritize ETH symbol to ensure it's always processed
#         if self.eth_symbol in usdt_symbols:
#             usdt_symbols.remove(self.eth_symbol)
#             prioritized_symbols = [self.eth_symbol] + usdt_symbols
#         else:
#             prioritized_symbols = usdt_symbols
        
#         results = {}
#         # Process each timeframe sequentially - safer than nested async loops
#         for timeframe_name, timeframe_code in self.timeframes.items():
#             print(f"\nAnalyzing {timeframe_name} timeframe")
            
#             # Reset rate limit counters between timeframes to ensure fresh start
#             self.weight_used = 0
#             self.weight_reset_time = time.time() + 60
#             self.requests_this_second = 0
#             self.second_start_time = time.time()
            
#             # Fetch data for all symbols
#             data = self.fetch_data_for_all_symbols(prioritized_symbols, timeframe_name, start_date, end_date, use_cache)
            
#             # Check if we have ETH data, which is required for correlation
#             if self.eth_symbol not in data or data[self.eth_symbol].empty:
#                 print(f"Warning: No {self.eth_symbol} data available for {timeframe_name}. Skipping correlation analysis.")
#                 continue
                
#             # Calculate correlation
#             correlation = self.calculate_correlation(data, self.eth_symbol)
#             results[timeframe_name] = correlation
            
#             # Save correlation to CSV
#             correlation.to_csv(f'correlation_{timeframe_name}.csv')
            
#             # Visualize correlation
#             self.visualize_correlation(correlation, timeframe_name, top_n)
            
#             # Add a delay between timeframes to avoid rate limit issues
#             if timeframe_name != list(self.timeframes.keys())[-1]:  # If not the last timeframe
#                 print(f"Completed analysis for {timeframe_name}. Waiting before next timeframe...")
#                 time.sleep(2)  # Short delay between timeframes
        
#         return results
    
#     def create_correlation_heatmap(self, results, top_n=20):
#         """
#         Create a heatmap of correlations across all timeframes.
        
#         Args:
#             results (dict): Dictionary with timeframe as key and correlation Series as value.
#             top_n (int, optional): Number of top correlated symbols to show. Defaults to 20.
#         """
#         # Get top N correlated symbols across all timeframes
#         all_corr = pd.DataFrame(results)
        
#         # Sort by average correlation
#         all_corr['avg'] = all_corr.mean(axis=1)
        
#         # Get top positive correlated symbols
#         top_symbols = all_corr.sort_values('avg', ascending=False).drop(self.eth_symbol).head(top_n).index.tolist()
        
#         # Get top negative correlated symbols
#         bottom_symbols = all_corr.sort_values('avg', ascending=True).drop(self.eth_symbol).head(top_n).index.tolist()
        
#         # Create positive correlation heatmap data
#         heatmap_data_positive = all_corr.loc[top_symbols].drop(columns=['avg'])
        
#         # Create negative correlation heatmap data
#         heatmap_data_negative = all_corr.loc[bottom_symbols].drop(columns=['avg'])
        
#         # Create positive correlation heatmap
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(heatmap_data_positive, annot=True, cmap='viridis', fmt='.2f')
#         plt.title(f'Top {top_n} Positively Correlated Symbols with {self.eth_symbol} Across Timeframes')
#         plt.tight_layout()
#         plt.savefig('correlation_heatmap_positive.png', dpi=300)
#         plt.show()
        
#         # Create negative correlation heatmap
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(heatmap_data_negative, annot=True, cmap='coolwarm', fmt='.2f')
#         plt.title(f'Top {top_n} Negatively Correlated Symbols with {self.eth_symbol} Across Timeframes')
#         plt.tight_layout()
#         plt.savefig('correlation_heatmap_negative.png', dpi=300)
#         plt.show()
        
#         # Create combined heatmap (original functionality)
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(heatmap_data_positive, annot=True, cmap='viridis', fmt='.2f')
#         plt.title(f'Top {top_n} Correlated Symbols with {self.eth_symbol} Across Timeframes')
#         plt.tight_layout()
#         plt.savefig('correlation_heatmap.png', dpi=300)
#         plt.show()
        
#         return heatmap_data_positive, heatmap_data_negative

#     def visualize_price_movements(self, data_dict, downtrend_periods=None, timeframe='1h', top_n=5):
#         """
#         Visualize price movements of ETH and top stable coins during downtrends.
        
#         Args:
#             data_dict (dict): Dictionary with symbol as key and DataFrame as value.
#             downtrend_periods (list, optional): List of downtrend periods. Defaults to None.
#             timeframe (str, optional): Timeframe for the visualization. Defaults to '1h'.
#             top_n (int, optional): Number of top stable coins to show. Defaults to 5.
#         """
#         if self.eth_symbol not in data_dict:
#             print(f"Error: {self.eth_symbol} data not found")
#             return
            
#         # Normalize all prices to percentage change from the first point
#         eth_data = data_dict[self.eth_symbol].copy()
#         eth_data['normalized'] = eth_data['close'] / eth_data['close'].iloc[0] * 100
        
#         # Create figure
#         plt.figure(figsize=(15, 10))
        
#         # Plot ETH price
#         plt.plot(eth_data.index, eth_data['normalized'], label=self.eth_symbol, linewidth=2, color='black')
        
#         # Get top stable coins if available
#         if hasattr(self, '_last_stable_coins') and self._last_stable_coins is not None:
#             stable_coins = self._last_stable_coins.head(top_n).index.tolist()
            
#             # Plot each stable coin
#             colors = plt.cm.viridis(np.linspace(0, 1, len(stable_coins)))
#             for i, symbol in enumerate(stable_coins):
#                 if symbol in data_dict:
#                     coin_data = data_dict[symbol].copy()
#                     coin_data['normalized'] = coin_data['close'] / coin_data['close'].iloc[0] * 100
#                     plt.plot(coin_data.index, coin_data['normalized'], label=symbol, linewidth=1.5, color=colors[i])
        
#         # Mark downtrend periods if provided
#         if downtrend_periods:
#             for i, period in enumerate(downtrend_periods):
#                 start = period['start']
#                 end = period['end']
#                 lowest_point = eth_data[eth_data['close'] == period['lowest_price']].index[0]
                
#                 # Shade the downtrend area
#                 plt.axvspan(start, end, alpha=0.2, color='red')
                
#                 # Mark the lowest point
#                 plt.scatter(lowest_point, eth_data.loc[lowest_point, 'normalized'], 
#                            color='red', s=100, zorder=5, marker='v')
                
#                 # Add text annotation for the period
#                 mid_point = start + (end - start) / 2
#                 y_pos = eth_data['normalized'].max() * (0.95 - 0.05 * i)
#                 plt.annotate(f"Period {i+1}: {period['drop_pct']:.1f}%", 
#                             xy=(mid_point, y_pos),
#                             xytext=(mid_point, y_pos),
#                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
#                             ha='center')
        
#         plt.title(f'Price Movements Comparison ({timeframe} timeframe)', fontsize=16)
#         plt.xlabel('Date', fontsize=12)
#         plt.ylabel('Normalized Price (%)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#         plt.tight_layout()
#         plt.savefig(f'price_movements_{timeframe}.png', dpi=300)
#         plt.show()
        
#     def create_scatter_plot(self, stable_coins, best_rebounders, top_n=20):
#         """
#         Create a scatter plot of downtrend performance vs rebound performance.
        
#         Args:
#             stable_coins (pandas.Series): Series with downtrend performance values.
#             best_rebounders (pandas.Series): Series with rebound performance values.
#             top_n (int, optional): Number of top coins to highlight. Defaults to 20.
#         """
#         # Create a DataFrame for the scatter plot
#         scatter_data = pd.DataFrame({
#             'Downtrend Performance (%)': stable_coins,
#             'Rebound Performance (%)': best_rebounders
#         })
        
#         # Calculate combined score
#         scatter_data['Combined Score'] = scatter_data['Downtrend Performance (%)'] + scatter_data['Rebound Performance (%)']
        
#         # Sort by combined score
#         scatter_data = scatter_data.sort_values('Combined Score', ascending=False)
        
#         # Create figure
#         plt.figure(figsize=(12, 10))
        
#         # Plot all points
#         plt.scatter(scatter_data['Downtrend Performance (%)'], 
#                    scatter_data['Rebound Performance (%)'], 
#                    alpha=0.5, s=50, color='gray')
        
#         # Highlight top N coins
#         top_coins = scatter_data.head(top_n)
#         plt.scatter(top_coins['Downtrend Performance (%)'], 
#                    top_coins['Rebound Performance (%)'], 
#                    alpha=1.0, s=100, color='green')
        
#         # Add labels for top coins
#         for idx, row in top_coins.iterrows():
#             plt.annotate(idx, 
#                         (row['Downtrend Performance (%)'], row['Rebound Performance (%)']),
#                         xytext=(5, 5), textcoords='offset points',
#                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
        
#         # Add quadrant lines
#         plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
#         plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
#         # Add quadrant labels
#         plt.annotate('Best Overall\n(Stable + Strong Rebound)', xy=(5, 5), xycoords='data',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.8))
        
#         plt.annotate('Good Rebound but Drops\nDuring Downtrend', xy=(-5, 5), xycoords='data',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
#         plt.annotate('Stable but Poor Rebound', xy=(5, -5), xycoords='data',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.8))
        
#         plt.annotate('Poor Overall\n(Drops + Weak Rebound)', xy=(-5, -5), xycoords='data',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8))
        
#         plt.title('Cryptocurrency Performance: Downtrend vs Rebound', fontsize=16)
#         plt.xlabel('Downtrend Performance (%)', fontsize=12)
#         plt.ylabel('Rebound Performance (%)', fontsize=12)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('performance_scatter_plot.png', dpi=300)
#         plt.show()
        
#         return scatter_data

#     def analyze_eth_downtrend_resilience(self, results, start_date, end_date, downtrend_threshold=-0.05, rebound_threshold=0.03, window_size=5, top_n=10):
    
#         print(f"\nAnalyzing cryptocurrencies that remain stable during ETH downtrend...")
        
#         # 1. Fetch historical data for ETH (using a 1-hour timeframe to capture more details)
#         eth_data = self.download_historical_data(self.eth_symbol, '1h', start_date, end_date)
        
#         # 2. Identify ETH downtrend periods
#         eth_data['pct_change'] = eth_data['close'].pct_change(window_size)
        
#         # Identify periods when ETH experienced a significant drop
#         downtrend_periods = []
#         current_downtrend = None
        
#         for idx, row in eth_data.iterrows():
#             if row['pct_change'] <= downtrend_threshold and current_downtrend is None:
#                 # Start a new downtrend period
#                 current_downtrend = {'start': idx, 'prices': [row['close']]}
#             elif row['pct_change'] <= downtrend_threshold and current_downtrend is not None:
#                 # Continue the current downtrend period
#                 current_downtrend['prices'].append(row['close'])
#             elif row['pct_change'] >= rebound_threshold and current_downtrend is not None:
#                 # End the downtrend period as ETH begins to rebound
#                 current_downtrend['end'] = idx
#                 current_downtrend['end_price'] = row['close']
#                 current_downtrend['start_price'] = current_downtrend['prices'][0]
#                 current_downtrend['lowest_price'] = min(current_downtrend['prices'])
#                 current_downtrend['drop_pct'] = (current_downtrend['lowest_price'] / current_downtrend['start_price'] - 1) * 100
#                 current_downtrend['rebound_pct'] = (current_downtrend['end_price'] / current_downtrend['lowest_price'] - 1) * 100
                
#                 downtrend_periods.append(current_downtrend)
#                 current_downtrend = None
        
#         # If the last downtrend period hasn't ended
#         if current_downtrend is not None:
#             current_downtrend['end'] = eth_data.index[-1]
#             current_downtrend['end_price'] = eth_data['close'].iloc[-1]
#             current_downtrend['start_price'] = current_downtrend['prices'][0]
#             current_downtrend['lowest_price'] = min(current_downtrend['prices'])
#             current_downtrend['drop_pct'] = (current_downtrend['lowest_price'] / current_downtrend['start_price'] - 1) * 100
#             current_downtrend['rebound_pct'] = (current_downtrend['end_price'] / current_downtrend['lowest_price'] - 1) * 100
            
#             downtrend_periods.append(current_downtrend)
        
#         print(f"Found {len(downtrend_periods)} ETH downtrend periods")
        
#         if len(downtrend_periods) == 0:
#             print("No ETH downtrend periods meeting the criteria were found within the specified date range")
#             return None, None, None
        
#         # 3. Analyze negatively correlated assets during each downtrend period
#         # Get all USDT trading pairs
#         all_symbols = self.get_all_futures_symbols()
#         usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        
#         # Store the performance of each coin during the downtrend period
#         downtrend_performance = {}
#         rebound_performance = {}
        
#         # Store data for all coins to use in visualization
#         all_coin_data = {self.eth_symbol: eth_data}
        
#         for period in downtrend_periods:
#             period_start = period['start']
#             period_end = period['end']
#             period_lowest = eth_data[eth_data['close'] == period['lowest_price']].index[0]
            
#             print(f"\nAnalyzing downtrend period: {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}")
#             print(f"ETH drop: {period['drop_pct']:.2f}%, Rebound: {period['rebound_pct']:.2f}%")
            
#             # Analyze performance for each coin
#             for symbol in tqdm(usdt_symbols, desc="Analyzing coin performance"):
#                 if symbol == self.eth_symbol:
#                     continue
                
#                 try:
#                     # Fetch historical data for the coin
#                     coin_data = self.download_historical_data(symbol, '1h', 
#                                                             period_start.strftime('%Y-%m-%d'), 
#                                                             period_end.strftime('%Y-%m-%d'))
                    
#                     if coin_data.empty or len(coin_data) < 2:
#                         continue
                    
#                     # Store data for visualization
#                     if symbol not in all_coin_data:
#                         all_coin_data[symbol] = coin_data
                    
#                     # Calculate performance during the downtrend period
#                     try:
#                         # Get the prices at the corresponding time points
#                         start_price = coin_data.loc[:period_start].iloc[-1]['close']
#                         lowest_time_price = coin_data.loc[:period_lowest].iloc[-1]['close']
#                         end_price = coin_data.loc[:period_end].iloc[-1]['close']
                        
#                         # Calculate the percentage change during the downtrend period
#                         downtrend_pct = (lowest_time_price / start_price - 1) * 100
                        
#                         # Calculate the percentage change during the ETH rebound period
#                         rebound_pct = (end_price / lowest_time_price - 1) * 100
                        
#                         # Store the results
#                         if symbol not in downtrend_performance:
#                             downtrend_performance[symbol] = []
                        
#                         if symbol not in rebound_performance:
#                             rebound_performance[symbol] = []
                        
#                         downtrend_performance[symbol].append(downtrend_pct)
#                         rebound_performance[symbol].append(rebound_pct)
                        
#                     except (KeyError, IndexError) as e:
#                         # Time point mismatch, skip this coin
#                         continue
                        
#                 except Exception as e:
#                     print(f"Error analyzing {symbol}: {e}")
#                     continue
        
#         # 4. Calculate average performance
#         avg_downtrend = {}
#         avg_rebound = {}
        
#         for symbol in downtrend_performance:
#             if len(downtrend_performance[symbol]) > 0:
#                 avg_downtrend[symbol] = sum(downtrend_performance[symbol]) / len(downtrend_performance[symbol])
        
#         for symbol in rebound_performance:
#             if len(rebound_performance[symbol]) > 0:
#                 avg_rebound[symbol] = sum(rebound_performance[symbol]) / len(rebound_performance[symbol])
        
#         # 5. Identify the coins with the most stable performance during ETH downtrend (i.e., the smallest drop or even an increase)
#         stable_coins = pd.Series(avg_downtrend).sort_values(ascending=False)
        
#         # 6. Identify the coins with the best performance during ETH rebound
#         best_rebounders = pd.Series(avg_rebound).sort_values(ascending=False)
        
#         # 7. Identify the coins that are both stable and have a good rebound
#         combined_score = {}
        
#         for symbol in stable_coins.index:
#             if symbol in best_rebounders.index:
#                 # Combined score = downtrend performance + rebound performance
#                 combined_score[symbol] = stable_coins[symbol] + best_rebounders[symbol]
        
#         best_combined = pd.Series(combined_score).sort_values(ascending=False)
        
#         # Store the results for later use
#         self._last_stable_coins = stable_coins
#         self._last_rebounders = best_rebounders
#         self._last_combined = best_combined
        
#         # 8. Create a results DataFrame
#         result_df = pd.DataFrame({
#             'Downtrend Performance (%)': stable_coins,
#             'Rebound Performance (%)': best_rebounders,
#             'Combined Score': best_combined
#         })
        
#         # 9. Visualize the results
#         # 9.1 Top N coins with the best downtrend performance
#         plt.figure(figsize=(12, 8))
#         sns.barplot(x=stable_coins.head(top_n).values, y=stable_coins.head(top_n).index, palette='viridis')
#         plt.title(f'Top {top_n} Cryptocurrencies with the Most Stable Performance during ETH Downtrend')
#         plt.xlabel('Price Change Percentage (%)')
#         plt.ylabel('Coin')
#         plt.tight_layout()
#         plt.savefig('eth_downtrend_stable_coins.png', dpi=300)
#         plt.show()
        
#         # 9.2 Top N coins with the best performance during ETH rebound
#         plt.figure(figsize=(12, 8))
#         sns.barplot(x=best_rebounders.head(top_n).values, y=best_rebounders.head(top_n).index, palette='viridis')
#         plt.title(f'Top {top_n} Cryptocurrencies with the Best Performance during ETH Rebound')
#         plt.xlabel('Price Change Percentage (%)')
#         plt.ylabel('Coin')
#         plt.tight_layout()
#         plt.savefig('eth_rebound_best_coins.png', dpi=300)
#         plt.show()
        
#         # 9.3 Top N coins with the best combined performance during the ETH market cycle
#         plt.figure(figsize=(12, 8))
#         sns.barplot(x=best_combined.head(top_n).values, y=best_combined.head(top_n).index, palette='viridis')
#         plt.title(f'Top {top_n} Cryptocurrencies with the Best Combined Performance in the ETH Market Cycle')
#         plt.xlabel('Combined Score')
#         plt.ylabel('Coin')
#         plt.tight_layout()
#         plt.savefig('eth_cycle_best_coins.png', dpi=300)
#         plt.show()
        
#         # 9.4 Create scatter plot of downtrend vs rebound performance
#         scatter_data = self.create_scatter_plot(stable_coins, best_rebounders, top_n)
        
#         # 9.5 Visualize price movements with downtrend periods marked
#         self.visualize_price_movements(all_coin_data, downtrend_periods, '1h', top_n=5)
        
#         # 10. Save the results to a CSV file
#         result_df.to_csv('eth_market_cycle_analysis.csv')
        
#         print("\nAnalysis complete!")
#         print(f"Top {top_n} cryptocurrencies with the most stable performance during ETH downtrend:")
#         print(stable_coins.head(top_n))
        
#         print(f"\nTop {top_n} cryptocurrencies with the best performance during ETH rebound:")
#         print(best_rebounders.head(top_n))
        
#         print(f"\nTop {top_n} cryptocurrencies with the best combined performance in the ETH market cycle:")
#         print(best_combined.head(top_n))
        
#         return stable_coins, best_rebounders, best_combined

#     def analyze_extreme_drops(self, start_date, end_date, window_size=20, drop_threshold=-0.03, timeframe='1m', top_n=10):
#         """
#         Analyze cryptocurrency performance during extreme ETH price drops (e.g., flash crashes).
        
#         Args:
#             start_date (str): Start date in format 'YYYY-MM-DD'.
#             end_date (str): End date in format 'YYYY-MM-DD'.
#             window_size (int, optional): Window size for percentage change calculation. Defaults to 20.
#             drop_threshold (float, optional): Threshold for extreme drops. Defaults to -0.03 (-3%).
#             timeframe (str, optional): Timeframe for analysis. Defaults to '1m'.
#             top_n (int, optional): Number of top coins to show. Defaults to 10.
            
#         Returns:
#             tuple: Tuple containing (extreme_drop_periods, stable_coins, best_rebounders, best_combined).
#         """
#         print(f"\nAnalyzing cryptocurrencies during extreme ETH price drops...")
#         print(f"Using {timeframe} timeframe, {window_size} window size, {drop_threshold*100}% drop threshold")
        
#         # 1. Fetch historical data for ETH
#         eth_data = self.download_historical_data(self.eth_symbol, timeframe, start_date, end_date)
        
#         # 2. Identify extreme drop periods
#         eth_data['pct_change'] = eth_data['close'].pct_change(window_size)
        
#         # Find periods where ETH dropped significantly in a short time
#         extreme_drop_periods = []
        
#         # Group consecutive extreme drops
#         in_extreme_drop = False
#         current_period = None
        
#         for idx, row in eth_data.iterrows():
#             if row['pct_change'] <= drop_threshold and not in_extreme_drop:
#                 # Start a new extreme drop period
#                 in_extreme_drop = True
#                 current_period = {
#                     'start': idx,
#                     'start_price': row['close'],
#                     'prices': [row['close']],
#                     'pct_changes': [row['pct_change']]
#                 }
#             elif row['pct_change'] <= drop_threshold and in_extreme_drop:
#                 # Continue the current extreme drop period
#                 current_period['prices'].append(row['close'])
#                 current_period['pct_changes'].append(row['pct_change'])
#             elif row['pct_change'] > drop_threshold and in_extreme_drop:
#                 # End the extreme drop period
#                 # Only consider it if it lasted for at least 3 candles
#                 if len(current_period['prices']) >= 3:
#                     current_period['end'] = idx
#                     current_period['end_price'] = row['close']
#                     current_period['lowest_price'] = min(current_period['prices'])
#                     current_period['lowest_pct_change'] = min(current_period['pct_changes'])
#                     current_period['drop_pct'] = (current_period['lowest_price'] / current_period['start_price'] - 1) * 100
                    
#                     # Calculate rebound (next 20 candles after the drop)
#                     rebound_end_idx = min(len(eth_data), eth_data.index.get_loc(idx) + 20)
#                     if rebound_end_idx < len(eth_data):
#                         rebound_end_price = eth_data.iloc[rebound_end_idx]['close']
#                         current_period['rebound_pct'] = (rebound_end_price / current_period['lowest_price'] - 1) * 100
#                     else:
#                         current_period['rebound_pct'] = 0
                    
#                     extreme_drop_periods.append(current_period)
                
#                 in_extreme_drop = False
#                 current_period = None
        
#         # Handle the case where we're still in an extreme drop at the end of the data
#         if in_extreme_drop and current_period is not None and len(current_period['prices']) >= 3:
#             current_period['end'] = eth_data.index[-1]
#             current_period['end_price'] = eth_data['close'].iloc[-1]
#             current_period['lowest_price'] = min(current_period['prices'])
#             current_period['lowest_pct_change'] = min(current_period['pct_changes'])
#             current_period['drop_pct'] = (current_period['lowest_price'] / current_period['start_price'] - 1) * 100
#             current_period['rebound_pct'] = 0  # No rebound data available
            
#             extreme_drop_periods.append(current_period)
        
#         print(f"Found {len(extreme_drop_periods)} extreme ETH drop periods")
        
#         if len(extreme_drop_periods) == 0:
#             print("No extreme ETH drop periods meeting the criteria were found within the specified date range")
#             return None, None, None, None
        
#         # 3. Get all USDT trading pairs
#         all_symbols = self.get_all_futures_symbols()
#         usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        
#         # 4. Analyze performance during extreme drops
#         downtrend_performance = {}
#         rebound_performance = {}
        
#         # Store data for all coins to use in visualization
#         all_coin_data = {self.eth_symbol: eth_data}
        
#         for period_idx, period in enumerate(extreme_drop_periods):
#             period_start = period['start']
#             period_end = period['end']
            
#             print(f"\nAnalyzing extreme drop period {period_idx+1}: {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}")
#             print(f"ETH drop: {period['drop_pct']:.2f}%, Max change in window: {period['lowest_pct_change']*100:.2f}%, Rebound: {period['rebound_pct']:.2f}%")
            
#             # Analyze performance for each coin
#             for symbol in tqdm(usdt_symbols, desc=f"Analyzing coin performance for period {period_idx+1}"):
#                 if symbol == self.eth_symbol:
#                     continue
                
#                 try:
#                     # Fetch historical data for the coin
#                     coin_data = self.download_historical_data(symbol, timeframe, 
#                                                             period_start.strftime('%Y-%m-%d'), 
#                                                             period_end.strftime('%Y-%m-%d'))
                    
#                     if coin_data.empty or len(coin_data) < 2:
#                         continue
                    
#                     # Store data for visualization
#                     if symbol not in all_coin_data:
#                         all_coin_data[symbol] = coin_data
                    
#                     # Calculate performance during the extreme drop period
#                     try:
#                         # Find the corresponding time points
#                         start_idx = coin_data.index.get_loc(period_start, method='nearest')
#                         end_idx = coin_data.index.get_loc(period_end, method='nearest')
                        
#                         # Get prices
#                         start_price = coin_data.iloc[start_idx]['close']
#                         end_price = coin_data.iloc[end_idx]['close']
                        
#                         # Calculate the percentage change during the drop period
#                         drop_pct = (end_price / start_price - 1) * 100
                        
#                         # Calculate rebound performance (next 20 candles)
#                         rebound_end_idx = min(len(coin_data), end_idx + 20)
#                         if rebound_end_idx < len(coin_data) and rebound_end_idx > end_idx:
#                             rebound_end_price = coin_data.iloc[rebound_end_idx]['close']
#                             rebound_pct = (rebound_end_price / end_price - 1) * 100
#                         else:
#                             rebound_pct = 0
                        
#                         # Store the results
#                         if symbol not in downtrend_performance:
#                             downtrend_performance[symbol] = []
                        
#                         if symbol not in rebound_performance:
#                             rebound_performance[symbol] = []
                        
#                         downtrend_performance[symbol].append(drop_pct)
#                         rebound_performance[symbol].append(rebound_pct)
                        
#                     except (KeyError, IndexError) as e:
#                         # Time point mismatch, skip this coin
#                         continue
                        
#                 except Exception as e:
#                     print(f"Error analyzing {symbol}: {e}")
#                     continue
        
#         # 5. Calculate average performance
#         avg_downtrend = {}
#         avg_rebound = {}
        
#         for symbol in downtrend_performance:
#             if len(downtrend_performance[symbol]) > 0:
#                 avg_downtrend[symbol] = sum(downtrend_performance[symbol]) / len(downtrend_performance[symbol])
            
#         for symbol in rebound_performance:
#             if len(rebound_performance[symbol]) > 0:
#                 avg_rebound[symbol] = sum(rebound_performance[symbol]) / len(rebound_performance[symbol])
        
#         # 6. Identify the coins with the most stable performance during extreme drops
#         stable_coins = pd.Series(avg_downtrend).sort_values(ascending=False)
        
#         # 7. Identify the coins with the best performance during rebounds
#         best_rebounders = pd.Series(avg_rebound).sort_values(ascending=False)
        
#         # 8. Identify the coins that are both stable and have a good rebound
#         combined_score = {}
        
#         for symbol in stable_coins.index:
#             if symbol in best_rebounders.index:
#                 # Combined score = downtrend performance + rebound performance
#                 combined_score[symbol] = stable_coins[symbol] + best_rebounders[symbol]
        
#         best_combined = pd.Series(combined_score).sort_values(ascending=False)
        
#         # 9. Create a results DataFrame
#         result_df = pd.DataFrame({
#             'Extreme Drop Performance (%)': stable_coins,
#             'Post-Drop Rebound (%)': best_rebounders,
#             'Combined Score': best_combined
#         })
        
#         # 10. Visualize the results
#         # 檢查是否有足夠的數據來繪圖
#         if not stable_coins.empty and len(stable_coins) >= 1:
#             # 10.1 Top N coins with the best performance during extreme drops
#             plt.figure(figsize=(12, 8))
#             sns.barplot(x=stable_coins.head(min(top_n, len(stable_coins))).values, 
#                        y=stable_coins.head(min(top_n, len(stable_coins))).index, 
#                        palette='viridis')
#             plt.title(f'Top {min(top_n, len(stable_coins))} Cryptocurrencies with the Most Stable Performance during Extreme ETH Drops')
#             plt.xlabel('Price Change Percentage (%)')
#             plt.ylabel('Symbol')
#             plt.tight_layout()
#             plt.savefig('extreme_drop_stable_coins.png', dpi=300)
#             plt.show()
#         else:
#             print("Not enough data to create stable coins performance chart")
        
#         if not best_rebounders.empty and len(best_rebounders) >= 1:
#             # 10.2 Top N coins with the best rebound performance
#             plt.figure(figsize=(12, 8))
#             sns.barplot(x=best_rebounders.head(min(top_n, len(best_rebounders))).values, 
#                        y=best_rebounders.head(min(top_n, len(best_rebounders))).index, 
#                        palette='viridis')
#             plt.title(f'Top {min(top_n, len(best_rebounders))} Cryptocurrencies with the Best Rebound after Extreme ETH Drops')
#             plt.xlabel('Price Change Percentage (%)')
#             plt.ylabel('Symbol')
#             plt.tight_layout()
#             plt.savefig('extreme_drop_rebounders.png', dpi=300)
#             plt.show()
#         else:
#             print("Not enough data to create rebound performance chart")
        
#         if not best_combined.empty and len(best_combined) >= 1:
#             # 10.3 Top N coins with the best combined performance
#             plt.figure(figsize=(12, 8))
#             sns.barplot(x=best_combined.head(min(top_n, len(best_combined))).values, 
#                        y=best_combined.head(min(top_n, len(best_combined))).index, 
#                        palette='viridis')
#             plt.title(f'Top {min(top_n, len(best_combined))} Cryptocurrencies with the Best Combined Performance during Extreme ETH Drops')
#             plt.xlabel('Combined Score')
#             plt.ylabel('Symbol')
#             plt.tight_layout()
#             plt.savefig('extreme_drop_combined.png', dpi=300)
#             plt.show()
#         else:
#             print("Not enough data to create combined performance chart")
        
#         # 10.4 Create scatter plot if we have enough data
#         if not stable_coins.empty and not best_rebounders.empty and len(stable_coins) >= 1 and len(best_rebounders) >= 1:
#             scatter_data = self.create_scatter_plot(stable_coins, best_rebounders, min(top_n, len(stable_coins)))
#         else:
#             print("Not enough data to create scatter plot")
        
#         # 10.5 Visualize price movements with extreme drop periods marked
#         if len(all_coin_data) > 1:  # 確保至少有ETH和一個其他幣種
#             self.visualize_price_movements(all_coin_data, extreme_drop_periods, timeframe, top_n=min(5, len(stable_coins) if not stable_coins.empty else 0))
#         else:
#             print("Not enough data to visualize price movements")
        
#         # 11. Save the results to a CSV file if we have data
#         if not result_df.empty:
#             result_df.to_csv('extreme_drop_analysis.csv')
            
#             print("\nExtreme drop analysis complete!")
#             if not stable_coins.empty and len(stable_coins) > 0:
#                 print(f"Top {min(top_n, len(stable_coins))} cryptocurrencies with the most stable performance during extreme ETH drops:")
#                 print(stable_coins.head(min(top_n, len(stable_coins))))
            
#             if not best_rebounders.empty and len(best_rebounders) > 0:
#                 print(f"\nTop {min(top_n, len(best_rebounders))} cryptocurrencies with the best rebound after extreme ETH drops:")
#                 print(best_rebounders.head(min(top_n, len(best_rebounders))))
            
#             if not best_combined.empty and len(best_combined) > 0:
#                 print(f"\nTop {min(top_n, len(best_combined))} cryptocurrencies with the best combined performance:")
#                 print(best_combined.head(min(top_n, len(best_combined))))
#         else:
#             print("No valid results to save")
        
#         return extreme_drop_periods, stable_coins, best_rebounders, best_combined

#     def multi_timeframe_analysis(self, start_date, end_date, timeframes=None, window_sizes=None, drop_thresholds=None, top_n=10):
#         """
#         Perform analysis across multiple timeframes and parameters.
        
#         Args:
#             start_date (str): Start date in format 'YYYY-MM-DD'.
#             end_date (str): End date in format 'YYYY-MM-DD'.
#             timeframes (list, optional): List of timeframes to analyze. Defaults to ['1m', '5m', '15m', '1h'].
#             window_sizes (list, optional): List of window sizes to use. Defaults to [5, 10, 20].
#             drop_thresholds (list, optional): List of drop thresholds to use. Defaults to [-0.01, -0.03, -0.05].
#             top_n (int, optional): Number of top coins to show. Defaults to 10.
            
#         Returns:
#             dict: Dictionary containing analysis results for each configuration.
#         """
#         if timeframes is None:
#             timeframes = ['1m', '5m', '15m', '1h']
            
#         if window_sizes is None:
#             window_sizes = [5, 10, 20]
            
#         if drop_thresholds is None:
#             drop_thresholds = [-0.01, -0.03, -0.05]
        
#         print(f"\nPerforming multi-timeframe analysis...")
#         print(f"Timeframes: {timeframes}")
#         print(f"Window sizes: {window_sizes}")
#         print(f"Drop thresholds: {[f'{t*100}%' for t in drop_thresholds]}")
        
#         # Store results for each configuration
#         results = {}
        
#         # Create a summary DataFrame to track best performers across configurations
#         summary_data = []
        
#         # Analyze each configuration
#         for timeframe in timeframes:
#             for window_size in window_sizes:
#                 for drop_threshold in drop_thresholds:
#                     config_name = f"{timeframe}_w{window_size}_d{abs(drop_threshold*100)}"
#                     print(f"\n{'='*80}")
#                     print(f"Analyzing configuration: {config_name}")
#                     print(f"Timeframe: {timeframe}, Window size: {window_size}, Drop threshold: {drop_threshold*100}%")
#                     print(f"{'='*80}")
                    
#                     # Perform extreme drop analysis
#                     extreme_drop_periods, stable_coins, best_rebounders, best_combined = self.analyze_extreme_drops(
#                         start_date, end_date, window_size, drop_threshold, timeframe, top_n
#                     )
                    
#                     if extreme_drop_periods is not None:
#                         # Store results
#                         results[config_name] = {
#                             'extreme_drop_periods': extreme_drop_periods,
#                             'stable_coins': stable_coins,
#                             'best_rebounders': best_rebounders,
#                             'best_combined': best_combined,
#                             'config': {
#                                 'timeframe': timeframe,
#                                 'window_size': window_size,
#                                 'drop_threshold': drop_threshold
#                             }
#                         }
                        
#                         # Add top performers to summary if data is available
#                         if best_combined is not None and not best_combined.empty:
#                             for symbol in best_combined.head(min(top_n, len(best_combined))).index:
#                                 summary_data.append({
#                                     'Symbol': symbol,
#                                     'Configuration': config_name,
#                                     'Timeframe': timeframe,
#                                     'Window Size': window_size,
#                                     'Drop Threshold (%)': drop_threshold * 100,
#                                     'Downtrend Performance (%)': stable_coins.get(symbol, 0) if stable_coins is not None else 0,
#                                     'Rebound Performance (%)': best_rebounders.get(symbol, 0) if best_rebounders is not None else 0,
#                                     'Combined Score': best_combined.get(symbol, 0) if best_combined is not None else 0
#                                 })
        
#         # Create summary DataFrame
#         if summary_data:
#             summary_df = pd.DataFrame(summary_data)
            
#             # Calculate frequency of appearance in top performers
#             symbol_counts = summary_df['Symbol'].value_counts()
            
#             # Calculate average scores across configurations
#             avg_scores = summary_df.groupby('Symbol').agg({
#                 'Downtrend Performance (%)': 'mean',
#                 'Rebound Performance (%)': 'mean',
#                 'Combined Score': 'mean'
#             })
            
#             # Add frequency to the DataFrame
#             avg_scores['Frequency'] = symbol_counts
            
#             # Sort by frequency and then by average combined score
#             avg_scores = avg_scores.sort_values(['Frequency', 'Combined Score'], ascending=False)
            
#             # Save summary to CSV
#             summary_df.to_csv('multi_timeframe_analysis_details.csv', index=False)
#             avg_scores.to_csv('multi_timeframe_analysis_summary.csv')
            
#             # Visualize top performers across configurations if we have enough data
#             if len(avg_scores) > 0:
#                 plt.figure(figsize=(14, 10))
                
#                 # Plot top symbols by frequency (up to 20, but not more than we have)
#                 top_symbols = avg_scores.head(min(20, len(avg_scores))).index
                
#                 # Create a bar plot for frequency
#                 ax1 = plt.subplot(2, 1, 1)
#                 sns.barplot(x=avg_scores.loc[top_symbols, 'Frequency'], y=top_symbols, palette='viridis', ax=ax1)
#                 ax1.set_title('Frequency of Appearance in Top Performers Across Configurations', fontsize=14)
#                 ax1.set_xlabel('Frequency', fontsize=12)
#                 ax1.set_ylabel('Symbol', fontsize=12)
                
#                 # Create a bar plot for average combined score
#                 ax2 = plt.subplot(2, 1, 2)
#                 sns.barplot(x=avg_scores.loc[top_symbols, 'Combined Score'], y=top_symbols, palette='viridis', ax=ax2)
#                 ax2.set_title('Average Combined Score Across Configurations', fontsize=14)
#                 ax2.set_xlabel('Average Combined Score', fontsize=12)
#                 ax2.set_ylabel('Symbol', fontsize=12)
                
#                 plt.tight_layout()
#                 plt.savefig('multi_timeframe_analysis_summary.png', dpi=300)
#                 plt.show()
                
#                 print("\nMulti-timeframe analysis complete!")
#                 print(f"Top performers across all configurations:")
#                 print(avg_scores.head(min(10, len(avg_scores))))
                
#                 return results, summary_df, avg_scores
#             else:
#                 print("\nMulti-timeframe analysis complete, but no consistent top performers were found.")
#                 return results, summary_df, None
#         else:
#             print("\nMulti-timeframe analysis complete, but no valid results were found across any configuration.")
#             return results, None, None

#     def track_api_usage(self, show_details=True):
#         """
#         Track and display current API usage statistics.
        
#         Args:
#             show_details (bool): Whether to print detailed usage information
            
#         Returns:
#             dict: Dictionary containing API usage statistics
#         """
#         current_time = time.time()
#         time_until_reset = max(0, self.weight_reset_time - current_time)
        
#         usage_stats = {
#             'weight_used': self.weight_used,
#             'weight_limit': self.MAX_WEIGHT_PER_MINUTE,
#             'weight_remaining': self.MAX_WEIGHT_PER_MINUTE - self.weight_used,
#             'weight_usage_percent': (self.weight_used / self.MAX_WEIGHT_PER_MINUTE) * 100,
#             'time_until_reset': time_until_reset,
#             'requests_this_second': self.requests_this_second,
#             'requests_per_second_limit': self.MAX_REQUESTS_PER_SECOND
#         }
        
#         if show_details:
#             print("\n=== API Usage Statistics ===")
#             print(f"Weight used: {self.weight_used}/{self.MAX_WEIGHT_PER_MINUTE} ({usage_stats['weight_usage_percent']:.1f}%)")
#             print(f"Time until weight reset: {time_until_reset:.1f} seconds")
#             print(f"Requests this second: {self.requests_this_second}/{self.MAX_REQUESTS_PER_SECOND}")
#             print("===========================\n")
            
#         return usage_stats
    
#     def optimize_request_parameters(self, days, timeframe, use_cache=True):
#         """
#         Optimize request parameters based on the analysis requirements.
        
#         Args:
#             days (int): Number of days to analyze
#             timeframe (str): Timeframe for analysis
#             use_cache (bool): Whether to use cached data
            
#         Returns:
#             dict: Dictionary containing optimized parameters
#         """
#         # Calculate optimal max_klines based on timeframe and days
#         timeframe_minutes = {
#             '1m': 1,
#             '5m': 5,
#             '15m': 15,
#             '1h': 60,
#             '4h': 240
#         }
        
#         # Calculate total minutes needed
#         total_minutes = days * 24 * 60
        
#         # Calculate how many candles we need
#         candles_needed = total_minutes / timeframe_minutes.get(timeframe, 60)
        
#         # Determine optimal max_klines (stay within API limits)
#         if candles_needed <= 100:
#             optimal_max_klines = 100  # Weight: 1
#         elif candles_needed <= 500:
#             optimal_max_klines = min(500, candles_needed)  # Weight: 2
#         elif candles_needed <= 1000:
#             optimal_max_klines = min(1000, candles_needed)  # Weight: 5
#         else:
#             # For very large requests, we'll need to make multiple calls
#             # Limit to 1000 to keep weight at 5
#             optimal_max_klines = 1000
        
#         # Determine optimal request delay based on the number of symbols
#         # and whether we're using cache
#         if use_cache:
#             # If using cache, we can be more aggressive
#             optimal_request_delay = 0.2
#         else:
#             # If not using cache, be more conservative
#             optimal_request_delay = 0.5
        
#         # Determine optimal max_workers based on system resources
#         cpu_count = os.cpu_count() or 4
#         optimal_max_workers = min(10, max(2, cpu_count - 1))
        
#         optimized_params = {
#             'max_klines': int(optimal_max_klines),
#             'request_delay': optimal_request_delay,
#             'max_workers': optimal_max_workers
#         }
        
#         print("\n=== Optimized Request Parameters ===")
#         print(f"Max K-lines: {optimized_params['max_klines']} (based on {days} days of {timeframe} data)")
#         print(f"Request delay: {optimized_params['request_delay']} seconds")
#         print(f"Max workers: {optimized_params['max_workers']}")
#         print("====================================\n")
        
#         return optimized_params


# def main():
#     # Set your Binance API credentials (optional for historical data)
#     api_key = None  # Replace with your API key if needed
#     api_secret = None  # Replace with your API secret if needed
    
#     # Set up command line argument parsing
#     parser = argparse.ArgumentParser(description='Cryptocurrency Correlation Analysis Tool')
#     parser.add_argument('--api_key', type=str, default=None, help='Binance API key')
#     parser.add_argument('--api_secret', type=str, default=None, help='Binance API secret')
#     parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
#     parser.add_argument('--max_klines', type=int, default=200, help='Maximum number of K-lines to fetch per timeframe (default: 200)')
#     parser.add_argument('--use_cache', action='store_true', help='Use cached data if available')
#     parser.add_argument('--request_delay', type=float, default=1.0, help='Delay between API requests in seconds (default: 1.0)')
#     parser.add_argument('--max_workers', type=int, default=5, help='Maximum number of concurrent workers (default: 5)')
#     parser.add_argument('--cache_expiry', type=int, default=86400, help='Cache expiry time in seconds (default: 86400)')
#     parser.add_argument('--top_n', type=int, default=20, help='Show top N most correlated coins (default: 20)')
    
#     # Add new parameters for extreme drop analysis
#     parser.add_argument('--window_size', type=int, default=20, help='Window size for percentage change calculation (default: 20)')
#     parser.add_argument('--drop_threshold', type=float, default=-0.03, help='Threshold for extreme drops, e.g., -0.03 means -3% (default: -0.03)')
#     parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe for extreme drop analysis (default: 1m)')
    
#     # Add parameter for multi-timeframe analysis
#     parser.add_argument('--multi_timeframe', action='store_true', help='Perform multi-timeframe analysis')
    
#     # Add new parameters for API optimization
#     parser.add_argument('--optimize', action='store_true', help='Automatically optimize request parameters')
#     parser.add_argument('--track_api', action='store_true', help='Track and display API usage statistics')
#     parser.add_argument('--safe_mode', action='store_true', help='Run in safe mode with conservative rate limits')
#     parser.add_argument('--max_symbols', type=int, default=None, help='Maximum number of symbols to analyze (default: all)')
    
#     args = parser.parse_args()
    
#     # Use command line arguments
#     api_key = args.api_key
#     api_secret = args.api_secret
    
#     # Auto-optimize parameters if requested
#     if args.optimize:
#         analyzer_temp = CryptoCorrelationAnalyzer()  # Temporary instance for optimization
#         optimized_params = analyzer_temp.optimize_request_parameters(args.days, args.timeframe, args.use_cache)
        
#         # Use optimized parameters unless explicitly overridden
#         max_klines = args.max_klines if args.max_klines != 200 else optimized_params['max_klines']
#         request_delay = args.request_delay if args.request_delay != 1.0 else optimized_params['request_delay']
#         max_workers = args.max_workers if args.max_workers != 5 else optimized_params['max_workers']
#     else:
#         max_klines = args.max_klines
#         request_delay = args.request_delay
#         max_workers = args.max_workers
    
#     # Apply safe mode if requested
#     if args.safe_mode:
#         print("Running in safe mode with conservative rate limits")
#         request_delay = max(request_delay, 1.0)  # Minimum 1 second delay
#         max_workers = min(max_workers, 3)  # Maximum 3 workers
    
#     # Initialize the analyzer with optimized settings
#     analyzer = CryptoCorrelationAnalyzer(
#         api_key=api_key, 
#         api_secret=api_secret, 
#         max_klines=max_klines,
#         request_delay=request_delay,
#         max_workers=max_workers,
#         cache_expiry=args.cache_expiry
#     )
    
#     # Set date range based on command line args
#     current_date = datetime.now()
#     end_date = current_date.strftime('%Y-%m-%d')
#     start_date = (current_date - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
#     # 檢查日期是否有效（不能是未來日期）
#     if datetime.strptime(end_date, '%Y-%m-%d') > current_date:
#         end_date = current_date.strftime('%Y-%m-%d')
    
#     if datetime.strptime(start_date, '%Y-%m-%d') > current_date:
#         # 如果開始日期也是未來日期，則使用當前日期減去指定天數
#         start_date = (current_date - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
#     print(f"Analysis date range: {start_date} to {end_date}")
#     print(f"Maximum K-lines: {max_klines}")
#     print(f"API request delay: {request_delay} seconds")
#     print(f"Maximum concurrent threads: {max_workers}")
#     print(f"Cache expiry time: {args.cache_expiry} seconds")
    
#     # Get all symbols
#     all_symbols = analyzer.get_all_futures_symbols()
#     usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
#     if analyzer.eth_symbol not in usdt_symbols:
#         usdt_symbols.append(analyzer.eth_symbol)
    
#     # Limit number of symbols if requested
#     if args.max_symbols is not None and args.max_symbols > 0:
#         # Always include ETH
#         if analyzer.eth_symbol in usdt_symbols:
#             usdt_symbols.remove(analyzer.eth_symbol)
#             limited_symbols = [analyzer.eth_symbol] + usdt_symbols[:args.max_symbols-1]
#         else:
#             limited_symbols = usdt_symbols[:args.max_symbols]
        
#         print(f"Limiting analysis to {len(limited_symbols)} symbols (out of {len(usdt_symbols)} available)")
#         usdt_symbols = limited_symbols
    
#     # Check if multi-timeframe analysis is requested
#     if args.multi_timeframe:
#         print("\nPerforming multi-timeframe analysis...")
#         results, summary_df, avg_scores = analyzer.multi_timeframe_analysis(
#             start_date, end_date, 
#             timeframes=['1m', '5m', '15m', '1h'],
#             window_sizes=[5, 10, 20],
#             drop_thresholds=[-0.01, -0.03, -0.05],
#             top_n=args.top_n
#         )
        
#         # Track API usage if requested
#         if args.track_api:
#             analyzer.track_api_usage()
            
#         return
    
#     # Choose analysis mode
#     print("\nSelect analysis mode:")
#     print("1. Standard correlation analysis")
#     print("2. ETH downtrend stable coins analysis")
#     print("3. Extreme ETH drop analysis")
#     print("4. Run all analyses")
    
#     choice = input("Enter option (1/2/3/4): ").strip()
    
#     # Standard correlation analysis
#     if choice in ['1', '4']:
#         # Analyze all timeframes
#         results = analyzer.analyze_all_timeframes(start_date, end_date, top_n=args.top_n, use_cache=args.use_cache)
        
#         # Create correlation heatmap
#         heatmap_data_positive, heatmap_data_negative = analyzer.create_correlation_heatmap(results, top_n=args.top_n)
        
#         print("Standard correlation analysis complete!")
#         print(f"Positive correlation heatmap saved as 'correlation_heatmap_positive.png'")
#         print(f"Negative correlation heatmap saved as 'correlation_heatmap_negative.png'")
#         print(f"Original correlation heatmap saved as 'correlation_heatmap.png'")
        
#         # Track API usage if requested
#         if args.track_api:
#             analyzer.track_api_usage()
    
#     # ETH downtrend analysis
#     if choice in ['2', '4']:
#         # Set a longer time range for ETH downtrend analysis
#         long_start_date = (current_date - timedelta(days=90)).strftime('%Y-%m-%d')
        
#         print(f"\nAnalyzing stable coins during ETH downtrends (date range: {long_start_date} to {end_date})")
        
#         # If standard analysis was already performed, use its results; otherwise perform a quick analysis
#         if choice == '4' and 'results' in locals():
#             # Use existing results
#             stable_coins, best_rebounders, best_combined = analyzer.analyze_eth_downtrend_resilience(
#                 results, long_start_date, end_date, 
#                 downtrend_threshold=-0.05,  # 5% ETH drop is considered a downtrend
#                 rebound_threshold=0.03,     # 3% ETH increase is considered a rebound
#                 window_size=5,              # 5-hour window
#                 top_n=args.top_n            # Show top N coins
#             )
#         else:
#             # Perform quick analysis to get results
#             quick_results = analyzer.analyze_all_timeframes(long_start_date, end_date, top_n=args.top_n, use_cache=args.use_cache)
#             stable_coins, best_rebounders, best_combined = analyzer.analyze_eth_downtrend_resilience(
#                 quick_results, long_start_date, end_date, 
#                 downtrend_threshold=-0.05,
#                 rebound_threshold=0.03,
#                 window_size=5,
#                 top_n=args.top_n
#             )
        
#         print("ETH downtrend stable coins analysis complete!")
#         print("Results saved to 'eth_market_cycle_analysis.csv'")
        
#         # Track API usage if requested
#         if args.track_api:
#             analyzer.track_api_usage()
    
#     # Extreme ETH drop analysis
#     if choice in ['3', '4']:
#         print(f"\nAnalyzing cryptocurrencies during extreme ETH price drops...")
#         extreme_drop_periods, stable_coins, best_rebounders, best_combined = analyzer.analyze_extreme_drops(
#             start_date, end_date,
#             window_size=args.window_size,
#             drop_threshold=args.drop_threshold,
#             timeframe=args.timeframe,
#             top_n=args.top_n
#         )
        
#         print("Extreme ETH drop analysis complete!")
#         print("Results saved to 'extreme_drop_analysis.csv'")
        
#         # Track API usage if requested
#         if args.track_api:
#             analyzer.track_api_usage()
    
#     print("\nAll analyses complete!")
    
#     # Final API usage report
#     if args.track_api:
#         analyzer.track_api_usage()


# if __name__ == "__main__":
#     main()
