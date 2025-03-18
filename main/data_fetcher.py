#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Fetcher Module
Responsible for fetching historical data from Binance
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from binance.client import Client

import config
from utils import RateLimiter, CacheManager, ProxyManager, calculate_optimal_batch_size


class DataFetcher:
    """Data fetcher responsible for retrieving historical data from Binance"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 max_klines: int = config.DEFAULT_MAX_KLINES,
                 request_delay: float = config.DEFAULT_REQUEST_DELAY, 
                 max_workers: int = config.DEFAULT_MAX_WORKERS, 
                 use_proxies: bool = False, 
                 proxies: List[str] = None, 
                 cache_expiry: int = config.DEFAULT_CACHE_EXPIRY):
        """
        Initialize the data fetcher
        
        Args:
            api_key (str, optional): Binance API key
            api_secret (str, optional): Binance API secret
            max_klines (int, optional): Maximum number of klines to fetch
            request_delay (float, optional): Delay between requests (seconds)
            max_workers (int, optional): Maximum number of concurrent worker threads
            use_proxies (bool, optional): Whether to use proxy rotation
            proxies (List[str], optional): List of proxy URLs
            cache_expiry (int, optional): Cache expiry time (seconds)
        """
        self.client = Client(api_key, api_secret)
        self.max_klines = max(100, min(1500, max_klines))  # Ensure max_klines is between 100-1500
        
        # Initialize helper managers
        self.rate_limiter = RateLimiter(request_delay)
        self.cache_manager = CacheManager(config.CACHE_DIR, cache_expiry)
        self.proxy_manager = ProxyManager(proxies, use_proxies)
        
        # Concurrency settings
        self.max_workers = max_workers
        
        # Data directory
        self.data_dir = config.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_all_futures_symbols(self) -> List[str]:
        """
        Get all available futures trading pairs
        
        Returns:
            List[str]: List of futures trading pairs
        """
        try:
            self.rate_limiter.wait(weight=10)  # Higher weight for exchange info
            exchange_info = self.client.futures_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            return symbols
        except Exception as e:
            print(f"Error getting futures trading pairs: {e}")
            # Return a set of popular trading pairs as fallback
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
                    'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT']
    
    def download_historical_data(self, symbol: str, timeframe: str, 
                                start_date: str, end_date: Optional[str] = None, 
                                retry_count: int = config.MAX_RETRY_COUNT) -> pd.DataFrame:
        """
        Download historical kline data for a trading pair and timeframe
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            timeframe (str): Kline timeframe (e.g., '1m', '1h')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            retry_count (int, optional): Number of retries on failure
            
        Returns:
            pd.DataFrame: DataFrame containing historical price data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 检查时间范围是否合理
        days_diff = (end_dt - start_dt).days
        if days_diff <= 0:
            raise ValueError(f"End date ({end_dt}) must be after start date ({start_dt})")
            
        # 将截止日期调整为当天结束时间
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
        
        print(f"Downloading data for {symbol} ({timeframe}) from {start_dt} to {end_dt} ({days_diff} days)")
        
        # Check cache first for the full date range
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        cache_key = self.cache_manager.get_cache_key(symbol, timeframe, start_ts, end_ts, 0)  # 0 means no limit
        cached_data = self.cache_manager.load(cache_key)
        if cached_data is not None:
            print(f"Using cached data for {symbol} ({timeframe}) from {start_date} to {end_date}")
            return cached_data
        
        # For long date ranges, we need to fetch data in chunks
        # Calculate the maximum time range we can fetch in one request based on max_klines
        minutes_per_chunk = self.max_klines * config.TIMEFRAME_MINUTES.get(timeframe, 60)
        
        # 对于较长的时间范围，调整每个块的大小以适应API限制
        if days_diff > 30 and timeframe == '1m':
            # 对于1分钟数据，每个块最多获取8小时数据（480分钟）
            minutes_per_chunk = min(minutes_per_chunk, 480)
            print(f"Adjusted chunk size to {minutes_per_chunk} minutes for long-range 1m data")
        elif days_diff > 60 and timeframe == '5m':
            # 对于5分钟数据，每个块最多获取1天数据（1440分钟）
            minutes_per_chunk = min(minutes_per_chunk, 1440)
            print(f"Adjusted chunk size to {minutes_per_chunk} minutes for long-range 5m data")
        
        # Convert to timedelta
        chunk_delta = timedelta(minutes=minutes_per_chunk)
        
        # Initialize empty DataFrame to store all data
        all_data = pd.DataFrame()
        
        # Calculate number of chunks needed
        total_minutes = (end_dt - start_dt).total_seconds() / 60
        num_chunks = max(1, int(total_minutes / minutes_per_chunk) + 1)
        
        print(f"Fetching data for {symbol} ({timeframe}) from {start_date} to {end_date}")
        print(f"Total time range: {total_minutes:.1f} minutes, will fetch in {num_chunks} chunks")
        
        # Fetch data in chunks
        current_start = start_dt
        chunk_num = 1
        max_chunks_without_progress = 5  # 如果连续多个块没有新数据，则提前退出
        chunks_without_progress = 0
        
        # 使用tqdm进度条显示进度
        pbar = tqdm(total=num_chunks, desc=f"Fetching {symbol} {timeframe}")
        
        while current_start < end_dt:
            # Calculate end of this chunk
            current_end = min(current_start + chunk_delta, end_dt)
            
            # Convert to timestamps
            current_start_ts = int(current_start.timestamp() * 1000)
            current_end_ts = int(current_end.timestamp() * 1000)
            
            if chunk_num % 10 == 0 or chunk_num == 1:
                print(f"Fetching chunk {chunk_num}/{num_chunks}: {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
            
            # Calculate request weight based on limit parameter
            request_weight = 1  # Default weight
            if self.max_klines > 100:
                if self.max_klines <= 500:
                    request_weight = 2
                elif self.max_klines <= 1000:
                    request_weight = 5
                else:
                    request_weight = 10
            
            # Try to get data with retries
            chunk_data = pd.DataFrame()
            for attempt in range(retry_count):
                try:
                    # Wait for rate limit, using appropriate weight
                    self.rate_limiter.wait(weight=request_weight)
                    
                    # Get data from Binance, using limit parameter
                    klines = self.client.futures_historical_klines(
                        symbol=symbol,
                        interval=config.TIMEFRAMES[timeframe],
                        start_str=current_start_ts,
                        end_str=current_end_ts,
                        limit=self.max_klines
                    )
                    
                    # Create DataFrame
                    if klines:
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
                        
                        # 检查是否获取了新数据
                        if len(df) > 0:
                            chunk_data = df
                            chunks_without_progress = 0
                        else:
                            chunks_without_progress += 1
                            if chunks_without_progress >= max_chunks_without_progress:
                                print(f"No new data in {max_chunks_without_progress} consecutive chunks, skipping forward...")
                                # 跳过一段时间以加速处理
                                current_start = current_start + chunk_delta * 5
                                chunks_without_progress = 0
                        break
                    else:
                        chunks_without_progress += 1
                        if chunks_without_progress >= max_chunks_without_progress:
                            print(f"No data in {max_chunks_without_progress} consecutive chunks, skipping forward...")
                            # 跳过一段时间以加速处理
                            current_start = current_start + chunk_delta * 5
                            chunks_without_progress = 0
                        break
                        
                except Exception as e:
                    print(f"Error getting data for {symbol} on {timeframe} (attempt {attempt+1}/{retry_count}): {e}")
                    
                    # Exponential backoff
                    if attempt < retry_count - 1:
                        backoff_time = (2 ** attempt) * self.rate_limiter.request_delay
                        print(f"Retrying in {backoff_time:.2f} seconds...")
                        time.sleep(backoff_time)
            
            # Append chunk data to all_data
            if not chunk_data.empty:
                all_data = pd.concat([all_data, chunk_data])
                if chunk_num % 10 == 0:
                    print(f"Downloaded {len(all_data)} data points so far")
            
            # Move to next chunk
            current_start = current_end
            chunk_num += 1
            pbar.update(1)
            
            # Add small delay between chunks
            time.sleep(0.5)
        
        # 关闭进度条
        pbar.close()
        
        # Remove duplicates if any
        if not all_data.empty:
            # 检查数据覆盖情况
            start_date_data = all_data.index.min().strftime('%Y-%m-%d')
            end_date_data = all_data.index.max().strftime('%Y-%m-%d')
            data_days = (all_data.index.max() - all_data.index.min()).days
            
            print(f"Data covers from {start_date_data} to {end_date_data} ({data_days} days)")
            
            # 检查是否有大的时间间隙
            time_diffs = all_data.index.to_series().diff()
            max_gap = time_diffs.max().total_seconds() / 60  # 最大间隙（分钟）
            avg_gap = time_diffs.mean().total_seconds() / 60  # 平均间隙（分钟）
            
            if max_gap > config.TIMEFRAME_MINUTES.get(timeframe, 60) * 5:
                print(f"Warning: Found large time gap in data: {max_gap:.0f} minutes (avg: {avg_gap:.1f} min)")
                
            # 删除重复数据并排序
            duplicates_count = all_data.index.duplicated().sum()
            if duplicates_count > 0:
                print(f"Removing {duplicates_count} duplicate entries")
            
            all_data = all_data[~all_data.index.duplicated(keep='first')]
            all_data.sort_index(inplace=True)
            
            # Save to cache
            self.cache_manager.save(cache_key, all_data)
            
            print(f"Successfully fetched {len(all_data)} candles for {symbol} ({timeframe})")
            return all_data
        else:
            print(f"Failed to get data for {symbol} on {timeframe}")
            return pd.DataFrame()
    
    def save_data_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Save DataFrame to CSV file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            symbol (str): Trading pair symbol
            timeframe (str): Kline timeframe
        """
        filename = f"{self.data_dir}/{symbol}_{timeframe}.csv"
        df.to_csv(filename)
        print(f"Saved {symbol} {timeframe} data to {filename}")
    
    def load_data_from_csv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from CSV file
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Kline timeframe
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing historical price data, or None if file doesn't exist
        """
        filename = f"{self.data_dir}/{symbol}_{timeframe}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        return None
    
    async def _fetch_single_symbol(self, symbol: str, timeframe: str, 
                                  start_date: str, end_date: Optional[str], 
                                  use_cache: bool) -> Tuple[str, Optional[pd.DataFrame]]:
        """Asynchronously fetch data for a single symbol"""
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
            print(f"Error fetching data for {symbol} on {timeframe}: {e}")
        return symbol, None

    async def fetch_data_for_all_symbols_async(self, symbols: List[str], timeframe: str, 
                                             start_date: str, end_date: Optional[str] = None, 
                                             use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols in parallel using asyncio"""
        data = {}
        remaining_symbols = symbols.copy()
        
        while remaining_symbols:
            # Calculate optimal batch size based on current rate limit usage
            batch_size = calculate_optimal_batch_size(
                self.rate_limiter, 
                len(remaining_symbols), 
                timeframe, 
                self.max_klines, 
                self.max_workers
            )
            
            # Get next batch of symbols
            batch = remaining_symbols[:batch_size]
            remaining_symbols = remaining_symbols[batch_size:]
            
            print(f"Processing batch of {len(batch)} symbols (remaining: {len(remaining_symbols)})")
            
            # Process batch
            batch_tasks = []
            for symbol in batch:
                if use_cache:
                    df = self.load_data_from_csv(symbol, timeframe)
                    if df is not None:
                        data[symbol] = df
                        continue
                
                task = asyncio.create_task(self._fetch_single_symbol(symbol, timeframe, start_date, end_date, use_cache))
                batch_tasks.append(task)
            
            if batch_tasks:
                for coro in tqdm(asyncio.as_completed(batch_tasks), total=len(batch_tasks), desc=f"Fetching {timeframe} data (batch)"):
                    symbol, df = await coro
                    if df is not None:
                        data[symbol] = df
            
            # Add small delay between batches to avoid hitting rate limits
            if remaining_symbols:
                await asyncio.sleep(0.5)
                
        return data
        
    def fetch_data_for_all_symbols(self, symbols: List[str], timeframe: str, 
                                  start_date: str, end_date: Optional[str] = None, 
                                  use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols in parallel using ThreadPoolExecutor"""
        data = {}
        remaining_symbols = symbols.copy()
        
        while remaining_symbols:
            # Calculate optimal batch size based on current rate limit usage
            batch_size = calculate_optimal_batch_size(
                self.rate_limiter, 
                len(remaining_symbols), 
                timeframe, 
                self.max_klines, 
                self.max_workers
            )
            
            # Get next batch of symbols
            batch = remaining_symbols[:batch_size]
            remaining_symbols = remaining_symbols[batch_size:]
            
            print(f"Processing batch of {len(batch)} symbols (remaining: {len(remaining_symbols)})")
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                futures = {}
                for symbol in batch:
                    if use_cache:
                        df = self.load_data_from_csv(symbol, timeframe)
                        if df is not None:
                            data[symbol] = df
                            continue
                            
                    futures[executor.submit(self.download_historical_data, symbol, timeframe, start_date, end_date)] = symbol
                
                if futures:
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {timeframe} data (batch)"):
                        symbol = futures[future]
                        try:
                            df = future.result()
                            if not df.empty:
                                data[symbol] = df
                                self.save_data_to_csv(df, symbol, timeframe)
                        except Exception as e:
                            print(f"Error fetching data for {symbol} on {timeframe}: {e}")
            
            # Add small delay between batches to avoid hitting rate limits
            if remaining_symbols:
                print(f"Batch complete. Moving to next batch...")
                time.sleep(0.5)  # Small delay between batches
                    
        return data 