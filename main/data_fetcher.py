#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cryptocurrency Data Fetcher
Responsible for fetching historical price data from cryptocurrency exchanges
"""

import os
import time
import random
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from binance.client import Client
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
import hashlib
import pickle

import config
from utils import RateLimiter, CacheManager, ProxyManager, calculate_optimal_batch_size_for_fetching

# Initialize Rich console
console = Console()

class DataFetcher:
    """Fetches cryptocurrency data from Binance"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 max_klines: int = config.DEFAULT_MAX_KLINES,
                 request_delay: float = config.DEFAULT_REQUEST_DELAY,
                 max_workers: int = config.DEFAULT_MAX_WORKERS,
                 use_proxies: bool = False):
        """
        Initialize the data fetcher
        
        Args:
            api_key (str, optional): Binance API key
            api_secret (str, optional): Binance API secret
            max_klines (int): Maximum number of klines to fetch per request
            request_delay (float): Delay between API requests in seconds
            max_workers (int): Maximum number of concurrent workers
            use_proxies (bool): Whether to use proxy rotation
        """
        # 允许使用公共API模式
        self.client = Client(api_key, api_secret)
        self.max_klines = max_klines
        self.max_workers = max_workers
        self.use_proxies = use_proxies
        
        # 添加标志来指示是否使用公共API
        self.using_public_api = api_key is None or api_secret is None
        if self.using_public_api:
            console.print("[yellow]Running with public API (no authentication). Some features may be limited.[/yellow]")
        
        # 普通用户的API权重限制是1200/分钟，保守设置为1100作为警戒线
        self.weight_limit = 1100  
        # 初始化API請求權重跟踪
        self.weight_used = 0
        self.weight_reset_time = time.time() + 60  # 每分鐘重置權重計數器
        self.last_weight_check = time.time()  # 上次检查权重的时间
        
        # Initialize rate limiter - 普通用户适当增加延迟以避免触发限制
        adjusted_delay = max(request_delay, 0.1)  # 确保至少0.1秒的延迟
        self.rate_limiter = RateLimiter(
            request_delay=adjusted_delay
        )
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=config.CACHE_DIR, 
            expiry=config.DEFAULT_CACHE_EXPIRY
        )
        
        # Initialize proxy manager if requested
        self.proxy_manager = ProxyManager() if use_proxies else None
        
        # 批量获取数据的配置
        self.batch_count = 0  # 当前批次计数
        self.max_batch_before_pause = 30  # 连续获取30个批次后暂停
        self.pause_duration = 1.0  # 暂停1秒

    def _get_symbols(self) -> List[str]:
        """
        Get list of all USDT trading pairs
        
        Returns:
            List[str]: List of symbol names
        """
        try:
            with console.status("[bold green]Fetching available symbols..."):
                # Get exchange info
                exchange_info = self.client.get_exchange_info()
                
                # Filter for USDT trading pairs that are TRADING status
                symbols = [
                    s['symbol'] for s in exchange_info['symbols']
                    if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
                ]
                
                console.print(f"[green]Found {len(symbols)} USDT trading pairs[/green]")
                return symbols
        except Exception as e:
            console.print(f"[bold red]Error fetching symbols: {str(e)}[/bold red]")
            return []
    
    def _fetch_klines(self, symbol: str, interval: str, start_time: Optional[int] = None, 
                    end_time: Optional[int] = None) -> List[List]:
        """
        Fetch klines data for a single period
        
        Args:
            symbol (str): Symbol name
            interval (str): Kline interval
            start_time (int, optional): Start time in milliseconds
            end_time (int, optional): End time in milliseconds
            
        Returns:
            List[List]: List of klines data
        """
        # 如果使用公共API，增加请求延迟以避免被限流
        if self.using_public_api:
            self.rate_limiter.wait(weight=5)  # 增加权重以降低请求频率
        else:
            # 原有的标准速率限制
            self.rate_limiter.wait()
        
        # Use proxy if enabled
        kwargs = {}
        if self.use_proxies and self.proxy_manager:
            kwargs['proxies'] = self.proxy_manager.get_proxy()
        
        try:
            return self.client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=self.max_klines,
                **kwargs
            )
        except Exception as e:
            # 公共API出错时提供更多详细信息
            if self.using_public_api and "APIError" in str(e):
                console.print(f"[bold yellow]公共API访问受限: {e}。如需更高访问限制，请配置API密钥。[/bold yellow]")
                # 增加强制延迟，避免频繁出错
                time.sleep(10)  
                # 重试一次
                return self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_time,
                    endTime=end_time,
                    limit=min(500, self.max_klines),  # 减少请求数据量
                    **kwargs
                )
            else:
                raise  # 非公共API错误，正常抛出
    
    def _process_klines(self, kline_data: List[List]) -> pd.DataFrame:
        """
        Process raw kline data into a pandas DataFrame
        
        Args:
            kline_data (List[List]): Raw kline data from Binance API
            
        Returns:
            pd.DataFrame: Processed data
        """
        if not kline_data:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(kline_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                       'quote_asset_volume', 'taker_buy_base_asset_volume',
                       'taker_buy_quote_asset_volume']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Set index
        df.set_index('open_time', inplace=True)
        
        # Sort by index
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def fetch_historical_klines(self, symbol: str, interval: str, 
                             start_date: datetime, end_date: datetime, 
                             show_progress: bool = True) -> pd.DataFrame:
        """
        Fetch historical klines data for a symbol
        
        Args:
            symbol (str): Symbol name
            interval (str): Kline interval
            start_date (datetime): Start date
            end_date (datetime): End date
            show_progress (bool): Whether to show progress bar
            
        Returns:
            pd.DataFrame: Historical klines data
        """
        # Check if data is in cache
        cache_params = {
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        cache_key = self.cache_manager.get_cache_key(
            symbol=symbol,
            timeframe=interval,
            start_str=start_date.strftime('%Y-%m-%d %H:%M:%S'),
            end_str=end_date.strftime('%Y-%m-%d %H:%M:%S'),
            max_klines=self.max_klines
        )
        cached_data = self.cache_manager.load(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # 公共API模式下，减少请求时间范围以提高稳定性
        if self.using_public_api and (end_date - start_date).days > 30:
            console.print("[yellow]使用公共API时，建议将请求时间范围限制在30天以内。将尝试分批获取数据...[/yellow]")
            
            # 分批获取数据
            all_klines = []
            current_start = start_date
            batch_days = 30
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=console if show_progress else None
            ) as progress:
                overall_task = progress.add_task(f"[cyan]Getting historical data for {symbol}...", total=(end_date - start_date).days)
                
                while current_start < end_date:
                    current_end = min(current_start + timedelta(days=batch_days), end_date)
                    
                    # 获取当前批次的数据
                    try:
                        batch_df = self._fetch_historical_klines_segment(symbol, interval, current_start, current_end)
                        if not batch_df.empty:
                            all_klines.append(batch_df)
                    except Exception as e:
                        console.print(f"[red]Error fetching batch for {symbol}: {e}. Trying to continue...[/red]")
                        # 增加延迟，避免频繁错误
                        time.sleep(30)
                    
                    progress.update(overall_task, advance=(current_end - current_start).days)
                    current_start = current_end
                    
                    # 为公共API减少服务器压力
                    time.sleep(5)
            
            # 合并所有数据
            if all_klines:
                combined_df = pd.concat(all_klines)
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df.sort_index(inplace=True)
                
                # 缓存合并后的结果
                self.cache_manager.save(cache_key, combined_df)
                
                return combined_df
            else:
                console.print(f"[red]No data obtained for {symbol} in the specified date range.[/red]")
                return pd.DataFrame()
        
        # 原有的获取数据逻辑，适用于未使用公共API或时间范围较短的情况
        return self._fetch_historical_klines_segment(symbol, interval, start_date, end_date, show_progress)
    
    def fetch_historical_data(self, symbol: str, interval: str, days: int, 
                           end_date: Optional[str] = None,
                           show_progress: bool = True,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical klines for a symbol
        
        Args:
            symbol (str): Symbol name
            interval (str): Kline interval
            days (int): Number of days to fetch
            end_date (str, optional): End date (YYYY-MM-DD)
            show_progress (bool): Whether to show progress bar
            use_cache (bool): Whether to use cached data
            
        Returns:
            pd.DataFrame: Historical data
        """
        # 為大量數據自動調整參數
        if days > 60 and interval in ['1m', '3m', '5m']:
            console.print(f"[yellow]請求大量數據 ({days} 天的 {interval} 數據)。建議使用更大的時間框架以提高效率。[/yellow]")
        
        # 重置批次計數器
        self.batch_count = 0
        
        # 增強緩存使用策略 - 始終先檢查緩存
        # 即使use_cache=False，也先檢查緩存是否存在，以顯示相關信息
        cache_key = self.cache_manager.generate_key(symbol, interval, days, end_date)
        cached_data = self.cache_manager.load(cache_key)
        
        if cached_data is not None:
            if use_cache:
                console.print(f"[green]從緩存加載 {symbol} 的數據。[/green]")
                return cached_data
            else:
                console.print(f"[yellow]發現 {symbol} 的緩存數據，但由於 use_cache=False，將重新獲取。[/yellow]")
        
        # 如果沒有緩存或不使用緩存
        # 計算結束日期（如果未提供）
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()
        
        # 計算開始日期
        start = end - timedelta(days=days)
        
        # 優化公共API使用的時間段策略
        # 為公共API使用更保守的參數
        if self.using_public_api:
            max_days_per_request = 15  # 更保守的每次請求天數
            
            # 對於高頻數據使用更小的時間段
            if interval in ['1m', '3m', '5m']:
                max_days_per_request = 7  # 分鐘級數據每次最多獲取7天
                console.print(f"[blue]使用公共API獲取高頻數據，將請求分為較小的時間段以避免限制。[/blue]")
        else:
            max_days_per_request = 30  # 使用API密鑰時可以請求更多數據
        
        # 分段獲取數據策略
        if days > max_days_per_request:
            # 分段獲取數據，每段最多max_days_per_request天
            all_dfs = []
            current_start = start
            current_end = min(current_start + timedelta(days=max_days_per_request), end)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console if show_progress else None
            ) as progress:
                total_segments = (days + max_days_per_request - 1) // max_days_per_request
                task_id = progress.add_task(f"[cyan]分段獲取 {symbol} 數據...", total=total_segments)
                
                for segment in range(total_segments):
                    if show_progress:
                        progress.update(task_id, description=f"[cyan]獲取 {symbol} 段 {segment+1}/{total_segments}...")
                    
                    # 檢查這個時間段的緩存
                    segment_cache_key = self.cache_manager.get_cache_key(
                        symbol=symbol,
                        timeframe=interval,
                        start_str=current_start.strftime('%Y-%m-%d %H:%M:%S'),
                        end_str=current_end.strftime('%Y-%m-%d %H:%M:%S'),
                        max_klines=self.max_klines
                    )
                    
                    segment_cached_data = self.cache_manager.load(segment_cache_key)
                    
                    if segment_cached_data is not None and use_cache:
                        # 使用緩存數據
                        if show_progress:
                            progress.console.print(f"[green]從緩存加載 {symbol} 段 {segment+1}/{total_segments}[/green]")
                        segment_df = segment_cached_data
                    else:
                        # 獲取這個時間段的數據
                        segment_df = self._fetch_historical_klines_segment(
                            symbol=symbol,
                            interval=interval,
                            start_date=current_start,
                            end_date=current_end,
                            show_progress=False  # 不在每個分段顯示進度條
                        )
                        
                        # 保存這個時間段到緩存
                        if not segment_df.empty:
                            self.cache_manager.save(segment_cache_key, segment_df)
                    
                    if not segment_df.empty:
                        all_dfs.append(segment_df)
                    
                    # 更新進度
                    progress.update(task_id, advance=1)
                    
                    # 移動到下一個時間段
                    current_start = current_end + timedelta(seconds=1)
                    current_end = min(current_start + timedelta(days=max_days_per_request), end)
                    
                    # 在每個分段之間暫停以避免觸發限制
                    if segment < total_segments - 1:
                        sleep_time = 3.0 if self.using_public_api else 1.0  # 公共API使用更長的暫停時間
                        if show_progress:
                            progress.console.print(f"[blue]分段之間暫停 {sleep_time}秒...[/blue]")
                        time.sleep(sleep_time)
            
            # 合併所有分段
            if not all_dfs:
                return pd.DataFrame()
            
            df = pd.concat(all_dfs)
            df = df[~df.index.duplicated(keep='first')]  # 移除可能的重複
            df.sort_index(inplace=True)
        else:
            # 對於較小的時間段，檢查是否有特定時間範圍的緩存
            specific_cache_key = self.cache_manager.get_cache_key(
                symbol=symbol,
                timeframe=interval,
                start_str=start.strftime('%Y-%m-%d %H:%M:%S'),
                end_str=end.strftime('%Y-%m-%d %H:%M:%S'),
                max_klines=self.max_klines
            )
            
            specific_cached_data = self.cache_manager.load(specific_cache_key)
            
            if specific_cached_data is not None and use_cache:
                console.print(f"[green]從精確緩存加載 {symbol} 數據。[/green]")
                return specific_cached_data
            
            # 直接獲取
            df = self._fetch_historical_klines_segment(
                symbol=symbol,
                interval=interval,
                start_date=start,
                end_date=end,
                show_progress=show_progress
            )
        
        # 保存到緩存
        if df is not None and not df.empty:
            self.cache_manager.save(cache_key, df)
        
        return df
    
    def _fetch_historical_klines_segment(self, symbol: str, interval: str,
                                      start_date: datetime, end_date: datetime,
                                      show_progress: bool = True,
                                      update_interval: int = 1) -> pd.DataFrame:
        """
        Fetch historical klines for a specific time segment
        
        Args:
            symbol (str): Symbol name
            interval (str): Kline interval
            start_date (datetime): Start date
            end_date (datetime): End date
            show_progress (bool): Whether to show progress bar
            update_interval (int): Interval for updating progress bar
            
        Returns:
            pd.DataFrame: Historical data
        """
        # Convert dates to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Calculate total time span
        time_span_ms = end_ms - start_ms
        
        # Calculate batch size in milliseconds
        try:
            # 計算最佳批次大小
            days_span = (end_date - start_date).days + 1
            batch_size = calculate_optimal_batch_size_for_fetching(interval, days_span, self.max_klines)
            
            # 轉換為毫秒
            interval_ms = self._get_interval_milliseconds(interval)
            batch_size_ms = batch_size * interval_ms
        except Exception as e:
            # 使用默認值
            console.print(f"[yellow]Using default batch size due to error: {str(e)}[/yellow]")
            # 默認使用12小時批次
            batch_size_ms = 12 * 60 * 60 * 1000
        
        # Calculate number of batches
        num_batches = max(1, time_span_ms // batch_size_ms + (1 if time_span_ms % batch_size_ms > 0 else 0))
        
        all_klines = []
        
        # Setup progress tracking
        if show_progress:
            label = f"Fetching {symbol} data..."
            progress_args = {
                "total": num_batches,
                "transient": True,
                "description": label,
                "refresh_per_second": 1
            }
            
            progress_columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn()
            ]
            
            progress = Progress(*progress_columns)
            task_id = progress.add_task(**progress_args)
            progress.start()
        else:
            progress = None
            task_id = None
        
        # Process each batch
        last_update = 0
        try:
            for i in range(num_batches):
                # Calculate batch start and end times
                batch_start = start_ms + i * batch_size_ms
                batch_end = min(end_ms, batch_start + batch_size_ms)
                
                # 定期检查权重限制
                current_time = time.time()
                if current_time - self.last_weight_check >= 5:  # 每5秒检查一次
                    self.last_weight_check = current_time
                    if current_time >= self.weight_reset_time:
                        if show_progress and progress and self.weight_used > 0:
                            progress.console.print(f"[cyan]Weight counter reset. Used {self.weight_used} in the last minute.[/cyan]")
                        self.weight_used = 0
                        self.weight_reset_time = current_time + 60
                
                # 检查是否接近权重限制
                if self.weight_used >= self.weight_limit:
                    wait_time = max(1, self.weight_reset_time - current_time)
                    if show_progress and progress:
                        progress.console.print(f"[yellow]Approaching weight limit ({self.weight_used}/{self.weight_limit}). Waiting {wait_time:.2f} seconds...[/yellow]")
                    time.sleep(wait_time)
                    self.weight_used = 0
                    self.weight_reset_time = current_time + 60 + wait_time
                    self.last_weight_check = time.time()
                
                # 定期暂停以避免连续请求
                self.batch_count += 1
                if self.batch_count >= self.max_batch_before_pause:
                    if show_progress and progress:
                        progress.console.print(f"[blue]Pausing for {self.pause_duration}s after {self.max_batch_before_pause} batches to avoid rate limiting...[/blue]")
                    time.sleep(self.pause_duration)
                    self.batch_count = 0
                
                # Fetch the batch of klines
                batch_klines = self._fetch_kline_batch(symbol, interval, batch_start, batch_end)
                
                if batch_klines is not None:
                    all_klines.extend(batch_klines)
                
                # Update progress
                if show_progress and progress:
                    if i - last_update >= update_interval or i == num_batches - 1:
                        progress.update(task_id, completed=i+1)
                        last_update = i
                
                # Apply rate limiting
                self.rate_limiter.wait()
        finally:
            # Clean up progress bar
            if show_progress and progress:
                progress.stop()
        
        if not all_klines:
            return pd.DataFrame()
        
        # Process all collected klines
        df = self._process_klines(all_klines)
        
        if show_progress:
            console.print(f"Fetched {len(df)} klines for {symbol}")
        
        return df
    
    def _get_interval_milliseconds(self, interval: str) -> int:
        """
        Convert interval string to milliseconds
        
        Args:
            interval (str): Interval string (e.g., '1m', '1h', '1d')
            
        Returns:
            int: Interval in milliseconds
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported interval: {interval}")
    
    def _fetch_kline_batch(self, symbol: str, interval: str, 
                        start_ms: int, end_ms: int) -> Optional[List[List]]:
        """
        Fetch a batch of klines
        
        Args:
            symbol (str): Symbol name
            interval (str): Kline interval
            start_ms (int): Start time in milliseconds
            end_ms (int): End time in milliseconds
            
        Returns:
            Optional[List[List]]: Batch of klines or None if error
        """
        # 為批次創建唯一的緩存鍵
        batch_cache_key = f"{symbol}_{interval}_{start_ms}_{end_ms}"
        batch_cache_key = hashlib.md5(batch_cache_key.encode()).hexdigest()
        batch_cache_path = os.path.join(self.cache_manager.cache_dir, f"{batch_cache_key}.pkl")
        
        # 檢查批次緩存
        if os.path.exists(batch_cache_path):
            try:
                with open(batch_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # 檢查緩存是否過期
                if time.time() - cache_data['timestamp'] <= self.cache_manager.expiry:
                    return cache_data['data']
            except Exception:
                # 如果讀取緩存出錯，繼續從API獲取
                pass
        
        try:
            # 應用速率限制
            self.rate_limiter.wait()
            
            # 使用代理（如果啟用）
            if self.use_proxies and self.proxy_manager:
                proxy = self.proxy_manager.get_proxy()
                if proxy:
                    # 如果需要，在這裡實現代理使用
                    pass
            
            # 獲取K線數據
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=self.max_klines
            )
            
            # 更新權重跟踪
            self.weight_used += 2  # 通常每個請求2個權重單位
            
            # 緩存批次數據
            try:
                with open(batch_cache_path, 'wb') as f:
                    pickle.dump({
                        'data': klines,
                        'timestamp': time.time()
                    }, f)
            except Exception as e:
                # 緩存失敗不影響主要功能
                console.print(f"[yellow]緩存批次數據失敗: {str(e)}[/yellow]")
            
            return klines
        except Exception as e:
            console.print(f"[red]獲取 {symbol} 數據時出錯: {str(e)}[/red]")
            return None
    
    def fetch_multi_symbols(self, symbols: List[str], interval: str, days: int,
                           end_date: Optional[str] = None, use_cache: bool = True,
                           max_workers: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently
        
        Args:
            symbols (List[str]): List of symbols to fetch
            interval (str): Kline interval
            days (int): Number of days to fetch
            end_date (str, optional): End date in YYYY-MM-DD format
            use_cache (bool): Whether to use cached data
            max_workers (int, optional): Maximum number of concurrent workers
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their historical data
        """
        # 優化公共API的並發請求數
        if self.using_public_api:
            # 公共API使用更保守的並發請求數量
            recommended_workers = 2
            console.print(f"[yellow]使用公共API，將並發請求限制為 {recommended_workers} 以避免速率限制。[/yellow]")
        else:
            # 有API密鑰時可以使用更多線程
            recommended_workers = min(self.max_workers, os.cpu_count() or 4)
        
        # 使用推薦的工作線程數或用戶指定的數量
        workers = max_workers or recommended_workers
        
        # 預先檢查緩存，優先獲取已緩存的數據
        # 這樣可以減少並發請求的數量
        results = {}
        symbols_to_fetch = []
        
        with console.status("[cyan]檢查緩存中的數據..."):
            for symbol in symbols:
                cache_key = self.cache_manager.generate_key(symbol, interval, days, end_date)
                cached_data = self.cache_manager.load(cache_key)
                
                if cached_data is not None and use_cache:
                    results[symbol] = cached_data
                    console.print(f"[green]從緩存加載 {symbol} 數據。[/green]")
                else:
                    symbols_to_fetch.append(symbol)
        
        if not symbols_to_fetch:
            console.print("[green]所有請求的數據都從緩存中獲取。[/green]")
            return results
        
        # 創建進度跟踪佈局
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            overall_task = progress.add_task(f"[cyan]獲取 {len(symbols_to_fetch)} 個幣種的數據...", total=len(symbols_to_fetch))
            
            # 為公共API優化批處理策略
            if self.using_public_api and len(symbols_to_fetch) > 5:
                # 分批處理以避免同時發起過多請求
                batch_size = 5  # 每批5個幣種
                for i in range(0, len(symbols_to_fetch), batch_size):
                    batch = symbols_to_fetch[i:i+batch_size]
                    progress.console.print(f"[blue]處理批次 {i//batch_size + 1}/{(len(symbols_to_fetch) + batch_size - 1)//batch_size}，包含 {len(batch)} 個幣種[/blue]")
                    
                    # 處理這一批
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = []
                        for symbol in batch:
                            future = executor.submit(
                                self.fetch_historical_data,
                                symbol=symbol,
                                interval=interval,
                                days=days,
                                end_date=end_date,
                                show_progress=False,
                                use_cache=use_cache
                            )
                            futures.append((symbol, future))
                        
                        # 處理結果
                        for symbol, future in futures:
                            try:
                                results[symbol] = future.result()
                                progress.update(overall_task, advance=1)
                            except Exception as e:
                                console.print(f"[bold red]獲取 {symbol} 數據時出錯: {str(e)}[/bold red]")
                                results[symbol] = pd.DataFrame()
                                progress.update(overall_task, advance=1)
                    
                    # 批次之間暫停，避免觸發限流
                    if i + batch_size < len(symbols_to_fetch):
                        sleep_time = 5  # 批次之間暫停5秒
                        progress.console.print(f"[blue]批次之間暫停 {sleep_time} 秒...[/blue]")
                        time.sleep(sleep_time)
            else:
                # 對於較少的幣種或使用API密鑰時，正常處理
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = []
                    for symbol in symbols_to_fetch:
                        future = executor.submit(
                            self.fetch_historical_data,
                            symbol=symbol,
                            interval=interval,
                            days=days,
                            end_date=end_date,
                            show_progress=False,
                            use_cache=use_cache
                        )
                        futures.append((symbol, future))
                    
                    # 處理結果
                    for symbol, future in futures:
                        try:
                            results[symbol] = future.result()
                            progress.update(overall_task, advance=1)
                        except Exception as e:
                            console.print(f"[bold red]獲取 {symbol} 數據時出錯: {str(e)}[/bold red]")
                            results[symbol] = pd.DataFrame()
                            progress.update(overall_task, advance=1)
        
        console.print(f"[green]成功獲取 {len([df for df in results.values() if not df.empty])}/{len(symbols)} 個幣種的數據[/green]")
        
        return results
    
    def get_top_volume_symbols(self, n: int = 100, quote_asset: str = 'USDT') -> List[str]:
        """
        Get top N symbols by 24h volume
        
        Args:
            n (int): Number of symbols to return
            quote_asset (str): Quote asset filter
            
        Returns:
            List[str]: List of symbol names
        """
        try:
            with console.status(f"[bold green]Fetching top {n} {quote_asset} pairs by volume..."):
                # Get 24h ticker for all symbols
                tickers = self.client.get_ticker()
                
                # Filter for USDT pairs and sort by volume
                usdt_tickers = [t for t in tickers if t['symbol'].endswith(quote_asset)]
                usdt_tickers.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                
                # Get the top N symbols
                top_symbols = [t['symbol'] for t in usdt_tickers[:n]]
                
                console.print(f"[green]Found top {len(top_symbols)} {quote_asset} trading pairs by volume[/green]")
                return top_symbols
        except Exception as e:
            console.print(f"[bold red]Error fetching top volume symbols: {str(e)}[/bold red]")
            return []
    
    def get_all_data(self, timeframe: str = '1m', days: int = 30, 
                   top_n: int = 100, end_date: Optional[str] = None,
                   use_cache: bool = True,
                   include_reference: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get data for all top symbols
        
        Args:
            timeframe (str): Timeframe to fetch
            days (int): Number of days to fetch
            top_n (int): Number of top symbols by volume to fetch
            end_date (str, optional): End date in YYYY-MM-DD format
            use_cache (bool): Whether to use cached data
            include_reference (bool): Whether to include reference symbol (ETH)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of symbol -> data
        """
        # 檢查緩存是否有之前獲取的符號列表
        symbols_cache_key = f"top_symbols_{top_n}_{datetime.now().strftime('%Y-%m-%d')}"
        cached_symbols = self.cache_manager.load(symbols_cache_key)
        
        if cached_symbols is not None and use_cache:
            console.print(f"[green]從緩存加載熱門代幣列表。[/green]")
            symbols = cached_symbols
        else:
            # 獲取交易量最大的代幣
            symbols = self.get_top_volume_symbols(n=top_n)
            # 緩存一天的代幣列表
            if symbols:
                self.cache_manager.save(symbols_cache_key, symbols)
        
        # 添加參考代幣（如果尚未包含）
        reference_symbol = config.DEFAULT_REFERENCE_SYMBOL
        if include_reference and reference_symbol not in symbols:
            symbols.insert(0, reference_symbol)
        
        # 如果使用公共API，建議減少代幣數量
        if self.using_public_api and len(symbols) > 20:
            original_count = len(symbols)
            if days <= (30 if timeframe in ['15m', '30m', '1h'] else 15):
                # 對於較大的時間框架，可以處理更多代幣
                reduced_count = min(original_count, 30)
            else:
                # 對於較小的時間框架和更長時間段，減少代幣數量
                reduced_count = min(original_count, 15)
            
            # 確保參考代幣在內
            if include_reference and reference_symbol in symbols:
                symbols = [reference_symbol] + symbols[1:reduced_count]
            else:
                symbols = symbols[:reduced_count]
            
            console.print(f"[yellow]使用公共API，將代幣數量從 {original_count} 減少到 {len(symbols)} 以避免速率限制。[/yellow]")
        
        console.print(f"[bold]獲取 {len(symbols)} 個代幣的數據[/bold]")
        
        # 獲取所有代幣的數據
        data = self.fetch_multi_symbols(
            symbols=symbols,
            interval=timeframe,
            days=days,
            end_date=end_date,
            use_cache=use_cache
        )
        
        return data
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        獲取緩存統計信息
        
        Returns:
            Dict[str, Any]: 緩存統計，包括大小、項目數量等
        """
        try:
            # 统计緩存文件数量和总大小
            cache_files = []
            total_size = 0
            
            for root, _, files in os.walk(config.CACHE_DIR):
                for file in files:
                    if file.endswith('.parquet') or file.endswith('.pkl'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        file_time = os.path.getmtime(file_path)
                        total_size += file_size
                        cache_files.append({
                            'path': file_path, 
                            'size': file_size, 
                            'modified': datetime.fromtimestamp(file_time)
                        })
            
            # 对緩存按时间排序
            cache_files.sort(key=lambda x: x['modified'], reverse=True)
            
            # 获取最老和最新的緩存
            oldest_cache = cache_files[-1] if cache_files else None
            newest_cache = cache_files[0] if cache_files else None
            
            # 计算緩存有效期
            current_time = time.time()
            expired_files = sum(1 for f in cache_files if current_time - os.path.getmtime(f['path']) > config.DEFAULT_CACHE_EXPIRY)
            
            # 計算緩存命中率（如果可能）
            if hasattr(self.cache_manager, 'cache_hits') and hasattr(self.cache_manager, 'cache_misses'):
                total_lookups = self.cache_manager.cache_hits + self.cache_manager.cache_misses
                hit_rate = (self.cache_manager.cache_hits / total_lookups * 100) if total_lookups > 0 else 0
            else:
                hit_rate = None
            
            return {
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'oldest_cache': oldest_cache,
                'newest_cache': newest_cache,
                'expired_files': expired_files,
                'cache_dir': config.CACHE_DIR,
                'cache_expiry_days': config.DEFAULT_CACHE_EXPIRY / (24 * 3600),
                'cache_hit_rate': hit_rate
            }
        except Exception as e:
            console.print(f"[bold red]獲取緩存統計時出錯: {str(e)}[/bold red]")
            return {
                'error': str(e),
                'cache_dir': config.CACHE_DIR
            }
    
    def clean_expired_cache(self, force_clean: bool = False) -> Dict[str, Any]:
        """
        清理過期緩存文件
        
        Args:
            force_clean (bool): 是否強制清理所有緩存
            
        Returns:
            Dict[str, Any]: 清理結果統計
        """
        try:
            current_time = time.time()
            removed_files = 0
            reclaimed_space = 0
            retained_files = 0
            
            # 識別所有緩存文件
            all_cache_files = []
            for root, _, files in os.walk(config.CACHE_DIR):
                for file in files:
                    if file.endswith('.parquet') or file.endswith('.pkl'):
                        file_path = os.path.join(root, file)
                        file_time = os.path.getmtime(file_path)
                        file_size = os.path.getsize(file_path)
                        
                        all_cache_files.append({
                            'path': file_path,
                            'time': file_time,
                            'size': file_size,
                            'age': current_time - file_time
                        })
            
            # 如果不是強制清理，僅刪除過期文件
            if not force_clean:
                # 識別過期文件
                expired_files = [f for f in all_cache_files if f['age'] > config.DEFAULT_CACHE_EXPIRY]
                
                console.print(f"[yellow]檢測到 {len(expired_files)} 個過期緩存文件，佔用 {sum(f['size'] for f in expired_files) / (1024 * 1024):.2f} MB[/yellow]")
                
                # 刪除過期文件
                for file_info in expired_files:
                    try:
                        os.remove(file_info['path'])
                        removed_files += 1
                        reclaimed_space += file_info['size']
                    except Exception as e:
                        console.print(f"[red]刪除文件 {os.path.basename(file_info['path'])} 時出錯: {str(e)}[/red]")
                
                retained_files = len(all_cache_files) - removed_files
            else:
                # 強制清理模式：可能先保留一些關鍵緩存
                # 例如，保留最近24小時的緩存
                recent_threshold = 24 * 3600  # 24小時
                recent_files = [f for f in all_cache_files if f['age'] <= recent_threshold]
                old_files = [f for f in all_cache_files if f['age'] > recent_threshold]
                
                console.print(f"[yellow]強制清理模式：將保留 {len(recent_files)} 個最近24小時的文件，刪除 {len(old_files)} 個舊文件[/yellow]")
                
                # 刪除所有不是最近的文件
                for file_info in old_files:
                    try:
                        os.remove(file_info['path'])
                        removed_files += 1
                        reclaimed_space += file_info['size']
                    except Exception as e:
                        console.print(f"[red]刪除文件 {os.path.basename(file_info['path'])} 時出錯: {str(e)}[/red]")
                
                retained_files = len(recent_files)
            
            result = {
                'removed_files': removed_files,
                'retained_files': retained_files,
                'reclaimed_space_mb': reclaimed_space / (1024 * 1024),
                'force_clean': force_clean
            }
            
            console.print(f"[bold green]已刪除 {removed_files} 個緩存文件，釋放 {result['reclaimed_space_mb']:.2f} MB 磁盤空間，保留 {retained_files} 個文件[/bold green]")
            return result
        except Exception as e:
            console.print(f"[bold red]清理緩存時出錯: {str(e)}[/bold red]")
            return {'error': str(e)} 