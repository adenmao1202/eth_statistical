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
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn

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
        self.client = Client(api_key, api_secret)
        self.max_klines = max_klines
        self.max_workers = max_workers
        self.use_proxies = use_proxies
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            request_delay=request_delay
        )
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_dir=config.CACHE_DIR, 
            expiry=config.DEFAULT_CACHE_EXPIRY
        )
        
        # Initialize proxy manager if requested
        self.proxy_manager = ProxyManager() if use_proxies else None

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
        # Apply rate limiting
        self.rate_limiter.wait()
        
        # Use proxy if enabled
        kwargs = {}
        if self.use_proxies and self.proxy_manager:
            kwargs['proxies'] = self.proxy_manager.get_proxy()
        
        return self.client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_time,
            endTime=end_time,
            limit=self.max_klines,
            **kwargs
        )
    
    def _process_klines(self, klines: List[List]) -> pd.DataFrame:
        """
        Process raw klines data into a DataFrame
        
        Args:
            klines (List[List]): Raw klines data
            
        Returns:
            pd.DataFrame: Processed klines data
        """
        if not klines:
            return pd.DataFrame()
            
        # Extract data columns
        data = []
        for k in klines:
            data.append({
                'timestamp': datetime.fromtimestamp(k[0] / 1000),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'close_time': datetime.fromtimestamp(k[6] / 1000),
                'quote_asset_volume': float(k[7]),
                'number_of_trades': int(k[8]),
                'taker_buy_base_asset_volume': float(k[9]),
                'taker_buy_quote_asset_volume': float(k[10])
            })
        
        # Create DataFrame and set index
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def fetch_historical_data(self, symbol: str, interval: str, days: int, 
                             end_date: Optional[str] = None,
                             use_cache: bool = True,
                             progress_bar: Optional[Any] = None) -> pd.DataFrame:
        """
        Fetch historical klines data for a symbol
        
        Args:
            symbol (str): Symbol name
            interval (str): Kline interval (e.g., '1m', '1h')
            days (int): Number of days to fetch
            end_date (str, optional): End date in YYYY-MM-DD format
            use_cache (bool): Whether to use cached data
            progress_bar (Any, optional): Optional external progress bar
            
        Returns:
            pd.DataFrame: Historical data
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = f"{symbol}_{interval}_{days}_{end_date}"
            cached_data = self.cache_manager.load(cache_key)
            if cached_data is not None:
                console.print(f"[cyan]Using cached data for {symbol}[/cyan]")
                return cached_data
        
        # Calculate time boundaries
        if end_date:
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_datetime = datetime.now()
            
        # Add one day to end_datetime to ensure we get all data for the last day
        end_datetime = end_datetime + timedelta(days=1)
        end_timestamp = int(end_datetime.timestamp() * 1000)
        
        start_datetime = end_datetime - timedelta(days=days)
        start_timestamp = int(start_datetime.timestamp() * 1000)
        
        # Calculate optimal batch size
        batch_size = calculate_optimal_batch_size_for_fetching(
            interval=interval,
            days=days,
            max_klines=self.max_klines
        )
        
        console.print(f"[yellow]Fetching {days} days of {interval} data for {symbol}[/yellow]")
        console.print(f"[yellow]Period: {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}[/yellow]")
        
        # Calculate number of batches
        total_minutes = days * 24 * 60
        interval_minutes = config.TIMEFRAME_MINUTES.get(interval, 1)
        total_candles = total_minutes // interval_minutes
        num_batches = (total_candles + self.max_klines - 1) // self.max_klines
        
        all_klines = []
        
        # Create timeframes for batch requests
        timeframes = []
        current_start = start_timestamp
        
        while current_start < end_timestamp:
            # Calculate batch end time
            batch_end = min(
                current_start + batch_size * 60 * 1000,  # batch_size minutes in milliseconds
                end_timestamp
            )
            
            timeframes.append((current_start, batch_end))
            current_start = batch_end
        
        # Fetch data in batches with progress tracking
        if progress_bar is None:
            # Create a new Progress instance if none was provided
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(), 
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Fetching {symbol}...", total=len(timeframes))
                
                for start, end in timeframes:
                    try:
                        batch_klines = self._fetch_klines(symbol, interval, start, end)
                        all_klines.extend(batch_klines)
                        progress.update(task, advance=1)
                    except Exception as e:
                        console.print(f"[bold red]Error fetching {symbol} from {datetime.fromtimestamp(start/1000)} to {datetime.fromtimestamp(end/1000)}: {str(e)}[/bold red]")
                        # Add a delay to recover from errors
                        time.sleep(2)
        else:
            # Use the provided progress bar
            for start, end in timeframes:
                try:
                    batch_klines = self._fetch_klines(symbol, interval, start, end)
                    all_klines.extend(batch_klines)
                    # No progress update here, handled by the caller
                except Exception as e:
                    console.print(f"[bold red]Error fetching {symbol} from {datetime.fromtimestamp(start/1000)} to {datetime.fromtimestamp(end/1000)}: {str(e)}[/bold red]")
                    # Add a delay to recover from errors
                    time.sleep(2)
        
        # Process all klines
        df = self._process_klines(all_klines)
        
        # Sort index to ensure chronological order
        if not df.empty:
            df.sort_index(inplace=True)
            console.print(f"[green]Successfully fetched {len(df)} candles for {symbol}[/green]")
            
            # Cache the result if enabled
            if use_cache:
                self.cache_manager.save(cache_key, df)
        else:
            console.print(f"[bold red]No data fetched for {symbol}[/bold red]")
        
        return df
    
    def fetch_multi_symbols(self, symbols: List[str], interval: str, days: int,
                           end_date: Optional[str] = None, 
                           use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently
        
        Args:
            symbols (List[str]): List of symbols
            interval (str): Kline interval
            days (int): Number of days to fetch
            end_date (str, optional): End date in YYYY-MM-DD format
            use_cache (bool): Whether to use cached data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of symbol -> data
        """
        result = {}
        
        # Determine symbols to process
        symbols_to_process = []
        cached_symbols = 0
        
        if use_cache:
            # Check cache first
            for symbol in symbols:
                cache_key = f"{symbol}_{interval}_{days}_{end_date}"
                cached_data = self.cache_manager.load(cache_key)
                if cached_data is not None:
                    result[symbol] = cached_data
                    cached_symbols += 1
                else:
                    symbols_to_process.append(symbol)
                    
            if cached_symbols > 0:
                console.print(f"[cyan]Using cached data for {cached_symbols} symbols[/cyan]")
        else:
            symbols_to_process = symbols
        
        # If we have symbols to process, use ThreadPoolExecutor
        if symbols_to_process:
            console.print(f"[yellow]Fetching data for {len(symbols_to_process)} symbols...[/yellow]")
            
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(), 
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Fetching symbols...", total=len(symbols_to_process))
                
                # Define a list to store (symbol, future) pairs
                futures = []
                
                # Submit all tasks
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for symbol in symbols_to_process:
                        future = executor.submit(
                            self.fetch_historical_data,
                            symbol=symbol,
                            interval=interval,
                            days=days,
                            end_date=end_date,
                            use_cache=use_cache,
                            progress_bar=True  # Just pass a flag indicating external progress tracking
                        )
                        futures.append((symbol, future))
                    
                    # Process results as they complete
                    for symbol, future in futures:
                        try:
                            data = future.result()
                            result[symbol] = data
                            progress.update(task, advance=1)
                        except Exception as e:
                            console.print(f"[bold red]Error fetching {symbol}: {str(e)}[/bold red]")
        
        console.print(f"[green]Successfully fetched data for {len(result)} symbols[/green]")
        return result
    
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
        # Get top symbols by volume
        symbols = self.get_top_volume_symbols(n=top_n)
        
        # Add reference symbol if not already included
        reference_symbol = config.DEFAULT_REFERENCE_SYMBOL
        if include_reference and reference_symbol not in symbols:
            symbols.insert(0, reference_symbol)
        
        console.print(f"[bold]Fetching data for {len(symbols)} symbols[/bold]")
        
        # Fetch data for all symbols
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
            # 统计缓存文件数量和总大小
            cache_files = []
            total_size = 0
            
            for root, _, files in os.walk(config.CACHE_DIR):
                for file in files:
                    if file.endswith('.parquet'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        file_time = os.path.getmtime(file_path)
                        total_size += file_size
                        cache_files.append({
                            'path': file_path, 
                            'size': file_size, 
                            'modified': datetime.fromtimestamp(file_time)
                        })
            
            # 对缓存按时间排序
            cache_files.sort(key=lambda x: x['modified'], reverse=True)
            
            # 获取最老和最新的缓存
            oldest_cache = cache_files[-1] if cache_files else None
            newest_cache = cache_files[0] if cache_files else None
            
            # 计算缓存有效期
            current_time = time.time()
            expired_files = sum(1 for f in cache_files if current_time - os.path.getmtime(f['path']) > config.DEFAULT_CACHE_EXPIRY)
            
            return {
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'oldest_cache': oldest_cache,
                'newest_cache': newest_cache,
                'expired_files': expired_files,
                'cache_dir': config.CACHE_DIR,
                'cache_expiry_days': config.DEFAULT_CACHE_EXPIRY / (24 * 3600)
            }
        except Exception as e:
            console.print(f"[bold red]Error getting cache stats: {str(e)}[/bold red]")
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
            
            for root, _, files in os.walk(config.CACHE_DIR):
                for file in files:
                    if file.endswith('.parquet'):
                        file_path = os.path.join(root, file)
                        file_time = os.path.getmtime(file_path)
                        file_size = os.path.getsize(file_path)
                        
                        # 檢查文件是否過期或強制清理
                        if force_clean or (current_time - file_time > config.DEFAULT_CACHE_EXPIRY):
                            try:
                                os.remove(file_path)
                                removed_files += 1
                                reclaimed_space += file_size
                            except Exception as e:
                                console.print(f"[bold red]Error removing file {file_path}: {str(e)}[/bold red]")
            
            result = {
                'removed_files': removed_files,
                'reclaimed_space_mb': reclaimed_space / (1024 * 1024),
                'force_clean': force_clean
            }
            
            console.print(f"[bold green]Removed {removed_files} cache files, reclaimed {result['reclaimed_space_mb']:.2f} MB of disk space[/bold green]")
            return result
        except Exception as e:
            console.print(f"[bold red]Error cleaning cache: {str(e)}[/bold red]")
            return {'error': str(e)} 