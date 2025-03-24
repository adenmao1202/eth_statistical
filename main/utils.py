#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Module
Contains helper functions for caching, proxies, and rate limiting
"""

import os
import time
import json
import pickle
import hashlib
import random
from typing import Optional, List, Any, Dict, Tuple
import pandas as pd
import scipy.stats as stats

import config

class RateLimiter:
    """API Rate Limit Manager"""
    
    def __init__(self, request_delay: float = config.DEFAULT_REQUEST_DELAY):
        """
        Initialize the rate limiter
        
        Args:
            request_delay (float): Delay between requests (seconds)
        """
        self.request_delay = request_delay
        self.last_request_time = 0
        
        # Rate limit tracking
        self.weight_used = 0
        self.weight_reset_time = time.time() + 60  # Reset weight counter every minute
        self.requests_this_second = 0
        self.second_start_time = time.time()
    
    def wait(self, weight: int = 1) -> None:
        """
        Wait to comply with rate limits
        
        Args:
            weight (int): Weight of the request
        """
        current_time = time.time()
        
        # Check if we need to reset the weight counter
        if current_time >= self.weight_reset_time:
            self.weight_used = 0
            self.weight_reset_time = current_time + 60
        
        # Check if we need to reset the requests per second counter
        if current_time - self.second_start_time >= 1:
            self.requests_this_second = 0
            self.second_start_time = current_time
        
        # Check if we're approaching the weight limit
        if self.weight_used + weight >= config.MAX_WEIGHT_PER_MINUTE:
            # Wait until the next minute starts
            sleep_time = self.weight_reset_time - current_time
            if sleep_time > 0:
                print(f"Approaching weight limit ({self.weight_used}/{config.MAX_WEIGHT_PER_MINUTE}). Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Reset counters after waiting
                self.weight_used = 0
                self.weight_reset_time = time.time() + 60
                self.requests_this_second = 0
                self.second_start_time = time.time()
                current_time = time.time()
        
        # Check if we're approaching the requests per second limit
        if self.requests_this_second >= config.MAX_REQUESTS_PER_SECOND:
            # Wait until the next second starts
            sleep_time = 1 - (current_time - self.second_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.requests_this_second = 0
                self.second_start_time = time.time()
                current_time = time.time()
        
        # Enforce minimum delay between requests
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_delay:
            sleep_time = self.request_delay - elapsed
            time.sleep(sleep_time)
        
        # Update counters
        self.last_request_time = time.time()
        self.weight_used += weight
        self.requests_this_second += 1
    
    def track_usage(self, show_details: bool = True) -> dict:
        """
        Track and display current API usage statistics
        
        Args:
            show_details (bool): Whether to print detailed usage information
            
        Returns:
            dict: Dictionary containing API usage statistics
        """
        current_time = time.time()
        time_until_reset = max(0, self.weight_reset_time - current_time)
        
        usage_stats = {
            'weight_used': self.weight_used,
            'weight_limit': config.MAX_WEIGHT_PER_MINUTE,
            'weight_remaining': config.MAX_WEIGHT_PER_MINUTE - self.weight_used,
            'weight_usage_percent': (self.weight_used / config.MAX_WEIGHT_PER_MINUTE) * 100,
            'time_until_reset': time_until_reset,
            'requests_this_second': self.requests_this_second,
            'requests_per_second_limit': config.MAX_REQUESTS_PER_SECOND
        }
        
        if show_details:
            print("\n=== API Usage Statistics ===")
            print(f"Weight used: {self.weight_used}/{config.MAX_WEIGHT_PER_MINUTE} ({usage_stats['weight_usage_percent']:.1f}%)")
            print(f"Time until weight reset: {time_until_reset:.1f} seconds")
            print(f"Requests this second: {self.requests_this_second}/{config.MAX_REQUESTS_PER_SECOND}")
            print("===========================\n")
            
        return usage_stats


class CacheManager:
    """管理API響應的緩存"""
    
    def __init__(self, cache_dir: str = config.CACHE_DIR, expiry: int = config.DEFAULT_CACHE_EXPIRY):
        """
        初始化緩存管理器
        
        Args:
            cache_dir (str): 緩存文件目錄
            expiry (int): 緩存過期時間（秒）
        """
        self.cache_dir = cache_dir
        self.expiry = expiry
        # 創建緩存目錄（如果不存在）
        os.makedirs(cache_dir, exist_ok=True)
        
        # 增加緩存命中統計
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 內存緩存，用於最常訪問的項目
        self.memory_cache = {}
        self.memory_cache_size = 50  # 內存中保留的最大項目數
        self.memory_cache_hits = 0
        
        # 調試信息
        self.debug_enabled = False
    
    def enable_debug(self, enabled: bool = True) -> None:
        """
        啟用或禁用調試輸出
        
        Args:
            enabled (bool): 是否啟用調試
        """
        self.debug_enabled = enabled
    
    def generate_key(self, symbol: str, interval: str, days: int, end_date: Optional[str] = None) -> str:
        """
        根據參數生成緩存鍵
        
        Args:
            symbol (str): 幣種符號
            interval (str): K線間隔
            days (int): 天數
            end_date (str, optional): 結束日期字符串
            
        Returns:
            str: 緩存鍵
        """
        # 根據參數生成唯一鍵
        key_components = f"{symbol}_{interval}_{days}"
        if end_date:
            key_components += f"_{end_date}"
        
        # 創建鍵組件的哈希
        key_hash = hashlib.md5(key_components.encode()).hexdigest()
        return key_hash
    
    def get_cache_key(self, symbol: str, timeframe: str, start_str: str, end_str: str, max_klines: int) -> str:
        """
        根據參數獲取緩存鍵
        
        Args:
            symbol (str): 幣種符號
            timeframe (str): K線時間框架
            start_str (str): 開始時間字符串
            end_str (str): 結束時間字符串
            max_klines (int): 每個請求的最大K線數
            
        Returns:
            str: 緩存鍵
        """
        # 基於所有參數創建唯一鍵
        key_str = f"{symbol}_{timeframe}_{start_str}_{end_str}_{max_klines}"
        
        # 創建鍵字符串的哈希
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_file_path(self, cache_key: str) -> str:
        """
        獲取緩存文件路徑
        
        Args:
            cache_key (str): 緩存鍵
            
        Returns:
            str: 緩存文件路徑
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def save(self, cache_key: str, data: Any) -> None:
        """
        將數據保存到緩存
        
        Args:
            cache_key (str): 緩存鍵
            data (Any): 要保存的數據
        """
        cache_path = self.get_file_path(cache_key)
        
        try:
            # 檢查數據是否為DataFrame，是否有足夠的行數來值得緩存
            save_to_disk = True
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    # 不緩存空DataFrame
                    save_to_disk = False
                elif len(data) < 10:
                    # 少於10行的數據不值得緩存到磁盤
                    self._log_debug(f"數據行數少於10（{len(data)}行），僅保存到內存緩存")
                    save_to_disk = False
            
            # 保存到內存緩存
            self.memory_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            
            # 如果內存緩存過大，移除最舊的項目
            if len(self.memory_cache) > self.memory_cache_size:
                oldest_key = min(self.memory_cache.keys(), 
                                key=lambda k: self.memory_cache[k]['timestamp'])
                del self.memory_cache[oldest_key]
            
            # 如果需要，保存到磁盤
            if save_to_disk:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'data': data,
                        'timestamp': time.time()
                    }, f)
                self._log_debug(f"已保存緩存：{os.path.basename(cache_path)}")
        except Exception as e:
            self._log_debug(f"保存緩存時出錯: {e}")
    
    def load(self, cache_key: str) -> Optional[Any]:
        """
        如果可用且未過期，從緩存加載數據
        
        Args:
            cache_key (str): 緩存鍵
            
        Returns:
            Optional[Any]: 緩存的數據或None（如果不可用）
        """
        # 首先檢查內存緩存
        if cache_key in self.memory_cache:
            cache_data = self.memory_cache[cache_key]
            
            # 檢查內存緩存是否過期
            if time.time() - cache_data['timestamp'] <= self.expiry:
                self.cache_hits += 1
                self.memory_cache_hits += 1
                self._log_debug(f"內存緩存命中: {cache_key[:8]}...")
                return cache_data['data']
            else:
                # 從內存緩存中移除過期項目
                del self.memory_cache[cache_key]
        
        # 然後檢查磁盤緩存
        cache_path = self.get_file_path(cache_key)
        
        # 檢查緩存文件是否存在
        if not os.path.exists(cache_path):
            self.cache_misses += 1
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 檢查緩存是否過期
            if time.time() - cache_data['timestamp'] > self.expiry:
                self.cache_misses += 1
                return None
            
            # 更新內存緩存
            self.memory_cache[cache_key] = cache_data
            
            # 如果內存緩存過大，移除最舊的項目
            if len(self.memory_cache) > self.memory_cache_size:
                oldest_key = min(self.memory_cache.keys(), 
                                key=lambda k: self.memory_cache[k]['timestamp'])
                del self.memory_cache[oldest_key]
            
            self.cache_hits += 1
            self._log_debug(f"磁盤緩存命中: {os.path.basename(cache_path)}")
            return cache_data['data']
        except Exception as e:
            self._log_debug(f"加載緩存時出錯: {e}")
            self.cache_misses += 1
            return None
    
    def exists(self, cache_key: str) -> bool:
        """
        檢查緩存項是否存在且未過期
        
        Args:
            cache_key (str): 緩存鍵
            
        Returns:
            bool: 如果緩存項存在且未過期，則為True
        """
        # 首先檢查內存緩存
        if cache_key in self.memory_cache:
            # 檢查內存緩存是否過期
            if time.time() - self.memory_cache[cache_key]['timestamp'] <= self.expiry:
                return True
        
        # 然後檢查磁盤緩存
        cache_path = self.get_file_path(cache_key)
        
        # 如果文件不存在，直接返回False
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 檢查緩存是否過期
            return time.time() - cache_data['timestamp'] <= self.expiry
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取緩存統計信息
        
        Returns:
            Dict[str, Any]: 緩存統計信息
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # 計算緩存目錄大小
        total_size = 0
        file_count = 0
        
        for root, _, files in os.walk(self.cache_dir):
            for file in files:
                if file.endswith('.pkl'):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_limit': self.memory_cache_size,
            'memory_cache_hits': self.memory_cache_hits,
            'disk_cache_files': file_count,
            'disk_cache_size_mb': total_size / (1024 * 1024),
            'expiry_seconds': self.expiry
        }
    
    def _log_debug(self, message: str) -> None:
        """
        記錄調試信息
        
        Args:
            message (str): 調試消息
        """
        if self.debug_enabled:
            print(f"[CacheManager] {message}")


class ProxyManager:
    """Proxy Manager"""
    
    def __init__(self, proxies: List[str] = None, use_proxies: bool = False):
        """
        Initialize the proxy manager
        
        Args:
            proxies (List[str], optional): List of proxy URLs
            use_proxies (bool, optional): Whether to use proxy rotation
        """
        self.proxies = proxies or []
        self.use_proxies = use_proxies
    
    def get_proxy(self) -> Optional[str]:
        """
        Get a random proxy from the list
        
        Returns:
            Optional[str]: Proxy URL or None if not using proxies or no proxies available
        """
        if not self.use_proxies or not self.proxies:
            return None
        return random.choice(self.proxies)


def calculate_optimal_batch_size(remaining_symbols: int, 
                               max_workers: int) -> int:
    """
    Calculate optimal batch size for data fetching
    
    Args:
        remaining_symbols (int): Number of remaining symbols to fetch
        max_workers (int): Maximum number of concurrent workers
        
    Returns:
        int: Optimal batch size
    """
    # No need for a batch if remaining symbols is less than max workers
    if remaining_symbols <= max_workers:
        return remaining_symbols
    
    # Use at least 1 batch
    batch_size = min(remaining_symbols, max_workers)
    
    # Ensure batch size is reasonable
    return max(1, min(batch_size, 20))  # Cap at 20 to avoid overwhelming the API


def optimize_request_parameters(days: int, timeframe: str, use_cache: bool = True) -> dict:
    """
    Optimize request parameters based on analysis needs
    
    Args:
        days (int): Number of days to analyze
        timeframe (str): Timeframe
        use_cache (bool): Whether to use cache
        
    Returns:
        dict: Optimized parameters
    """
    # Base parameters
    max_klines = config.DEFAULT_MAX_KLINES
    request_delay = config.DEFAULT_REQUEST_DELAY
    max_workers = config.DEFAULT_MAX_WORKERS
    
    # Adjust based on timeframe and duration
    if timeframe == '1m':
        if days > 30:
            # Smaller batches, slower requests for longer periods
            max_klines = 500
            request_delay = 1.0
            max_workers = 3
        else:
            # Larger batches, faster requests for shorter periods
            max_klines = 1000
            request_delay = 0.5
            max_workers = 5
    elif timeframe == '5m':
        if days > 60:
            max_klines = 750
            request_delay = 0.8
            max_workers = 4
        else:
            max_klines = 1000
            request_delay = 0.5
            max_workers = 5
    else:
        # For larger timeframes, use maximum efficiency
        max_klines = 1000
        request_delay = 0.25
        max_workers = 8
    
    # If using cache, we can be more aggressive
    if use_cache:
        request_delay = max(0.1, request_delay / 2)
        max_workers = min(10, max_workers * 2)
    
    return {
        'max_klines': max_klines,
        'request_delay': request_delay,
        'max_workers': max_workers
    }

def calculate_abnormal_returns(coin_data: Dict[str, pd.DataFrame], 
                              event_periods: List[Dict[str, Any]], 
                              window_minutes: int) -> Dict[str, List[float]]:
    """
    Calculate abnormal returns for coins during event periods
    
    Args:
        coin_data (Dict[str, pd.DataFrame]): Dictionary of coin data
        event_periods (List[Dict[str, Any]]): List of event periods
        window_minutes (int): Window size in minutes
        
    Returns:
        Dict[str, List[float]]: Dictionary of abnormal returns for each coin
    """
    abnormal_returns = {}
    
    for symbol, df in coin_data.items():
        if symbol != config.DEFAULT_REFERENCE_SYMBOL:  # Skip reference symbol
            abnormal_returns[symbol] = []
            
            for period in event_periods:
                start_time = period['start']
                end_time = period['end']
                
                # Get data for the event window
                window_data = df[(df.index >= start_time) & (df.index <= end_time)]
                
                if not window_data.empty:
                    # Calculate percentage return during the window
                    start_price = window_data.iloc[0]['close']
                    end_price = window_data.iloc[-1]['close']
                    pct_return = (end_price / start_price - 1) * 100
                    
                    abnormal_returns[symbol].append(pct_return)
    
    return abnormal_returns

def calculate_optimal_batch_size_for_fetching(interval: str, days: int = 30, max_klines: int = 1000) -> int:
    """
    Calculate the optimal batch size for fetching historical data
    based on the interval and number of days
    
    Args:
        interval (str): Kline interval (e.g., '1m', '5m', '1h')
        days (int): Number of days of data to fetch
        max_klines (int): Maximum number of klines allowed per request
        
    Returns:
        int: Optimal batch size
    """
    # Calculate interval in minutes
    interval_minutes = 0
    if 'm' in interval:
        interval_minutes = int(interval.replace('m', ''))
    elif 'h' in interval:
        interval_minutes = int(interval.replace('h', '')) * 60
    elif 'd' in interval:
        interval_minutes = int(interval.replace('d', '')) * 60 * 24
    
    # Calculate total number of klines
    total_klines = (days * 24 * 60) // interval_minutes
    
    # Calculate optimal batch size
    batch_size = min(total_klines, max_klines)
    
    return batch_size

def calculate_sample_size_requirement(effect_size: float = 0.5, 
                                     alpha: float = 0.05, 
                                     power: float = 0.8) -> int:
    """
    计算基于t检验所需的最小样本量
    
    参数:
        effect_size (float): 预期效应大小 (Cohen's d)
        alpha (float): 显著性水平 (Type I error)
        power (float): 检验功效 (1 - Type II error)
        
    返回:
        int: 需要的最小样本量
    """
    # 计算临界t值（双尾检验）
    t_crit = stats.t.ppf(1 - alpha/2, 100)  # 自由度初始设为100
    
    # 非中心参数
    delta = effect_size
    
    # 迭代查找样本量
    for n in range(4, 1000):  # 从4开始（最小合理样本）
        # 更新自由度和临界t值
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # 计算特定样本量下的检验功效
        ncp = delta * (n**0.5)  # 非中心参数
        actual_power = 1 - stats.nct.cdf(t_crit, df, ncp)
        
        # 如果达到所需功效，返回样本量
        if actual_power >= power:
            return n
    
    return 1000  # 如果迭代结束仍未达到功效，返回上限

def analyze_sample_adequacy(observed_effect: float, current_sample_size: int, 
                           p_value: float, alpha: float = 0.05) -> dict:
    """
    分析当前样本量是否足够，并估计达到显著性所需的样本量
    
    参数:
        observed_effect (float): 观察到的效应（如平均异常收益）
        current_sample_size (int): 当前样本大小
        p_value (float): 当前p值
        alpha (float): 显著性水平
        
    返回:
        dict: 样本充分性分析结果
    """
    # 计算当前效应大小 (Cohen's d)
    # 由于我们只有t统计量，可以从t和样本量估算d
    t_stat = stats.t.ppf(1 - p_value/2, current_sample_size - 1)
    
    # Cohen's d = t / sqrt(n)
    cohen_d = abs(t_stat / (current_sample_size ** 0.5))
    
    # 计算在相同效应大小下，达到显著性所需的样本量
    for power in [0.8, 0.9, 0.95]:
        required_n = calculate_sample_size_requirement(
            effect_size=cohen_d,
            alpha=alpha,
            power=power
        )
        
        # 如果所需样本量非常大，可能意味着效应非常小
        if required_n > 1000:
            required_n = ">1000"
    
    # 返回分析结果
    return {
        "current_sample_size": current_sample_size,
        "observed_effect": observed_effect,
        "estimated_effect_size": cohen_d,
        "p_value": p_value,
        "significant": p_value < alpha,
        "required_sample_size_80power": calculate_sample_size_requirement(cohen_d, alpha, 0.8),
        "required_sample_size_90power": calculate_sample_size_requirement(cohen_d, alpha, 0.9),
        "required_sample_size_95power": calculate_sample_size_requirement(cohen_d, alpha, 0.95)
    }

def calculate_effect_size(mean_difference, std_dev):
    """
    計算效應大小 (Cohen's d) 並給出解釋

    Parameters:
    -----------
    mean_difference : float
        平均差異
    std_dev : float
        標準差

    Returns:
    --------
    tuple
        (效應大小, 效應大小解釋)
    """
    # 避免除以零
    if std_dev == 0:
        return 0, "無法計算"
    
    # 計算Cohen's d
    effect_size = mean_difference / std_dev
    
    # 解釋效應大小
    if abs(effect_size) < 0.2:
        effect_interpretation = "微小"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "小"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "中等" 
    else:
        effect_interpretation = "大"
    
    return effect_size, effect_interpretation 