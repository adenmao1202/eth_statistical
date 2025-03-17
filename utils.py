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
from typing import Optional, List, Any

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
    """Cache Manager"""
    
    def __init__(self, cache_dir: str = config.CACHE_DIR, expiry: int = config.DEFAULT_CACHE_EXPIRY):
        """
        Initialize the cache manager
        
        Args:
            cache_dir (str): Cache directory
            expiry (int): Cache expiry time (seconds)
        """
        self.cache_dir = cache_dir
        self.cache_expiry = expiry
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_key(self, symbol: str, timeframe: str, start_str: str, end_str: str, max_klines: int) -> str:
        """
        Generate a unique cache key for a data request
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            start_str (str): Start time
            end_str (str): End time
            max_klines (int): Maximum number of klines
            
        Returns:
            str: Cache key
        """
        key_data = f"{symbol}_{timeframe}_{start_str}_{end_str}_{max_klines}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def save(self, cache_key: str, data: Any) -> None:
        """
        Save data to a cache file
        
        Args:
            cache_key (str): Cache key
            data (Any): Data to save
        """
        cache_file = f"{self.cache_dir}/{cache_key}.pkl"
        cache_meta = f"{self.cache_dir}/{cache_key}.meta"
        
        # Save data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata
        meta = {
            'timestamp': time.time(),
            'symbol': data.index.name if hasattr(data, 'index') and data.index.name else None
        }
        with open(cache_meta, 'w') as f:
            json.dump(meta, f)
    
    def load(self, cache_key: str) -> Optional[Any]:
        """
        Load data from cache if it exists and is not expired
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            Optional[Any]: Cached data, or None if not found or expired
        """
        cache_file = f"{self.cache_dir}/{cache_key}.pkl"
        cache_meta = f"{self.cache_dir}/{cache_key}.meta"
        
        if not (os.path.exists(cache_file) and os.path.exists(cache_meta)):
            return None
        
        # Check if cache is expired
        with open(cache_meta, 'r') as f:
            meta = json.load(f)
        
        if time.time() - meta['timestamp'] > self.cache_expiry:
            return None  # Cache expired
        
        # Load data from cache
        with open(cache_file, 'rb') as f:
            return pickle.load(f)


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
        Get a random proxy from the proxy list
        
        Returns:
            Optional[str]: Random proxy URL, or None if not using proxies
        """
        if not self.use_proxies or not self.proxies:
            return None
        return random.choice(self.proxies)


def calculate_optimal_batch_size(rate_limiter: RateLimiter, remaining_symbols: int, 
                                timeframe: str, max_klines: int, max_workers: int) -> int:
    """
    Calculate the optimal batch size based on current rate limit usage and timeframe
    
    Args:
        rate_limiter (RateLimiter): Rate limiter instance
        remaining_symbols (int): Number of symbols remaining to process
        timeframe (str): Timeframe being processed
        max_klines (int): Maximum number of klines
        max_workers (int): Maximum number of worker threads
        
    Returns:
        int: Optimal batch size
    """
    # Calculate remaining weight capacity
    remaining_weight = config.MAX_WEIGHT_PER_MINUTE - rate_limiter.weight_used
    
    # Estimate weight per request based on timeframe and max_klines
    request_weight = 1  # Default weight
    if max_klines > 100:
        if max_klines <= 500:
            request_weight = 2
        elif max_klines <= 1000:
            request_weight = 5
        else:
            request_weight = 10
            
    # Calculate how many requests we can make with remaining weight
    max_possible_requests = max(1, remaining_weight // request_weight)
    
    # Consider time remaining in current minute
    time_remaining = max(0.1, rate_limiter.weight_reset_time - time.time())
    requests_per_second = config.MAX_REQUESTS_PER_SECOND
    
    # Calculate how many requests we can make in the remaining time
    max_time_based_requests = int(time_remaining * requests_per_second * 0.8)  # 80% to ensure safety
    
    # Take the minimum of weight-based and time-based limits
    max_requests = min(max_possible_requests, max_time_based_requests)
    
    # Consider max_workers setting
    max_workers_limit = min(max_workers, remaining_symbols)
    
    # Calculate final batch size
    optimal_batch_size = min(max_requests, max_workers_limit, remaining_symbols)
    
    # Ensure batch size is at least 1 but not more than 20 (to avoid overwhelming the API)
    return max(1, min(20, optimal_batch_size))


def optimize_request_parameters(days: int, timeframe: str, use_cache: bool = True) -> dict:
    """
    Optimize request parameters based on analysis needs
    
    Args:
        days (int): Number of days to analyze
        timeframe (str): Timeframe for analysis
        use_cache (bool): Whether to use cached data
        
    Returns:
        dict: Dictionary containing optimized parameters
    """
    # Calculate optimal max_klines based on timeframe and days
    # Calculate total minutes needed
    total_minutes = days * 24 * 60
    
    # Calculate how many candles we need
    candles_needed = total_minutes / config.TIMEFRAME_MINUTES.get(timeframe, 60)
    
    # Determine optimal max_klines (keeping within API limits)
    if candles_needed <= 100:
        optimal_max_klines = 100  # Weight: 1
    elif candles_needed <= 500:
        optimal_max_klines = min(500, candles_needed)  # Weight: 2
    elif candles_needed <= 1000:
        optimal_max_klines = min(1000, candles_needed)  # Weight: 5
    else:
        # For very large requests, we need to make multiple calls
        # Limit to 1000 to keep weight at 5
        optimal_max_klines = 1000
    
    # Determine optimal request delay based on symbol count and cache usage
    if use_cache:
        # We can be more aggressive if using cache
        optimal_request_delay = 0.2
    else:
        # More conservative if not using cache
        optimal_request_delay = 0.5
    
    # Determine optimal max_workers based on system resources
    cpu_count = os.cpu_count() or 4
    optimal_max_workers = min(10, max(2, cpu_count - 1))
    
    optimized_params = {
        'max_klines': int(optimal_max_klines),
        'request_delay': optimal_request_delay,
        'max_workers': optimal_max_workers
    }
    
    print("\n=== Optimized Request Parameters ===")
    print(f"Max klines: {optimized_params['max_klines']} (based on {days} days of {timeframe} data)")
    print(f"Request delay: {optimized_params['request_delay']} seconds")
    print(f"Max workers: {optimized_params['max_workers']}")
    print("====================================\n")
    
    return optimized_params 