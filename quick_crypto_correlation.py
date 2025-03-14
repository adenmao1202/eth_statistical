#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Quick Cryptocurrency Correlation Analyzer
A high-performance version that analyzes the price correlation between ETH and other cryptocurrencies
using Binance historical data across different timeframes with parallel processing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from binance.client import Client
from datetime import datetime, timedelta
import os
import warnings
import concurrent.futures
import time
import pickle
import hashlib
import json
from functools import lru_cache

warnings.filterwarnings('ignore')

# Set up Binance client
client = Client()

# Define parameters
eth_symbol = 'ETHUSDT'
# Select a limited set of popular cryptocurrencies
selected_symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
    'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT',
    'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT'
]

# Define timeframes
timeframes = {
    '1h': Client.KLINE_INTERVAL_1HOUR,
    '4h': Client.KLINE_INTERVAL_4HOUR
}

# Set max K-lines (candles) to fetch (between 50-200 for better performance)
max_klines = 100  # Reduced for better performance
max_klines = max(50, min(200, max_klines))  # Ensure max_klines is between 50-200

# Define timeframe minutes for date calculations
timeframe_minutes = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240
}

# Set date range (last 3 days for faster processing)
end_date = datetime.now()
start_date = end_date - timedelta(days=3)

# Create data directory
data_dir = 'quick_data'
cache_dir = f'{data_dir}/cache'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Maximum number of concurrent API requests
MAX_WORKERS = 5

# API rate limit settings
REQUEST_DELAY = 0.1  # seconds between API calls

# Cache settings
CACHE_EXPIRY = 3600  # seconds (1 hour)

def get_cache_key(symbol, interval, start_str, end_str):
    """Generate a unique cache key for the data request."""
    key_data = f"{symbol}_{interval}_{start_str}_{end_str}_{max_klines}"
    return hashlib.md5(key_data.encode()).hexdigest()

def save_to_cache(cache_key, data):
    """Save data to cache file."""
    cache_file = f"{cache_dir}/{cache_key}.pkl"
    cache_meta = f"{cache_dir}/{cache_key}.meta"
    
    # Save data
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    # Save metadata
    meta = {
        'timestamp': time.time(),
        'symbol': data.index.name if hasattr(data, 'index') else None
    }
    with open(cache_meta, 'w') as f:
        json.dump(meta, f)

def load_from_cache(cache_key):
    """Load data from cache if it exists and is not expired."""
    cache_file = f"{cache_dir}/{cache_key}.pkl"
    cache_meta = f"{cache_dir}/{cache_key}.meta"
    
    if not (os.path.exists(cache_file) and os.path.exists(cache_meta)):
        return None
    
    # Check if cache is expired
    with open(cache_meta, 'r') as f:
        meta = json.load(f)
    
    if time.time() - meta['timestamp'] > CACHE_EXPIRY:
        return None  # Cache expired
    
    # Load data from cache
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

@lru_cache(maxsize=32)
def get_timeframe_minutes(interval):
    """Get minutes for a timeframe with caching."""
    timeframe_code = interval.split('_')[-1]
    
    if timeframe_code == '1h':
        return timeframe_minutes['1h']
    elif timeframe_code == '4h':
        return timeframe_minutes['4h']
    else:
        return 60  # Default to 1h if unknown

def download_klines(symbol, interval):
    """Download historical kline data for a symbol and interval with caching."""
    # Calculate appropriate date range based on max_klines and timeframe
    minutes = get_timeframe_minutes(interval)
    minutes_needed = max_klines * minutes
    
    # Calculate adjusted start date to get approximately max_klines candles
    adjusted_start = end_date - timedelta(minutes=minutes_needed)
    start = max(start_date, adjusted_start)
    
    # Convert to timestamps
    start_str = int(start.timestamp() * 1000)
    end_str = int(end_date.timestamp() * 1000)
    
    # Generate cache key
    cache_key = get_cache_key(symbol, interval, start_str, end_str)
    
    # Try to load from cache first
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        print(f"Using cached data for {symbol} ({interval})")
        return cached_data
    
    print(f"Downloading {symbol} data for {interval} timeframe...")
    
    # Add delay to avoid rate limits
    time.sleep(REQUEST_DELAY)
    
    # Get the data from Binance
    klines = client.futures_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str,
        limit=max_klines
    )
    
    # Create DataFrame with only necessary columns for correlation analysis
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Optimize: Convert only necessary columns
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)  # Only convert 'close' for correlation
    
    # Set timestamp as index and symbol as name
    df.set_index('timestamp', inplace=True)
    df.index.name = symbol
    
    # Save to cache
    save_to_cache(cache_key, df)
    
    return df

def calculate_correlation(data_dict, reference_symbol=eth_symbol):
    """Calculate correlation between reference symbol and all other symbols."""
    # Optimize: Extract only the close prices for correlation calculation
    close_prices = {}
    for symbol, df in data_dict.items():
        # Only use the 'close' column to save memory
        close_prices[symbol] = df['close']
    
    # Create DataFrame with all close prices
    price_df = pd.DataFrame(close_prices)
    
    # Calculate correlation with reference symbol only
    # This is faster than calculating the full correlation matrix
    correlation = price_df.corr()[reference_symbol].sort_values(ascending=False)
    
    return correlation

def visualize_correlation(correlation, timeframe):
    """Visualize correlation between reference symbol and all symbols."""
    # Save the correlation data first (this is the important part)
    correlation.to_csv(f'{data_dir}/correlation_{timeframe}.csv')
    
    # Get top positive and negative correlations
    top_positive = correlation.drop(eth_symbol).head(10)
    top_negative = correlation.drop(eth_symbol).tail(10)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot top positive correlations
    sns.barplot(x=top_positive.values, y=top_positive.index, ax=ax1, palette='viridis')
    ax1.set_title(f'Top Positive Correlations with {eth_symbol} ({timeframe})')
    ax1.set_xlabel('Correlation')
    ax1.set_ylabel('Symbol')
    
    # Plot top negative correlations
    sns.barplot(x=top_negative.values, y=top_negative.index, ax=ax2, palette='coolwarm')
    ax2.set_title(f'Top Negative Correlations with {eth_symbol} ({timeframe})')
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Symbol')
    
    plt.tight_layout()
    plt.savefig(f'{data_dir}/correlation_{timeframe}.png', dpi=150)  # Lower DPI for faster saving

def fetch_symbol_data(args):
    """Helper function for parallel data fetching."""
    symbol, timeframe_code = args
    try:
        return symbol, download_klines(symbol, timeframe_code)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return symbol, None

def main():
    start_time = time.time()
    print(f"Analyzing cryptocurrency correlations with {eth_symbol} as benchmark")
    print(f"Time period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Selected symbols: {len(selected_symbols)}")
    print(f"Max K-lines per symbol: {max_klines}")
    print(f"Using parallel processing with {MAX_WORKERS} workers")
    
    results = {}
    
    # Analyze each timeframe
    for timeframe_name, timeframe_code in timeframes.items():
        timeframe_start = time.time()
        print(f"\nAnalyzing {timeframe_name} timeframe")
        
        # Prepare arguments for parallel processing
        args_list = [(symbol, timeframe_code) for symbol in selected_symbols]
        
        # Fetch data for all symbols in parallel
        data = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for symbol, df in executor.map(fetch_symbol_data, args_list):
                if df is not None and not df.empty:
                    data[symbol] = df
        
        # Calculate correlation
        correlation = calculate_correlation(data, eth_symbol)
        results[timeframe_name] = correlation
        
        # Print correlation
        print(f"\nCorrelation with {eth_symbol} ({timeframe_name}):")
        print(correlation)
        
        # Visualize correlation
        visualize_correlation(correlation, timeframe_name)
        
        timeframe_end = time.time()
        print(f"Timeframe {timeframe_name} analyzed in {timeframe_end - timeframe_start:.2f} seconds")
    
    # Create heatmap of correlations across timeframes
    all_corr = pd.DataFrame(results)
    
    # Save correlation data
    all_corr.to_csv(f'{data_dir}/all_correlations.csv')
    
    # Create heatmap with lower resolution for speed
    plt.figure(figsize=(8, 6))
    sns.heatmap(all_corr, annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Correlation with {eth_symbol} Across Timeframes')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/correlation_heatmap.png', dpi=150)  # Lower DPI for faster saving
    
    end_time = time.time()
    print(f"\nAnalysis complete in {end_time - start_time:.2f} seconds!")
    print(f"Results saved to the '{data_dir}' directory.")

if __name__ == "__main__":
    main()
