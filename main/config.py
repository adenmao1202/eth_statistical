#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration File
Contains all constants and configuration parameters for the cryptocurrency analysis tool
"""

import os
from binance.client import Client

# Default API settings
DEFAULT_MAX_KLINES = 1000
DEFAULT_REQUEST_DELAY = 1.0
DEFAULT_MAX_WORKERS = 5
DEFAULT_CACHE_EXPIRY = 86400  # 1 day in seconds

# API rate limits ( HARD LIMITS )
MAX_WEIGHT_PER_MINUTE = 2400  # Binance Futures API weight limit per minute
MAX_REQUESTS_PER_SECOND = 20  # Maximum Binance API requests per second

# Timeframe configuration
TIMEFRAMES = {
    '1m': Client.KLINE_INTERVAL_1MINUTE,
    '5m': Client.KLINE_INTERVAL_5MINUTE,
    '15m': Client.KLINE_INTERVAL_15MINUTE,
    '1h': Client.KLINE_INTERVAL_1HOUR,
    '4h': Client.KLINE_INTERVAL_4HOUR
}

# Timeframe minutes mapping
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240
}

# Default symbol
DEFAULT_REFERENCE_SYMBOL = 'ETHUSDT'

# Data directories
DATA_DIR = 'historical_data'
CACHE_DIR = f'{DATA_DIR}/cache'

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Analysis parameters --> Change here !!! 
DEFAULT_TOP_N = 20
DEFAULT_WINDOW_SIZE = 20
DEFAULT_DROP_THRESHOLD = -0.03
DEFAULT_REBOUND_THRESHOLD = 0.03

# Multi-timeframe analysis default parameters --> Change here !!! 
DEFAULT_MULTI_TIMEFRAMES = ['1m', '5m', '15m', '1h']
DEFAULT_WINDOW_SIZES = [5, 10, 20]
DEFAULT_DROP_THRESHOLDS = [-0.01, -0.03, -0.05]

# Retry strategy parameters
MAX_RETRY_COUNT = 3 