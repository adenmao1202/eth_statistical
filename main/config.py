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
DEFAULT_CACHE_EXPIRY = 604800  # 7 days in seconds (increased from 1 day)

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

# Data directories - UNIFIED to central location
DATA_DIR = 'main/historical_data'
CACHE_DIR = f'{DATA_DIR}/cache'

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Analysis parameters --> Change here !!! 
DEFAULT_TOP_N = 20
DEFAULT_WINDOW_SIZE = 20
DEFAULT_DROP_THRESHOLD = -0.01  # Detect 1% drops
DEFAULT_REBOUND_THRESHOLD = 0.003  # 0.3% rebound

# Multi-timeframe analysis default parameters --> Change here !!! 
DEFAULT_MULTI_TIMEFRAMES = ['1m', '5m', '15m', '1h']
DEFAULT_WINDOW_SIZES = [5, 10, 20]
DEFAULT_DROP_THRESHOLDS = [-0.01, -0.03, -0.05]

# Retry strategy parameters
MAX_RETRY_COUNT = 3

# ----- Event Study Parameters -----
EVENT_WINDOW_MINUTES = 30  # Minutes before and after the event
DETECTION_WINDOW_SIZE = 1  # Window size to detect 1% drops in 1 minute
MIN_DROP_PCT = -0.01  # 1% minimum drop to constitute an event

# ----- Clustering Analysis Parameters -----
# Feature selection parameters for clustering
VOLATILITY_WINDOW = 20  # Window for calculating volatility
VOLUME_NORMALIZE = True  # Whether to normalize volume
CORRELATION_THRESHOLD = 0.3  # Minimum correlation to consider
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_N_ESTIMATORS = 100
NUM_CLUSTERS = 5  # Default number of clusters to form

# ----- Hypothesis Testing Parameters -----
# Parameters for hypothesis testing
SIGNIFICANCE_LEVEL = 0.05  # 5% significance level
PRE_EVENT_WINDOW = 30  # 30 minutes before event
POST_EVENT_WINDOW = 30  # 30 minutes after event 