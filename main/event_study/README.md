# ETH Event Study Module

## Introduction
This module conducts robust event studies to analyze the impact of ETH price drop events on various cryptocurrencies. The implementation focuses on avoiding look-ahead bias by classifying coins based on pre-event characteristics.

## Key Features
1. **Cross-period Validation**: Analyzes coin stability across multiple time periods
2. **Objective Classification**: Classifies coins based on market cap, volume, volatility, or sector
3. **Performance Metrics**: Measures mean performance and stability around ETH drop events
4. **Visual Analysis**: Generates performance heatmaps and stability charts

## Usage
Run the main script with the following command:
```bash
python main.py [options]
```

### Options:
- `--timeframe`: Data timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d), default: 1h
- `--total_days`: Total days to analyze, default: 360
- `--period_length`: Length of each validation period in days, default: 90
- `--use_cache`: Use cached data when available
- `--request_delay`: Delay between API requests in seconds, default: 0.1

### Classification Options:
- `--classify_by_market_cap`: Classify coins by market capitalization
- `--classify_by_volume`: Classify coins by trading volume
- `--classify_by_volatility`: Classify coins by price volatility
- `--classify_by_sector`: Classify coins by sector (experimental)
- `--create_random_baskets`: Create random coin baskets for comparison

### ETH Event Parameters:
- `--drop_threshold`: ETH price drop threshold, default: -0.02
- `--consecutive_drops`: Number of consecutive candles with drops, default: 1
- `--volume_factor`: Volume increase factor for ETH events, default: 1.5

## Output Results
The analysis generates:
1. Stability scores for coins in each classification
2. Comparison of performance across time periods
3. Visualizations of performance distributions
4. Detailed reports in the results directory

## Improvements Over Original Version
1. **No Look-ahead Bias**: Classification based on pre-event characteristics
2. **Robustness**: Validation across multiple time periods
3. **Modularity**: Clean separation of functionality
4. **Error Handling**: Better error handling and recovery
5. **Interactive Output**: Rich console output with progress bars

## Dependencies
- Python 3.7+
- pandas, numpy, matplotlib, seaborn
- rich (for pretty console output)
- Binance API (for data fetching)

## Important Notes
- Sufficient historical data is required for reliable results
- Analysis may take a significant amount of time depending on the number of coins and time periods
- API rate limits may require adjusting request_delay parameter

## Contact
For questions or improvements, please contact the development team.
