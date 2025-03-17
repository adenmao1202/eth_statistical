# Cryptocurrency Analysis Tool

A comprehensive tool for analyzing cryptocurrency correlations and performance during market downtrends, with a focus on optimizing API usage to avoid rate limits.

## Features

- **Correlation Analysis**: Identify cryptocurrencies with high or low correlation to a reference symbol (default: ETH)
- **Downtrend Analysis**: Find stable coins during ETH downtrends and coins that rebound well after downtrends
- **Multi-Timeframe Analysis**: Analyze correlations across different timeframes (1m, 5m, 15m, 1h, 4h)
- **Visualization**: Generate charts and heatmaps to visualize correlations and price movements
- **API Optimization**: Smart handling of Binance API rate limits with caching, batching, and proxy support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-analysis-tool.git
cd crypto-analysis-tool
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Correlation Analysis

```bash
python main.py --timeframe 1h --days 30
```

This will analyze correlations between all USDT trading pairs and ETHUSDT over the last 30 days using 1-hour candles.

### Downtrend Analysis

```bash
python main.py --downtrend --timeframe 1h --days 30 --drop_threshold -0.05
```

This will identify ETH downtrends (drops of 5% or more) and analyze which coins remain stable during these periods and which rebound well afterward.

### Multi-Timeframe Analysis

```bash
python main.py --correlation --all_timeframes --days 14
```

This will analyze correlations across all available timeframes (1m, 5m, 15m, 1h, 4h) over the last 14 days and generate a heatmap.

### Optimizing API Usage

```bash
python main.py --optimize --track_api_usage --request_delay 0.5 --max_workers 4
```

This will optimize request parameters based on your analysis needs and track API usage statistics.

## Command Line Arguments

### Data Fetching Parameters

- `--api_key`: Binance API key
- `--api_secret`: Binance API secret
- `--max_klines`: Maximum number of klines to fetch (default: 200)
- `--request_delay`: Delay between API requests in seconds (default: 1.0)
- `--max_workers`: Maximum number of concurrent workers (default: 5)
- `--use_proxies`: Use proxy rotation for API requests
- `--proxies`: List of proxy URLs
- `--cache_expiry`: Cache expiry time in seconds (default: 86400)

### Analysis Parameters

- `--reference_symbol`: Reference trading pair symbol (default: ETHUSDT)
- `--timeframe`: Timeframe for analysis (default: 1h, choices: 1m, 5m, 15m, 1h, 4h)
- `--days`: Number of days to analyze (default: 30)
- `--start_date`: Start date in YYYY-MM-DD format
- `--end_date`: End date in YYYY-MM-DD format
- `--top_n`: Number of top results to display (default: 20)
- `--window_size`: Window size for calculations (default: 20)
- `--drop_threshold`: Threshold for identifying significant drops (default: -0.03)

### Analysis Modes

- `--correlation`: Perform correlation analysis
- `--downtrend`: Perform downtrend analysis
- `--all_timeframes`: Analyze all timeframes
- `--no_cache`: Disable cache usage

### Optimization

- `--optimize`: Optimize request parameters based on analysis needs
- `--track_api_usage`: Track and display API usage statistics

## Avoiding Binance API Rate Limits

The tool includes several features to help avoid hitting Binance API rate limits:

1. **Caching**: Historical data is cached to avoid redundant API calls
2. **Batching**: Requests are processed in optimal batch sizes based on current rate limit usage
3. **Rate Limiting**: Automatic waiting between requests to stay within rate limits
4. **Proxy Support**: Option to rotate through multiple proxies to increase request capacity
5. **Exponential Backoff**: Automatic retry with increasing delays when rate limits are approached

## Output Files

The tool generates several output files:

- `correlation_[timeframe].csv`: CSV file with correlation values for each trading pair
- `correlation_[timeframe].png`: Bar chart showing top positive and negative correlations
- `correlation_heatmap.png`: Heatmap of correlations across timeframes
- `eth_downtrends.png`: Chart showing identified ETH downtrend periods
- `eth_stable_coins.png`: Chart showing performance of stable coins during downtrends
- `eth_rebound_best_coins.png`: Chart showing performance of coins that rebound well after downtrends
- `performance_scatter_plot.png`: Scatter plot of downtrend vs rebound performance

## Project Structure

- `main.py`: Entry point for the tool
- `config.py`: Configuration parameters
- `data_fetcher.py`: Handles data retrieval from Binance API
- `utils.py`: Utility functions for caching, rate limiting, and proxy management
- `analyzer.py`: Correlation analysis functionality
- `downtrend_analyzer.py`: Downtrend analysis functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Always do your own research before making investment decisions.
