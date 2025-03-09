# Cryptocurrency Correlation Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[繁體中文](README.md) | English

This tool uses historical data from Binance to compare the price correlation between ETH and all other cryptocurrency contracts. The analysis covers multiple timeframes (1 minute, 5 minutes, 15 minutes, 1 hour, and 4 hours). Designed specifically for traders and investors to help discover market correlation patterns.

<p align="center">
  <img src="correlation_heatmap.png" alt="Correlation Heatmap Example" width="600">
</p>

## Features

- Download historical price data from Binance
- Calculate price correlations between ETH and other cryptocurrencies
- Support for multiple timeframe analysis (1m, 5m, 15m, 1h, 4h)
- Visualize correlation results (bar charts and heatmaps)
- Data caching functionality to avoid redundant downloads
- Multi-threaded parallel processing for faster analysis
- Optimized data processing and correlation calculations

## Installation

1. Clone this repository:

```bash
git clone https://github.com/zhiyu1105/crypto-correlation-analyzer.git
cd crypto-correlation-analyzer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main program directly:

```bash
python crypto_correlation_analyzer.py
```

By default, the program will:
- Analyze data from the past 30 days
- Display the top 20 cryptocurrencies with the highest correlation to ETH
- Generate correlation charts for each timeframe
- Create a cross-timeframe correlation heatmap

### Quick Analysis Mode

If you only need to quickly analyze specific timeframes, you can use:

```bash
python quick_crypto_correlation.py
```

## Custom Analysis

To customize analysis parameters, you can modify the `main()` function in the `crypto_correlation_analyzer.py` file:

```python
# Set date range
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Modify number of days

# Analyze all timeframes
results = analyzer.analyze_all_timeframes(start_date, end_date, top_n=20, use_cache=True)  # Modify top_n parameter
```

## API Keys (Optional)

If you need higher API request limits, you can set Binance API keys in the `main()` function:

```python
# Set Binance API credentials
api_key = "YOUR_API_KEY"  # Replace with your API key
api_secret = "YOUR_API_SECRET"  # Replace with your API secret
```

## Output Files

The program generates the following files:
- `historical_data/` directory: Contains downloaded historical data CSV files
- `correlation_1m.csv`, `correlation_5m.csv`, etc.: Correlation data for each timeframe
- `correlation_1m.png`, `correlation_5m.png`, etc.: Correlation bar charts for each timeframe
- `correlation_heatmap.png`: Cross-timeframe correlation heatmap

## Project Structure

```
├── crypto_correlation_analyzer.py  # Main program
├── quick_crypto_correlation.py     # Quick analysis program
├── requirements.txt               # Dependency list
├── README.md                      # Chinese documentation
├── README_EN.md                   # English documentation
├── .gitignore                     # Git ignore file
├── historical_data/               # Historical data directory
├── correlation_*.csv              # Correlation data files
└── correlation_*.png              # Correlation chart files
```

## Notes

- When running for the first time, the program needs to download a large amount of data from Binance, which may take a considerable amount of time
- Using the cache functionality (`use_cache=True`) can speed up subsequent runs
- Binance API has request frequency limits, the program has added delays to avoid exceeding these limits
- For large data analysis, it is recommended to use a higher performance computer

## Contribution Guidelines

Contributions, issues, and feature requests are welcome! Feel free to submit an Issue or Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
