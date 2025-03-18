# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Usage examples: ( in bash )
# # Use a smaller downtrend threshold (e.g., -3% instead of -5%)
# python local_eth_analyzer.py --downtrend -0.03

# # Use a smaller rebound threshold (e.g., 1% instead of 3%)
# python local_eth_analyzer.py --rebound 0.01

# # Use a larger window size
# python local_eth_analyzer.py --window 10

# # Analyze a longer time range
# python local_eth_analyzer.py --days 60
# """

# """
# Local Data ETH Downtrend Stable Cryptocurrency Analyzer
# This script analyzes the stable cryptocurrencies during ETH downtrend using local CSV files,
# avoiding data downloads from the Binance API.
# """

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# from tqdm import tqdm
# import warnings
# import argparse
# import glob

# warnings.filterwarnings('ignore')


# def parse_arguments():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Analyze stable cryptocurrencies during ETH downtrend using local data')
    
#     parser.add_argument('--data_dir', type=str, default='historical_data',
#                         help='Local data directory (default: historical_data)')
    
#     parser.add_argument('--days', type=int, default=60,
#                         help='Number of days to analyze (default: 30 days of data )')
    
#     parser.add_argument('--downtrend', type=float, default=-0.01,
#                         help='ETH downtrend threshold, e.g., -0.1 means a 10% drop (default: -0.1)')
    
#     parser.add_argument('--rebound', type=float, default=0.02,
#                         help='ETH rebound threshold, e.g., 0.05 means a 5% rebound (default: 0.05)')
    
#     parser.add_argument('--window', type=int, default=10,
#                         help='Sliding window size used for identifying trends (default: 5)')
    
#     parser.add_argument('--top', type=int, default=10,
#                         help='Show the top N best-performing cryptocurrencies (default: 10)')
    
#     parser.add_argument('--timeframe', type=str, default='1m',
#                         help='Timeframe to use (default: 1m)')
    
#     return parser.parse_args()


# def load_local_data(data_dir, symbol, timeframe):
#     """Load data from a local file"""
#     filename = f"{data_dir}/{symbol}_{timeframe}.csv"
    
#     if os.path.exists(filename):
#         df = pd.read_csv(filename)
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         df.set_index('timestamp', inplace=True)
#         return df
    
#     return None


# def get_available_symbols(data_dir, timeframe):
#     """Get all available symbols from local data"""
#     pattern = f"{data_dir}/*_{timeframe}.csv"
#     files = glob.glob(pattern)
#     symbols = [os.path.basename(f).split('_')[0] for f in files]
#     return symbols


# def analyze_eth_downtrend_resilience(data_dir, timeframe, start_date, end_date, 
#                                      downtrend_threshold=-0.05, rebound_threshold=0.03, 
#                                      window_size=5, top_n=10):
#     """
#     Analyze the stable performance of cryptocurrencies during ETH downtrend,
#     and their behavior during the ETH rebound.
#     This analysis uses local data rather than downloading from Binance.
#     """
#     print(f"\nAnalyzing stable cryptocurrencies during ETH downtrend...")
    
#     eth_symbol = 'ETHUSDT'
    
#     # 1. Load historical ETH data
#     eth_data = load_local_data(data_dir, eth_symbol, timeframe)
    
#     if eth_data is None:
#         print(f"Error: Could not find ETH data file {data_dir}/{eth_symbol}_{timeframe}.csv")
#         return None, None, None
    
#     # Filter by date range
#     start_dt = pd.to_datetime(start_date)
#     end_dt = pd.to_datetime(end_date)
#     eth_data = eth_data[(eth_data.index >= start_dt) & (eth_data.index <= end_dt)]
    
#     if len(eth_data) < window_size + 1:
#         print(f"Error: Insufficient ETH data, at least {window_size + 1} data points are required")
#         return None, None, None
    
#     # 2. Identify ETH downtrend periods
#     eth_data['pct_change'] = eth_data['close'].pct_change(window_size)
    
#     # Find periods where ETH has a significant drop
#     downtrend_periods = []
#     current_downtrend = None
    
#     # Debug information: print statistics of ETH percentage changes
#     print("ETH Price Change Statistics:")
#     print(f"Min: {eth_data['pct_change'].min():.2%}")
#     print(f"Max: {eth_data['pct_change'].max():.2%}")
#     print(f"Mean: {eth_data['pct_change'].mean():.2%}")
#     print(f"Median: {eth_data['pct_change'].median():.2%}")
#     print(f"Downtrend threshold: {downtrend_threshold:.2%}")
#     print(f"Rebound threshold: {rebound_threshold:.2%}")
    
#     for idx, row in eth_data.iterrows():
#         if row['pct_change'] <= downtrend_threshold and current_downtrend is None:
#             # Start a new downtrend period
#             current_downtrend = {'start': idx, 'prices': [row['close']]}
#         elif current_downtrend is not None:
#             # Continue the current downtrend period
#             current_downtrend['prices'].append(row['close'])
            
#             # Check if the rebound threshold is reached
#             if row['pct_change'] >= rebound_threshold:
#                 # End the downtrend period as ETH starts rebounding
#                 current_downtrend['end'] = idx
#                 current_downtrend['end_price'] = row['close']
#                 current_downtrend['start_price'] = current_downtrend['prices'][0]
#                 current_downtrend['lowest_price'] = min(current_downtrend['prices'])
#                 current_downtrend['lowest_idx'] = eth_data[eth_data['close'] == current_downtrend['lowest_price']].index[0]
#                 current_downtrend['drop_pct'] = (current_downtrend['lowest_price'] / current_downtrend['start_price'] - 1) * 100
#                 current_downtrend['rebound_pct'] = (current_downtrend['end_price'] / current_downtrend['lowest_price'] - 1) * 100
                
#                 # Only add this period if the drop and rebound are significant
#                 if current_downtrend['drop_pct'] <= downtrend_threshold * 100 and current_downtrend['rebound_pct'] >= rebound_threshold * 100:
#                     downtrend_periods.append(current_downtrend)
#                     print(f"Found downtrend period: {current_downtrend['start']} to {current_downtrend['end']}")
#                     print(f"  Drop: {current_downtrend['drop_pct']:.2f}%, Rebound: {current_downtrend['rebound_pct']:.2f}%")
                
#                 current_downtrend = None
    
#     # If the last downtrend period hasn't ended, but a significant drop exists
#     if current_downtrend is not None and len(current_downtrend['prices']) > window_size:
#         current_downtrend['end'] = eth_data.index[-1]
#         current_downtrend['end_price'] = eth_data['close'].iloc[-1]
#         current_downtrend['start_price'] = current_downtrend['prices'][0]
#         current_downtrend['lowest_price'] = min(current_downtrend['prices'])
#         current_downtrend['lowest_idx'] = eth_data[eth_data['close'] == current_downtrend['lowest_price']].index[0]
#         current_downtrend['drop_pct'] = (current_downtrend['lowest_price'] / current_downtrend['start_price'] - 1) * 100
#         current_downtrend['rebound_pct'] = (current_downtrend['end_price'] / current_downtrend['lowest_price'] - 1) * 100
        
#         # Only add the period if the drop is significant
#         if current_downtrend['drop_pct'] <= downtrend_threshold * 100:
#             downtrend_periods.append(current_downtrend)
#             print(f"Found downtrend period (unfinished): {current_downtrend['start']} to {current_downtrend['end']}")
#             print(f"  Drop: {current_downtrend['drop_pct']:.2f}%, Rebound: {current_downtrend['rebound_pct']:.2f}%")
    
#     print(f"Found {len(downtrend_periods)} ETH downtrend period(s)")
    
#     if len(downtrend_periods) == 0:
#         # If no downtrend period is found, try relaxing the conditions
#         print("No qualifying ETH downtrend period found in the specified range. Trying to relax conditions...")
        
#         # Attempt to find the largest drop
#         eth_data['rolling_min'] = eth_data['close'].rolling(window=window_size).min()
#         eth_data['rolling_max'] = eth_data['close'].rolling(window=window_size).max()
#         eth_data['drop_pct'] = (eth_data['rolling_min'] / eth_data['rolling_max'] - 1) * 100
        
#         # Identify the index of the maximum drop
#         max_drop_idx = eth_data['drop_pct'].idxmin()
#         if max_drop_idx is not None:
#             # Find the start and end times for the drop
#             max_drop_window = eth_data.loc[:max_drop_idx].tail(window_size * 2)
#             start_idx = max_drop_window['close'].idxmax()
#             end_idx = eth_data.loc[max_drop_idx:].head(window_size * 2)['close'].idxmax()
            
#             if start_idx is not None and end_idx is not None:
#                 # Create a downtrend period
#                 period = {
#                     'start': start_idx,
#                     'end': end_idx,
#                     'start_price': eth_data.loc[start_idx, 'close'],
#                     'lowest_price': eth_data.loc[max_drop_idx, 'close'],
#                     'lowest_idx': max_drop_idx,
#                     'end_price': eth_data.loc[end_idx, 'close'],
#                     'drop_pct': (eth_data.loc[max_drop_idx, 'close'] / eth_data.loc[start_idx, 'close'] - 1) * 100,
#                     'rebound_pct': (eth_data.loc[end_idx, 'close'] / eth_data.loc[max_drop_idx, 'close'] - 1) * 100
#                 }
                
#                 downtrend_periods.append(period)
#                 print(f"Found maximum downtrend period: {period['start']} to {period['end']}")
#                 print(f"  Drop: {period['drop_pct']:.2f}%, Rebound: {period['rebound_pct']:.2f}%")
        
#         if len(downtrend_periods) == 0:
#             print("Even after relaxing conditions, no valid downtrend period was found")
#             return None, None, None
    
#     # 3. Get all available symbols from local data
#     usdt_symbols = get_available_symbols(data_dir, timeframe)
#     print(f"Found {len(usdt_symbols)} available local symbol datasets")
    
#     # Store performance for each coin during the downtrend periods
#     downtrend_performance = {}
#     rebound_performance = {}
    
#     for period in downtrend_periods:
#         period_start = period['start']
#         period_end = period['end']
#         period_lowest = period['lowest_idx']
        
#         print(f"\nAnalyzing downtrend period: {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}")
#         print(f"ETH drop: {period['drop_pct']:.2f}%, Rebound: {period['rebound_pct']:.2f}%")
#         print(f"Lowest point time: {period_lowest.strftime('%Y-%m-%d %H:%M')}")
        
#         # Analyze performance for each coin
#         for symbol in tqdm(usdt_symbols, desc="Analyzing coin performance"):
#             if symbol == eth_symbol:
#                 continue
            
#             try:
#                 # Load historical data for the coin
#                 coin_data = load_local_data(data_dir, symbol, timeframe)
                
#                 if coin_data is None or len(coin_data) < 2:
#                     continue
                
#                 # Calculate performance during the downtrend period
#                 try:
#                     # Find the price at the corresponding time points
#                     # Note: Local data might not have an exact match, so we choose the closest
#                     start_mask = coin_data.index <= period_start
#                     if not any(start_mask):
#                         continue
#                     start_price = coin_data[start_mask].iloc[-1]['close']
                    
#                     lowest_mask = coin_data.index <= period_lowest
#                     if not any(lowest_mask):
#                         continue
#                     lowest_time_price = coin_data[lowest_mask].iloc[-1]['close']
                    
#                     end_mask = coin_data.index <= period_end
#                     if not any(end_mask):
#                         continue
#                     end_price = coin_data[end_mask].iloc[-1]['close']
                    
#                     # Calculate performance during the downtrend (from start to lowest point)
#                     downtrend_pct = (lowest_time_price / start_price - 1) * 100
                    
#                     # Calculate performance during the ETH rebound period (from lowest point to end)
#                     rebound_pct = (end_price / lowest_time_price - 1) * 100
                    
#                     # Debug info for selected symbols
#                     if symbol in ['BTCUSDT', 'BNBUSDT', 'XRPUSDT']:
#                         print(f"{symbol} performance:")
#                         print(f"  Start Price: {start_price:.4f}, Lowest Price: {lowest_time_price:.4f}, End Price: {end_price:.4f}")
#                         print(f"  Downtrend Performance: {downtrend_pct:.2f}%, Rebound Performance: {rebound_pct:.2f}%")
                    
#                     # Store the results
#                     if symbol not in downtrend_performance:
#                         downtrend_performance[symbol] = []
                    
#                     if symbol not in rebound_performance:
#                         rebound_performance[symbol] = []
                    
#                     downtrend_performance[symbol].append(downtrend_pct)
#                     rebound_performance[symbol].append(rebound_pct)
                    
#                 except (KeyError, IndexError) as e:
#                     # Mismatch in time points, skip this coin
#                     continue
                    
#             except Exception as e:
#                 print(f"Error analyzing {symbol}: {e}")
#                 continue
    
#     # 4. Calculate average performance
#     avg_downtrend = {}
#     avg_rebound = {}
    
#     for symbol in downtrend_performance:
#         if len(downtrend_performance[symbol]) > 0:
#             avg_downtrend[symbol] = sum(downtrend_performance[symbol]) / len(downtrend_performance[symbol])
        
#     for symbol in rebound_performance:
#         if len(rebound_performance[symbol]) > 0:
#             avg_rebound[symbol] = sum(rebound_performance[symbol]) / len(rebound_performance[symbol])
    
#     # Check if all rebound performances are zero
#     if all(value == 0 for value in avg_rebound.values()):
#         print("Warning: All coins show 0 rebound performance, which may indicate a data issue or a flaw in the analysis logic")
        
#         # Attempt an alternative calculation for each coin's rebound performance during ETH rebound period
#         print("Trying alternative method to calculate rebound performance...")
        
#         for period in downtrend_periods:
#             period_lowest = period['lowest_idx']
#             period_end = period['end']
            
#             for symbol in usdt_symbols:
#                 if symbol == eth_symbol:
#                     continue
                
#                 try:
#                     coin_data = load_local_data(data_dir, symbol, timeframe)
#                     if coin_data is None or len(coin_data) < 2:
#                         continue
                    
#                     # Filter data for the rebound period
#                     rebound_data = coin_data[(coin_data.index >= period_lowest) & (coin_data.index <= period_end)]
                    
#                     if len(rebound_data) >= 2:
#                         # Calculate the price change over the entire rebound period
#                         start_price = rebound_data.iloc[0]['close']
#                         end_price = rebound_data.iloc[-1]['close']
#                         rebound_pct = (end_price / start_price - 1) * 100
                        
#                         if symbol not in avg_rebound:
#                             avg_rebound[symbol] = rebound_pct
#                         else:
#                             avg_rebound[symbol] = (avg_rebound[symbol] + rebound_pct) / 2
                        
#                 except Exception as e:
#                     continue
    
#     # 5. Identify the coins with the most stable performance during ETH downtrend (i.e., minimal drop or even an increase)
#     stable_coins = pd.Series(avg_downtrend).sort_values(ascending=False)
    
#     # 6. Identify the coins with the best performance during ETH rebound
#     best_rebounders = pd.Series(avg_rebound).sort_values(ascending=False)
    
#     # 7. Identify the coins that are both stable and have a good rebound
#     combined_score = {}
    
#     for symbol in stable_coins.index:
#         if symbol in best_rebounders.index:
#             # Combined score = downtrend performance + rebound performance
#             combined_score[symbol] = stable_coins[symbol] + best_rebounders[symbol]
    
#     best_combined = pd.Series(combined_score).sort_values(ascending=False)
    
#     # 8. Create a results DataFrame
#     result_df = pd.DataFrame({
#         'Downtrend Performance (%)': stable_coins,
#         'Rebound Performance (%)': best_rebounders,
#         'Combined Score': best_combined
#     })
    
#     # 9. Visualize the results
#     # 9.1 Top N coins with the best downtrend performance
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x=stable_coins.head(top_n).values, y=stable_coins.head(top_n).index, palette='viridis')
#     plt.title(f'Top {top_n} Cryptocurrencies with the Most Stable Performance during ETH Downtrend')
#     plt.xlabel('Price Change Percentage (%)')
#     plt.ylabel('Symbol')
#     plt.tight_layout()
#     plt.savefig('eth_downtrend_stable_coins.png', dpi=300)
#     plt.show()
    
#     # 9.2 Top N coins with the best rebound performance
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x=best_rebounders.head(top_n).values, y=best_rebounders.head(top_n).index, palette='viridis')
#     plt.title(f'Top {top_n} Cryptocurrencies with the Best Rebound Performance during ETH Rebound')
#     plt.xlabel('Price Change Percentage (%)')
#     plt.ylabel('Symbol')
#     plt.tight_layout()
#     plt.savefig('eth_rebound_best_coins.png', dpi=300)
#     plt.show()
    
#     # 9.3 Top N coins with the best combined performance
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x=best_combined.head(top_n).values, y=best_combined.head(top_n).index, palette='viridis')
#     plt.title(f'Top {top_n} Cryptocurrencies with the Best Combined Performance in the ETH Market Cycle')
#     plt.xlabel('Combined Score')
#     plt.ylabel('Symbol')
#     plt.tight_layout()
#     plt.savefig('eth_cycle_best_coins.png', dpi=300)
#     plt.show()
    
#     # 10. Save results to CSV
#     result_df.to_csv('eth_market_cycle_analysis.csv')
    
#     print("\nAnalysis complete!")
#     print(f"Top {top_n} cryptocurrencies with the most stable performance during ETH downtrend:")
#     print(stable_coins.head(top_n))
    
#     print(f"\nTop {top_n} cryptocurrencies with the best rebound performance during ETH rebound:")
#     print(best_rebounders.head(top_n))
    
#     print(f"\nTop {top_n} cryptocurrencies with the best combined performance in the ETH market cycle:")
#     print(best_combined.head(top_n))
    
#     return stable_coins, best_rebounders, best_combined


# def main():
#     """Main function"""
#     # Parse command line arguments
#     args = parse_arguments()
    
#     # Set date range
#     end_date = datetime.now().strftime('%Y-%m-%d')
#     start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
#     print("Analyzing stable cryptocurrencies during ETH downtrend using local data")
#     print(f"Data directory: {args.data_dir}")
#     print(f"Timeframe: {args.timeframe}")
#     print(f"Date range: {start_date} to {end_date}")
#     print(f"ETH Downtrend Threshold: {args.downtrend * 100}%")
#     print(f"ETH Rebound Threshold: {args.rebound * 100}%")
#     print(f"Sliding Window Size: {args.window}")
#     print(f"Displaying top {args.top} best-performing coins")
    
#     # Analyze stable cryptocurrencies during ETH downtrend
#     stable_coins, best_rebounders, best_combined = analyze_eth_downtrend_resilience(
#         args.data_dir,
#         args.timeframe,
#         start_date, 
#         end_date, 
#         downtrend_threshold=args.downtrend,
#         rebound_threshold=args.rebound,
#         window_size=args.window,
#         top_n=args.top
#     )
    
#     if stable_coins is None:
#         print("No qualifying ETH downtrend period found within the specified date range")
#         return
    
#     print("\nAnalysis complete!")
#     print("Results have been saved to 'eth_market_cycle_analysis.csv'")
#     print("Charts have been saved as 'eth_downtrend_stable_coins.png', 'eth_rebound_best_coins.png', and 'eth_cycle_best_coins.png'")


# if __name__ == "__main__":
#     main()
