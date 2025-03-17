#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Module
Entry point for the cryptocurrency analysis tool
"""

import os
import argparse
from datetime import datetime, timedelta

import config
from data_fetcher import DataFetcher
from analyzer import CorrelationAnalyzer
from downtrend_analyzer import DowntrendAnalyzer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Correlation and Downtrend Analysis Tool')
    
    # Data fetching parameters
    parser.add_argument('--api_key', type=str, help='Binance API key')
    parser.add_argument('--api_secret', type=str, help='Binance API secret')
    parser.add_argument('--max_klines', type=int, default=config.DEFAULT_MAX_KLINES, 
                        help=f'Maximum number of klines to fetch (default: {config.DEFAULT_MAX_KLINES})')
    parser.add_argument('--request_delay', type=float, default=config.DEFAULT_REQUEST_DELAY, 
                        help=f'Delay between API requests in seconds (default: {config.DEFAULT_REQUEST_DELAY})')
    parser.add_argument('--max_workers', type=int, default=config.DEFAULT_MAX_WORKERS, 
                        help=f'Maximum number of concurrent workers (default: {config.DEFAULT_MAX_WORKERS})')
    parser.add_argument('--use_proxies', action='store_true', help='Use proxy rotation for API requests')
    parser.add_argument('--proxies', type=str, nargs='+', help='List of proxy URLs')
    parser.add_argument('--cache_expiry', type=int, default=config.DEFAULT_CACHE_EXPIRY, 
                        help=f'Cache expiry time in seconds (default: {config.DEFAULT_CACHE_EXPIRY})')
    
    # Analysis parameters
    parser.add_argument('--reference_symbol', type=str, default=config.DEFAULT_REFERENCE_SYMBOL, 
                        help=f'Reference trading pair symbol (default: {config.DEFAULT_REFERENCE_SYMBOL})')
    parser.add_argument('--timeframe', type=str, default='1h', choices=config.TIMEFRAMES.keys(),
                        help=f'Timeframe for analysis (default: 1h)')
    parser.add_argument('--days', type=int, default=30, 
                        help='Number of days to analyze (default: 30)')
    parser.add_argument('--start_date', type=str, 
                        help='Start date in YYYY-MM-DD format (default: days ago from today)')
    parser.add_argument('--end_date', type=str, 
                        help='End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--top_n', type=int, default=config.DEFAULT_TOP_N, 
                        help=f'Number of top results to display (default: {config.DEFAULT_TOP_N})')
    parser.add_argument('--window_size', type=int, default=config.DEFAULT_WINDOW_SIZE, 
                        help=f'Window size for calculations (default: {config.DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--drop_threshold', type=float, default=config.DEFAULT_DROP_THRESHOLD, 
                        help=f'Threshold for identifying significant drops (default: {config.DEFAULT_DROP_THRESHOLD})')
    
    # Analysis modes
    parser.add_argument('--correlation', action='store_true', help='Perform correlation analysis')
    parser.add_argument('--downtrend', action='store_true', help='Perform downtrend analysis')
    parser.add_argument('--all_timeframes', action='store_true', help='Analyze all timeframes')
    parser.add_argument('--no_cache', action='store_true', help='Disable cache usage')
    
    # Optimization
    parser.add_argument('--optimize', action='store_true', 
                        help='Optimize request parameters based on analysis needs')
    parser.add_argument('--track_api_usage', action='store_true', 
                        help='Track and display API usage statistics')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set up dates
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    
    print(f"Analysis period: {start_date} to {end_date}")
    
    # Optimize request parameters if requested
    if args.optimize:
        from utils import optimize_request_parameters
        optimized_params = optimize_request_parameters(args.days, args.timeframe, not args.no_cache)
        max_klines = optimized_params['max_klines']
        request_delay = optimized_params['request_delay']
        max_workers = optimized_params['max_workers']
    else:
        max_klines = args.max_klines
        request_delay = args.request_delay
        max_workers = args.max_workers
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(
        api_key=args.api_key,
        api_secret=args.api_secret,
        max_klines=max_klines,
        request_delay=request_delay,
        max_workers=max_workers,
        use_proxies=args.use_proxies,
        proxies=args.proxies,
        cache_expiry=args.cache_expiry
    )
    
    # Initialize analyzers
    correlation_analyzer = CorrelationAnalyzer(data_fetcher, args.reference_symbol)
    downtrend_analyzer = DowntrendAnalyzer(correlation_analyzer)
    
    # Perform requested analyses
    if args.all_timeframes and args.correlation:
        print("\n=== Analyzing correlations across all timeframes ===")
        results = correlation_analyzer.analyze_all_timeframes(
            start_date=start_date,
            end_date=end_date,
            top_n=args.top_n,
            use_cache=not args.no_cache
        )
        
        # Create correlation heatmap
        correlation_analyzer.create_correlation_heatmap(results, args.top_n)
    
    elif args.correlation:
        print(f"\n=== Analyzing correlations for {args.timeframe} timeframe ===")
        # Get all symbols
        all_symbols = data_fetcher.get_all_futures_symbols()
        usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        if args.reference_symbol not in usdt_symbols:
            usdt_symbols.append(args.reference_symbol)
        
        # Prioritize reference symbol
        if args.reference_symbol in usdt_symbols:
            usdt_symbols.remove(args.reference_symbol)
            prioritized_symbols = [args.reference_symbol] + usdt_symbols
        else:
            prioritized_symbols = usdt_symbols
        
        # Get data for all symbols
        data = data_fetcher.fetch_data_for_all_symbols(
            prioritized_symbols, args.timeframe, start_date, end_date, not args.no_cache
        )
        
        # Calculate correlations
        correlation = correlation_analyzer.calculate_correlation(data, args.reference_symbol)
        
        # Save correlations to CSV
        correlation.to_csv(f'correlation_{args.timeframe}.csv')
        
        # Visualize correlations
        correlation_analyzer.visualize_correlation(correlation, args.timeframe, args.top_n)
    
    if args.downtrend:
        print(f"\n=== Analyzing {args.reference_symbol} downtrends for {args.timeframe} timeframe ===")
        print(f"Analysis period: {start_date} to {end_date}")
        print(f"Drop threshold: {args.drop_threshold}, Window size: {args.window_size}")
        
        # 確保使用正確的閾值
        # 注意：如果用戶輸入的是-0.003，這表示-0.3%，可能需要調整
        drop_threshold = args.drop_threshold
        if abs(drop_threshold) < 0.01:
            print(f"Warning: Drop threshold {drop_threshold} seems very small.")
            print(f"This value represents {drop_threshold*100}% change.")
            print(f"If you meant {drop_threshold*100}%, continue. If you meant {drop_threshold}%, consider using {drop_threshold/100} instead.")
        
        try:
            results = downtrend_analyzer.analyze_downtrends(
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date,
                drop_threshold=drop_threshold,
                window_size=args.window_size,
                top_n=args.top_n,
                use_cache=not args.no_cache
            )
            
            # 如果沒有找到下跌趨勢，提供建議
            if len(results.get('downtrend_periods', [])) == 0:
                print("\n=== No downtrend periods found. Suggestions: ===")
                print("1. Try a less extreme drop threshold (e.g., -0.01 for -1%)")
                print("2. Try a different timeframe (e.g., 15m, 1h)")
                print("3. Try a different window size (e.g., 10, 30)")
                print("4. Check if the date range contains significant market movements")
                
                # 運行return_dist.py來獲取建議的參數
                print("\nConsider running return_dist.py to get recommended parameters:")
                print("python return_dist.py --symbol ETHUSDT --timeframe 1m --days 180")
        except Exception as e:
            print(f"Error during downtrend analysis: {e}")
            import traceback
            traceback.print_exc()
            print("\nTry adjusting parameters or check the data availability for the specified period.")
    
    # If no analysis mode specified, run correlation analysis
    if not (args.correlation or args.downtrend):
        print("\n=== Running default correlation analysis ===")
        # Get all symbols
        all_symbols = data_fetcher.get_all_futures_symbols()
        usdt_symbols = [s for s in all_symbols if s.endswith('USDT')]
        if args.reference_symbol not in usdt_symbols:
            usdt_symbols.append(args.reference_symbol)
        
        # Prioritize reference symbol
        if args.reference_symbol in usdt_symbols:
            usdt_symbols.remove(args.reference_symbol)
            prioritized_symbols = [args.reference_symbol] + usdt_symbols
        else:
            prioritized_symbols = usdt_symbols
        
        # Get data for all symbols
        data = data_fetcher.fetch_data_for_all_symbols(
            prioritized_symbols, args.timeframe, start_date, end_date, not args.no_cache
        )
        
        # Calculate correlations
        correlation = correlation_analyzer.calculate_correlation(data, args.reference_symbol)
        
        # Save correlations to CSV
        correlation.to_csv(f'correlation_{args.timeframe}.csv')
        
        # Visualize correlations
        correlation_analyzer.visualize_correlation(correlation, args.timeframe, args.top_n)
    
    # Track API usage if requested
    if args.track_api_usage:
        data_fetcher.rate_limiter.track_usage(show_details=True)
    
    print("\n=== Analysis complete ===")
    print(f"Results saved to current directory")


if __name__ == "__main__":
    main() 