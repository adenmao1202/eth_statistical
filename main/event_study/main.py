#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event Study Main Module
Cross-period robust validation for ETH price drop events
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import random

# Ensure parent directory is in module search path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary modules from parent directory
import config
from data_fetcher import DataFetcher
from coin_classifier import CoinClassifier
from time_series_validator import TimeSeriesValidator
from robust_event_analyzer import RobustEventAnalyzer
from robust_event_study import RobustEventStudy


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "event_study.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Rich console for pretty output
console = Console()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ETH Event Study with Cross-period Validation")
    
    # Analysis type
    parser.add_argument("--mode", type=str, default="robust_event_study", 
                       choices=["validation", "event_study", "robust_event_study", "robust_event_analyzer"],
                       help="Analysis mode to run")
    
    # Data parameters
    parser.add_argument("--timeframe", type=str, default="5m", help="Data timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--total_days", type=int, default=120, help="Total days to analyze")
    parser.add_argument("--period_length", type=int, default=30, help="Length of each validation period in days")
    parser.add_argument("--request_delay", type=float, default=0.1, help="Delay between API requests in seconds")
    parser.add_argument("--use_cache", action="store_true", help="Use cached data when available")
    
    # Classification parameters
    parser.add_argument("--classify_by_market_cap", action="store_true", help="Classify coins by market cap")
    parser.add_argument("--classify_by_volume", action="store_true", help="Classify coins by trading volume")
    parser.add_argument("--classify_by_volatility", action="store_true", help="Classify coins by price volatility")
    parser.add_argument("--classify_by_sector", action="store_true", help="Classify coins by sector")
    parser.add_argument("--create_random_baskets", action="store_true", help="Create random coin baskets for comparison")
    
    # ETH event parameters
    parser.add_argument("--drop_threshold", type=float, default=-0.01, help="ETH price drop threshold (negative value)")
    parser.add_argument("--consecutive_drops", type=int, default=1, help="Number of consecutive candles with drops")
    parser.add_argument("--volume_factor", type=float, default=1.2, help="Volume increase factor for ETH events")
    
    # Hypothesis testing parameters
    parser.add_argument("--pre_event_window", type=int, default=config.PRE_EVENT_WINDOW, 
                       help="Pre-event window in minutes")
    parser.add_argument("--post_event_window", type=int, default=config.POST_EVENT_WINDOW, 
                       help="Post-event window in minutes")
    parser.add_argument("--significance_level", type=float, default=config.SIGNIFICANCE_LEVEL, 
                       help="Statistical significance level")
    
    # Output parameters
    parser.add_argument("--output_suffix", type=str, default="", 
                       help="Suffix to add to output directory name")
    
    return parser.parse_args()

def run_validation(args):
    """Run cross-period robustness validation"""
    try:
        data_fetcher = DataFetcher(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            request_delay=args.request_delay,
            max_workers=config.MAX_WORKERS,
            use_proxies=config.USE_PROXIES
        )
        
        # Initialize coin classifier
        classifier = CoinClassifier(data_fetcher)
        
        # Create time series validator
        validator = TimeSeriesValidator(data_fetcher)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            # Get all available symbols
            fetch_symbols_task = progress.add_task("[cyan]Fetching available symbols...", total=1)
            all_symbols = data_fetcher._get_symbols()  # Use _get_symbols instead of fetch_all_symbols
            progress.update(fetch_symbols_task, completed=1)
            
            if not all_symbols:
                console.print("[bold red]Unable to fetch trading pairs, please check network connection or API status[/bold red]")
                return
                
            console.print(f"[green]Found {len(all_symbols)} trading pairs[/green]")
            
            # Classify coins based on options
            classify_task = progress.add_task("[cyan]Classifying coins...", total=1)
            
            classifications = {}
            
            # Remove stablecoins
            stable_coins = classifier.get_stable_coins(all_symbols)
            filtered_symbols = [s for s in all_symbols if s not in stable_coins]
            console.print(f"[yellow]Excluded {len(stable_coins)} stablecoins[/yellow]")
            
            # Classify by market cap
            if args.classify_by_market_cap:
                try:
                    market_cap_tiers = classifier.classify_by_market_cap(filtered_symbols)
                    classifications['market_cap'] = market_cap_tiers
                    console.print("[green]Completed market cap classification[/green]")
                except Exception as e:
                    logger.error(f"Market cap classification failed: {e}")
                    console.print(f"[bold yellow]Market cap classification failed: {e}, using simple random grouping instead[/bold yellow]")
                    # Use simple grouping as fallback
                    sampled_symbols = filtered_symbols[:100] if len(filtered_symbols) > 100 else filtered_symbols
                    classifications['market_cap'] = {'all_market_cap': sampled_symbols}
            
            # Classify by volume
            if args.classify_by_volume:
                try:
                    volume_tiers = classifier.classify_by_volume(
                        filtered_symbols, 
                        days=min(30, args.period_length),
                        timeframe=args.timeframe
                    )
                    classifications['volume'] = volume_tiers
                    console.print("[green]Completed volume classification[/green]")
                except Exception as e:
                    logger.error(f"Volume classification failed: {e}")
                    console.print(f"[bold yellow]Volume classification failed: {e}[/bold yellow]")
            
            # Classify by volatility
            if args.classify_by_volatility:
                try:
                    volatility_tiers = classifier.classify_by_volatility(
                        filtered_symbols,
                        days=min(30, args.period_length),
                        timeframe=args.timeframe
                    )
                    classifications['volatility'] = volatility_tiers
                    console.print("[green]Completed volatility classification[/green]")
                except Exception as e:
                    logger.error(f"Volatility classification failed: {e}")
                    console.print(f"[bold yellow]Volatility classification failed: {e}[/bold yellow]")
            
            # Create random baskets
            if args.create_random_baskets:
                try:
                    random_baskets = classifier.create_index_baskets(filtered_symbols)
                    classifications['random_baskets'] = random_baskets
                    console.print("[green]Completed creating random baskets[/green]")
                except Exception as e:
                    logger.error(f"Failed to create random baskets: {e}")
                    console.print(f"[bold yellow]Failed to create random baskets: {e}[/bold yellow]")
            
            # If no classification method specified, use all symbols
            if not classifications:
                # Limit number to avoid API limitations
                max_symbols = 100
                sample_size = min(max_symbols, len(filtered_symbols))
                if sample_size < len(filtered_symbols):
                    sampled_symbols = random.sample(filtered_symbols, sample_size)
                else:
                    sampled_symbols = filtered_symbols
                    
                classifications['all'] = {'all_symbols': sampled_symbols}
                console.print(f"[yellow]No classification method specified, using {sample_size} random coins[/yellow]")
            
            progress.update(classify_task, completed=1)
            
            # Set ETH event parameters
            eth_event_params = {
                'drop_threshold': args.drop_threshold,
                'consecutive_drops': args.consecutive_drops,
                'volume_factor': args.volume_factor
            }
            
            # Process each classification
            validation_task = progress.add_task("[cyan]Running validation across time periods...", total=len(classifications))
            validation_results = {}
            
            for class_type, tiers in classifications.items():
                validation_results[class_type] = {}
                
                for tier_name, symbols in tiers.items():
                    console.print(f"\n[bold blue]Analyzing {class_type} - {tier_name} ({len(symbols)} coins)[/bold blue]")
                    
                    try:
                        # Run cross-period validation
                        results_df = validator.validate_across_time_periods(
                            symbols=symbols,
                            total_days=args.total_days,
                            period_length=args.period_length,
                            timeframe=args.timeframe,
                            eth_event_params=eth_event_params
                        )
                        
                        # Further analyze performance distributions
                        top_symbols = results_df.head(10)['symbol'].tolist()
                        validator.compare_performance_distributions(
                            symbols=top_symbols,
                            num_periods=args.total_days // args.period_length,
                            period_length=args.period_length,
                            timeframe=args.timeframe
                        )
                        
                        validation_results[class_type][tier_name] = results_df
                        
                        # Output top stable coins
                        console.print(f"\n[bold green]Top stable coins for {class_type} - {tier_name}:[/bold green]")
                        
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("Rank")
                        table.add_column("Symbol")
                        table.add_column("Stability Score")
                        table.add_column("Mean Performance")
                        table.add_column("StdDev")
                        table.add_column("Period Coverage")
                        
                        for i, (_, row) in enumerate(results_df.head(10).iterrows()):
                            table.add_row(
                                str(i+1),
                                row['symbol'],
                                f"{row['weighted_score']:.4f}",
                                f"{row['mean_performance']:.4f}",
                                f"{row['performance_std']:.4f}",
                                f"{row['period_coverage']:.2f}"
                            )
                        
                        console.print(table)
                        
                    except Exception as e:
                        logger.error(f"Error in validation for {class_type} - {tier_name}: {e}")
                        console.print(f"[bold red]Error in validation for {class_type} - {tier_name}: {e}[/bold red]")
                
                progress.update(validation_task, advance=1)
            
            # Summarize top stable coins across all classifications
            console.print("\n[bold green]===== Validation Summary =====[/bold green]")
            
            # Calculate average stability of top 10 coins across all classifications
            top10_stability = []
            
            for class_type, tiers_results in validation_results.items():
                for tier_name, results_df in tiers_results.items():
                    if len(results_df) >= 10:
                        top10_avg = results_df.head(10)['weighted_score'].mean()
                        top10_stability.append(top10_avg)
                        console.print(
                            f"[yellow]{class_type} - {tier_name}:[/yellow] "
                            f"Top10 average stability: {top10_avg:.4f}"
                        )
    except Exception as e:
        logger.error(f"Error running cross-period robustness validation: {e}")
        console.print(f"[bold red]Error running cross-period robustness validation: {e}[/bold red]")

def run_event_study(args):
    """Run the robust event study with hypothesis testing integration"""
    try:
        # 修改API检查逻辑，允许使用公共API
        # 初始化data_fetcher，即使没有API密钥也能继续
        data_fetcher = DataFetcher(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            request_delay=args.request_delay,
            max_workers=config.MAX_WORKERS,
            use_proxies=config.USE_PROXIES
        )
        
        # Initialize robust event study
        event_study = RobustEventStudy(data_fetcher)
        
        # Determine classification methods
        classification_methods = []
        if args.classify_by_market_cap:
            classification_methods.append("market_cap")
        if args.classify_by_volume:
            classification_methods.append("volume")
        if args.classify_by_volatility:
            classification_methods.append("volatility")
        if args.create_random_baskets:
            classification_methods.append("random")
            
        # If no methods specified, use default
        if not classification_methods:
            classification_methods = ["market_cap"]
            console.print("[yellow]No classification method specified, using market cap classification by default[/yellow]")
        
        # Set ETH event parameters
        eth_event_params = {
            'drop_threshold': args.drop_threshold,
            'consecutive_drops': args.consecutive_drops,
            'volume_factor': args.volume_factor
        }
        
        # Run analysis
        results = event_study.run_analysis(
            classification_methods=classification_methods,
            total_days=args.total_days,
            period_length=args.period_length,
            timeframe=args.timeframe,
            eth_event_params=eth_event_params,
            output_suffix=args.output_suffix
        )
        
        # Display output directory
        if 'output_dir' in results:
            console.print(f"\n[bold green]Analysis completed! Results saved to: {results['output_dir']}[/bold green]")
        
    except Exception as e:
        logger.error(f"Error running robust event study: {e}")
        console.print(f"[bold red]Error running robust event study: {e}[/bold red]")

def run_robust_event_analyzer(args):
    """Run the robust event analyzer with hypothesis testing"""
    try:
   
        data_fetcher = DataFetcher(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            request_delay=args.request_delay,
            max_workers=config.MAX_WORKERS,
            use_proxies=config.USE_PROXIES
        )
        
        # Initialize robust event analyzer
        analyzer = RobustEventAnalyzer(data_fetcher)
        
        # Initialize coin classifier to get symbols
        classifier = CoinClassifier(data_fetcher)
        
        # Get symbols to analyze
        all_symbols = data_fetcher._get_symbols()
        if not all_symbols:
            console.print("[bold red]Unable to fetch trading pairs. Please check your network connection and API settings.[/bold red]")
            return
            
        # Remove stablecoins
        stable_coins = classifier.get_stable_coins(all_symbols)
        filtered_symbols = [s for s in all_symbols if s not in stable_coins]
        
        # Sample random symbols if too many
        max_symbols = 100
        if len(filtered_symbols) > max_symbols:
            symbols = random.sample(filtered_symbols, max_symbols)
        else:
            symbols = filtered_symbols
            
        console.print(f"[green]Analyzing {len(symbols)} coins out of {len(filtered_symbols)} non-stablecoin pairs[/green]")
        
        # Set ETH event parameters
        eth_event_params = {
            'drop_threshold': args.drop_threshold,
            'consecutive_drops': args.consecutive_drops,
            'volume_factor': args.volume_factor
        }
        
        # Run analysis
        results = analyzer.run_robust_analysis(
            symbols=symbols,
            total_days=args.total_days,
            period_length=args.period_length,
            timeframe=args.timeframe,
            eth_event_params=eth_event_params
        )
        
        console.print("[bold green]Robust event analysis completed![/bold green]")
        
    except Exception as e:
        logger.error(f"Error running robust event analyzer: {e}")
        console.print(f"[bold red]Error running robust event analyzer: {e}[/bold red]")

def main():
    """Main function"""
    console.print("[bold blue]ETH Event Study with Cross-period Validation[/bold blue]")
    
    # Parse command line arguments
    args = parse_args()
    
    # Run based on selected mode
    if args.mode == "validation":
        run_validation(args)
    elif args.mode == "event_study":
        run_event_study(args)
    elif args.mode == "robust_event_study":
        run_event_study(args)
    elif args.mode == "robust_event_analyzer":
        run_robust_event_analyzer(args)
    else:
        console.print(f"[bold red]Unknown mode: {args.mode}[/bold red]")
        return
    
    console.print("[bold green]Analysis completed![/bold green]")

if __name__ == "__main__":
    main() 