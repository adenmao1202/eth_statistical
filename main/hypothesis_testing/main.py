#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hypothesis Testing Main Module
Entry point for the cryptocurrency hypothesis testing analysis
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Ensure parent directory is in module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import required modules
try:
    import config
    from data_fetcher import DataFetcher
    from hypothesis_testing import HypothesisTesting
except ImportError:
    # If running in subdirectory, try relative import
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(parent_dir, ".."))
    import config
    from data_fetcher import DataFetcher
    from hypothesis_testing import HypothesisTesting

# Initialize Rich console
console = Console()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Hypothesis Testing')
    
    # Data fetching parameters
    parser.add_argument('--api_key', type=str, help='Binance API key')
    parser.add_argument('--api_secret', type=str, help='Binance API secret')
    parser.add_argument('--max_klines', type=int, default=config.DEFAULT_MAX_KLINES, 
                       help=f'Maximum number of klines to fetch per request (default: {config.DEFAULT_MAX_KLINES})')
    parser.add_argument('--request_delay', type=float, default=config.DEFAULT_REQUEST_DELAY, 
                       help=f'Delay between API requests in seconds (default: {config.DEFAULT_REQUEST_DELAY})')
    parser.add_argument('--max_workers', type=int, default=config.DEFAULT_MAX_WORKERS, 
                       help=f'Maximum number of concurrent workers (default: {config.DEFAULT_MAX_WORKERS})')
    parser.add_argument('--use_proxies', action='store_true', help='Use proxy rotation for API requests')
    parser.add_argument('--no_cache', action='store_true', help='Disable cache usage')
    
    # Hypothesis testing parameters
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--timeframe', type=str, default='1m', choices=config.TIMEFRAMES.keys(),
                       help=f'Timeframe for analysis (default: 1m)')
    parser.add_argument('--drop_threshold', type=float, default=config.MIN_DROP_PCT,
                       help=f'Threshold for identifying ETH price drops (default: {config.MIN_DROP_PCT})')
    parser.add_argument('--pre_event_window', type=int, default=config.PRE_EVENT_WINDOW,
                       help=f'Pre-event window in minutes (default: {config.PRE_EVENT_WINDOW})')
    parser.add_argument('--post_event_window', type=int, default=config.POST_EVENT_WINDOW,
                       help=f'Post-event window in minutes (default: {config.POST_EVENT_WINDOW})')
    parser.add_argument('--top_n', type=int, default=config.DEFAULT_TOP_N,
                       help=f'Number of top cryptocurrencies to analyze (default: {config.DEFAULT_TOP_N})')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--show_cache_info', action='store_true', help='Display cache information')
    parser.add_argument('--clean_cache', action='store_true', help='Clean expired cache files')
    parser.add_argument('--force_clean', action='store_true', help='Force clean all cache files')
    # Additional parameters
    parser.add_argument('--consecutive_drops', type=int, default=1, 
                       help='Number of consecutive K-line drops (default: 1)')
    parser.add_argument('--volume_factor', type=float, default=1.5, 
                       help='Volume multiple threshold (default: 1.5)')
    parser.add_argument('--rebound_threshold', type=float, default=0.005, 
                       help='Rebound threshold definition (default: 0.5%%)')
    parser.add_argument('--take_profit_pct', type=float, default=0.03, 
                       help='Take profit percentage (default: 3%%)')
    parser.add_argument('--stop_loss_pct', type=float, default=0.02, 
                       help='Stop loss percentage (default: 2%%)')
    
    return parser.parse_args()

def display_cache_info(data_fetcher):
    """Display cache statistics information"""
    cache_stats = data_fetcher.get_cache_stats()
    
    console.print(Panel(
        f"""[bold cyan]Cache Statistics[/bold cyan]
        
[green]Total cache files:[/green] {cache_stats.get('total_files', 'N/A')}
[green]Total cache size:[/green] {cache_stats.get('total_size_mb', 0):.2f} MB
[green]Expired files:[/green] {cache_stats.get('expired_files', 'N/A')}
[green]Cache expiry days:[/green] {cache_stats.get('cache_expiry_days', 'N/A')}
[green]Cache directory:[/green] {cache_stats.get('cache_dir', 'N/A')}
""",
        border_style="blue",
        expand=False
    ))
    
    # Display newest and oldest cache information if available
    newest = cache_stats.get('newest_cache')
    oldest = cache_stats.get('oldest_cache')
    
    if newest:
        console.print(f"[green]Newest cache file:[/green] {os.path.basename(newest['path'])}")
        console.print(f"[green]Modified time:[/green] {newest['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"[green]File size:[/green] {newest['size'] / 1024:.2f} KB")
    
    if oldest:
        console.print(f"\n[yellow]Oldest cache file:[/yellow] {os.path.basename(oldest['path'])}")
        console.print(f"[yellow]Modified time:[/yellow] {oldest['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"[yellow]File size:[/yellow] {oldest['size'] / 1024:.2f} KB")

def clean_cache(data_fetcher, force_clean=False):
    """Clean cache"""
    result = data_fetcher.clean_expired_cache(force_clean)
    
    if 'error' in result:
        console.print(f"[bold red]Cache cleaning failed: {result['error']}[/bold red]")
    else:
        if force_clean:
            console.print(f"[bold green]Force cleaning completed![/bold green]")
        else:
            console.print(f"[bold green]Expired cache cleaning completed![/bold green]")
        
        console.print(f"[green]Removed files:[/green] {result['removed_files']}")
        console.print(f"[green]Reclaimed space:[/green] {result['reclaimed_space_mb']:.2f} MB")

def main():
    """Run the hypothesis testing analysis"""
    args = parse_arguments()
    
    # Set up dates
    if args.start_date:
        start_date = args.start_date
    else:
        # Calculate start date based on days
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    
    # Create output directory, considering execution from different locations
    output_dir_name = f"{args.timeframe}_{args.days}days_drop{abs(args.drop_threshold*100)}pct_window{args.pre_event_window}_{args.post_event_window}min_cd{args.consecutive_drops}_vf{args.volume_factor}_rb{args.rebound_threshold*100}pct"
    
    # Detect if running in subdirectory
    if os.path.basename(os.getcwd()) == "hypothesis_testing":
        # Running in subdirectory
        if os.path.exists("results"):
            output_dir = f"results/{output_dir_name}"
        else:
            os.makedirs("results", exist_ok=True)
            output_dir = f"results/{output_dir_name}"
    else:
        # Running from project root or other location
        base_results_dir = "results/hypothesis_testing"
        os.makedirs(base_results_dir, exist_ok=True)
        output_dir = f"{base_results_dir}/{output_dir_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write parameters to JSON file
    parameters = {
        'start_date': start_date,
        'end_date': end_date,
        'days': args.days,
        'timeframe': args.timeframe,
        'drop_threshold': args.drop_threshold,
        'pre_event_window': args.pre_event_window,
        'post_event_window': args.post_event_window,
        'top_n': args.top_n,
        'consecutive_drops': args.consecutive_drops,
        'volume_factor': args.volume_factor,
        'rebound_threshold': args.rebound_threshold,
        'take_profit_pct': args.take_profit_pct,
        'stop_loss_pct': args.stop_loss_pct,
        'use_cache': not args.no_cache,
        'request_delay': args.request_delay,
        'runtime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{output_dir}/parameters.json", 'w') as f:
        json.dump(parameters, f, indent=4)
    
    # Display parameters
    param_panel = Panel(
        f"""[bold cyan]Hypothesis Testing Analysis Parameters[/bold cyan]
        
[green]Analysis Period:[/green] {start_date} to {end_date}
[green]Analysis Days:[/green] {args.days} days
[green]Timeframe:[/green] {args.timeframe}
[green]Drop Threshold:[/green] {args.drop_threshold * 100}%
[green]Consecutive Drops:[/green] {args.consecutive_drops} K-lines
[green]Volume Factor:[/green] {args.volume_factor}x
[green]Rebound Threshold:[/green] {args.rebound_threshold * 100}%
[green]Take Profit Percentage:[/green] {args.take_profit_pct * 100}%
[green]Stop Loss Percentage:[/green] {args.stop_loss_pct * 100}%
[green]Pre-Event Window:[/green] {args.pre_event_window} minutes
[green]Post-Event Window:[/green] {args.post_event_window} minutes
[green]Cache Status:[/green] {'Disabled' if args.no_cache else 'Enabled'}
[green]API Request Delay:[/green] {args.request_delay} seconds
[green]Output Directory:[/green] {output_dir}""",
        border_style="blue",
        expand=False
    )
    console.print(param_panel)
    
    # Initialize data fetcher
    with console.status("[bold green]Initializing data fetcher...", spinner="dots"):
        data_fetcher = DataFetcher(
            api_key=args.api_key,
            api_secret=args.api_secret,
            max_klines=args.max_klines,
            request_delay=args.request_delay,
            max_workers=args.max_workers,
            use_proxies=args.use_proxies
        )
    
    # Handle cache commands
    if args.show_cache_info:
        display_cache_info(data_fetcher)
        return
        
    if args.clean_cache or args.force_clean:
        clean_cache(data_fetcher, args.force_clean)
        return
    
    # Initialize hypothesis testing
    hypothesis_testing = HypothesisTesting(data_fetcher)
    
    # Run hypothesis testing with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running hypothesis testing analysis...", total=100)
        
        # Update progress to 10%
        progress.update(task, completed=10, description="[cyan]Getting ETH historical data...")
        
        # Create wrapper function for progress updates
        def progress_wrapper(percent_complete, description):
            try:
                # Ensure percent_complete is an integer or can be converted to an integer
                percent_int = int(float(percent_complete) * 100)
                progress.update(task, completed=percent_int, description=f"[cyan]{description}")
            except Exception as e:
                # If conversion fails, log error but don't interrupt program
                print(f"Error updating progress: {str(e)}")
        
        # Run hypothesis testing
        results_df, rebound_df, strategy_df = hypothesis_testing.run_hypothesis_testing(
            days=args.days,
            timeframe=args.timeframe,
            drop_threshold=args.drop_threshold,
            window_size=config.DETECTION_WINDOW_SIZE,
            top_n=args.top_n,
            pre_event_window=args.pre_event_window,
            post_event_window=args.post_event_window,
            end_date=args.end_date,
            use_cache=not args.no_cache,
            consecutive_drops=args.consecutive_drops,
            volume_factor=args.volume_factor,
            rebound_threshold=args.rebound_threshold,
            take_profit_pct=args.take_profit_pct,
            stop_loss_pct=args.stop_loss_pct,
            progress_callback=progress_wrapper,
            output_dir=output_dir
        )
    
    # Print summary of results
    if not results_df.empty:
        # Calculate significant proportions
        significant_results = results_df[results_df['p_value'] < 0.05]
        positive_sig = significant_results[significant_results['mean_diff'] > 0]
        negative_sig = significant_results[significant_results['mean_diff'] < 0]
        
        # Display effect size distribution
        effect_size_counts = {}
        if 'effect_interpretation' in results_df.columns:
            effect_size_counts = results_df['effect_interpretation'].value_counts().to_dict()
        
        effect_size_summary = ""
        if effect_size_counts:
            effect_size_summary = "\n\n[bold cyan]Effect Size Distribution:[/bold cyan]\n"
            for effect, count in effect_size_counts.items():
                percentage = (count / len(results_df)) * 100
                effect_size_summary += f"[green]{effect}:[/green] {count} ({percentage:.1f}%)\n"
        
        results_panel = Panel(
            f"""[bold cyan]Hypothesis Testing Results Summary[/bold cyan]
            
[green]Total Coins Analyzed:[/green] {len(results_df)}
[green]Statistically Significant Coins:[/green] {len(significant_results)} ({len(significant_results)/len(results_df)*100:.1f}%)
[green]Significant Positive Changes:[/green] {len(positive_sig)} ({len(positive_sig)/len(results_df)*100:.1f}%)
[green]Significant Negative Changes:[/green] {len(negative_sig)} ({len(negative_sig)/len(results_df)*100:.1f}%){effect_size_summary}""",
            border_style="green", 
            expand=False
        )
        console.print(results_panel)
        
        # Create table for significant positive changes
        if not positive_sig.empty:
            positive_table = Table(title="Coins with Significant Positive Changes (Top 5)")
            positive_table.add_column("Coin", style="cyan")
            positive_table.add_column("Mean Diff", style="green")
            positive_table.add_column("p-value", style="blue")
            positive_table.add_column("t-statistic", style="yellow")
            
            # Add effect size columns if available
            has_effect_size = 'effect_size' in positive_sig.columns
            has_effect_interp = 'effect_interpretation' in positive_sig.columns
            
            if has_effect_size:
                positive_table.add_column("Effect Size", style="magenta")
            if has_effect_interp:
                positive_table.add_column("Effect Level", style="magenta")
            
            # Sort by mean difference (descending)
            top_5_positive = positive_sig.sort_values('mean_diff', ascending=False).head(5)
            for _, row in top_5_positive.iterrows():
                # Check which t-statistic column name to use
                t_stat_value = row.get('t_stat', row.get('t_statistic', 0.0))
                
                columns = [
                    row['symbol'],
                    f"{row['mean_diff']*100:.2f}%",
                    f"{row['p_value']:.4f}",
                    f"{t_stat_value:.2f}"
                ]
                
                if has_effect_size:
                    columns.append(f"{row['effect_size']:.2f}")
                if has_effect_interp:
                    columns.append(row['effect_interpretation'])
                
                positive_table.add_row(*columns)
            
            console.print(positive_table)
        
        # Create table for significant negative changes
        if not negative_sig.empty:
            negative_table = Table(title="Coins with Significant Negative Changes (Top 5)")
            negative_table.add_column("Coin", style="cyan")
            negative_table.add_column("Mean Diff", style="red")
            negative_table.add_column("p-value", style="blue")
            negative_table.add_column("t-statistic", style="yellow")
            
            # Add effect size columns if available
            if has_effect_size:
                negative_table.add_column("Effect Size", style="magenta")
            if has_effect_interp:
                negative_table.add_column("Effect Level", style="magenta")
            
            # Sort by mean difference (ascending)
            top_5_negative = negative_sig.sort_values('mean_diff', ascending=True).head(5)
            for _, row in top_5_negative.iterrows():
                # Check which t-statistic column name to use
                t_stat_value = row.get('t_stat', row.get('t_statistic', 0.0))
                
                columns = [
                    row['symbol'],
                    f"{row['mean_diff']*100:.2f}%",
                    f"{row['p_value']:.4f}",
                    f"{t_stat_value:.2f}"
                ]
                
                if has_effect_size:
                    columns.append(f"{row['effect_size']:.2f}")
                if has_effect_interp:
                    columns.append(row['effect_interpretation'])
                
                negative_table.add_row(*columns)
            
            console.print(negative_table)
        
        console.print(f"\n[green]Detailed results have been saved to:[/green] {output_dir}/")
    else:
        console.print("[yellow]No valid hypothesis testing results found[/yellow]")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("[bold red]Program interrupted by user[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Execution error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1) 