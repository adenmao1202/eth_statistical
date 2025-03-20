#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event Study Main Module
Entry point for the cryptocurrency event study analysis
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

# 確保父目錄在模塊搜索路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 接著導入所需模塊
try:
    import config
    from data_fetcher import DataFetcher
    from event_study import EventStudy
except ImportError:
    # 如果在子目錄執行，嘗試使用相對導入
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(parent_dir, ".."))
    import config
    from data_fetcher import DataFetcher
    from event_study import EventStudy

# Initialize Rich console
console = Console()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Event Study Analysis')
    
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
    
    # Event study parameters
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--timeframe', type=str, default='1m', choices=config.TIMEFRAMES.keys(),
                       help=f'Timeframe for analysis (default: 1m)')
    parser.add_argument('--drop_threshold', type=float, default=config.MIN_DROP_PCT,
                       help=f'Threshold for identifying ETH price drops (default: {config.MIN_DROP_PCT})')
    parser.add_argument('--event_window', type=int, default=config.EVENT_WINDOW_MINUTES,
                       help=f'Event window in minutes (default: {config.EVENT_WINDOW_MINUTES})')
    parser.add_argument('--window_size', type=int, default=config.DETECTION_WINDOW_SIZE,
                       help=f'Window size for price drop detection (default: {config.DETECTION_WINDOW_SIZE})')
    parser.add_argument('--top_n', type=int, default=config.DEFAULT_TOP_N,
                       help=f'Number of top coins to display (default: {config.DEFAULT_TOP_N})')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--show_cache_info', action='store_true', help='Display cache information')
    parser.add_argument('--clean_cache', action='store_true', help='Clean expired cache files')
    parser.add_argument('--force_clean', action='store_true', help='Force clean all cache files')
    
    return parser.parse_args()

def display_cache_info(data_fetcher):
    """顯示緩存統計信息"""
    cache_stats = data_fetcher.get_cache_stats()
    
    console.print(Panel(
        f"""[bold cyan]緩存統計信息[/bold cyan]
        
[green]緩存文件總數:[/green] {cache_stats.get('total_files', 'N/A')} 個
[green]緩存總大小:[/green] {cache_stats.get('total_size_mb', 0):.2f} MB
[green]過期文件數量:[/green] {cache_stats.get('expired_files', 'N/A')} 個
[green]緩存有效期:[/green] {cache_stats.get('cache_expiry_days', 'N/A')} 天
[green]緩存目錄:[/green] {cache_stats.get('cache_dir', 'N/A')}
""",
        border_style="blue",
        expand=False
    ))
    
    # 如果有最新和最舊的緩存，顯示它們的信息
    newest = cache_stats.get('newest_cache')
    oldest = cache_stats.get('oldest_cache')
    
    if newest:
        console.print(f"[green]最新緩存文件:[/green] {os.path.basename(newest['path'])}")
        console.print(f"[green]修改時間:[/green] {newest['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"[green]文件大小:[/green] {newest['size'] / 1024:.2f} KB")
    
    if oldest:
        console.print(f"\n[yellow]最舊緩存文件:[/yellow] {os.path.basename(oldest['path'])}")
        console.print(f"[yellow]修改時間:[/yellow] {oldest['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"[yellow]文件大小:[/yellow] {oldest['size'] / 1024:.2f} KB")

def clean_cache(data_fetcher, force_clean=False):
    """清理緩存"""
    result = data_fetcher.clean_expired_cache(force_clean)
    
    if 'error' in result:
        console.print(f"[bold red]緩存清理失敗: {result['error']}[/bold red]")
    else:
        if force_clean:
            console.print(f"[bold green]強制清理完成！[/bold green]")
        else:
            console.print(f"[bold green]過期緩存清理完成！[/bold green]")
        
        console.print(f"[green]移除文件數:[/green] {result['removed_files']}")
        console.print(f"[green]釋放空間:[/green] {result['reclaimed_space_mb']:.2f} MB")

def main():
    """Run the event study analysis"""
    args = parse_arguments()
    
    # Set up dates
    if args.start_date:
        start_date = args.start_date
    else:
        # Calculate start date based on days
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    
    # 創建輸出目錄，考慮到從不同位置執行的情況
    output_dir_name = f"{args.timeframe}_{args.days}days_drop{abs(args.drop_threshold*100)}pct_window{args.event_window}min"
    
    # 檢測是否在子目錄執行
    if os.path.basename(os.getcwd()) == "event_study":
        # 在子目錄執行
        if os.path.exists("results"):
            output_dir = f"results/{output_dir_name}"
        else:
            os.makedirs("results", exist_ok=True)
            output_dir = f"results/{output_dir_name}"
    else:
        # 從項目根目錄或其他位置執行
        base_results_dir = "results/event_study"
        os.makedirs(base_results_dir, exist_ok=True)
        output_dir = f"{base_results_dir}/{output_dir_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write parameters to file
    parameters = {
        'start_date': start_date,
        'end_date': end_date,
        'days': args.days,
        'timeframe': args.timeframe,
        'drop_threshold': args.drop_threshold,
        'event_window': args.event_window,
        'window_size': args.window_size,
        'top_n': args.top_n,
        'use_cache': not args.no_cache,
        'request_delay': args.request_delay,
        'runtime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{output_dir}/parameters.json", 'w') as f:
        json.dump(parameters, f, indent=4)
    
    # Display parameters
    param_panel = Panel(
        f"""[bold cyan]事件研究分析参数[/bold cyan]
        
[green]分析周期:[/green] {start_date} 至 {end_date}
[green]分析天数:[/green] {args.days} 天
[green]时间框架:[/green] {args.timeframe}
[green]下跌阈值:[/green] {args.drop_threshold * 100}%
[green]事件窗口:[/green] {args.event_window} 分钟
[green]检测窗口:[/green] {args.window_size} 个数据点
[green]缓存状态:[/green] {'禁用' if args.no_cache else '启用'}
[green]API请求延迟:[/green] {args.request_delay} 秒
[green]输出目录:[/green] {output_dir}""",
        border_style="blue",
        expand=False
    )
    console.print(param_panel)
    
    # Initialize data fetcher with progress indicator
    with console.status("[bold green]初始化数据获取器...", spinner="dots"):
        data_fetcher = DataFetcher(
            api_key=args.api_key,
            api_secret=args.api_secret,
            max_klines=args.max_klines,
            request_delay=args.request_delay,
            max_workers=args.max_workers,
            use_proxies=args.use_proxies
        )
    
    # 處理緩存命令
    if args.show_cache_info:
        display_cache_info(data_fetcher)
        return
        
    if args.clean_cache or args.force_clean:
        clean_cache(data_fetcher, args.force_clean)
        return
    
    # Initialize event study analyzer
    event_study = EventStudy(data_fetcher)
    event_study.event_window_minutes = args.event_window
    
    # Run event study analysis with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]正在运行事件研究分析...", total=100)
        
        # Update progress to 10%
        progress.update(task, completed=10, description="[cyan]获取ETH历史数据...")
        
        # 创建一个包装函数来确保progress_callback的第一个参数(进度值)是整数
        def progress_wrapper(percent_complete, description):
            try:
                # 确保percent_complete是整数或可以转换为整数的值
                percent_int = int(float(percent_complete) * 100)
                progress.update(task, completed=percent_int, description=f"[cyan]{description}")
            except Exception as e:
                # 如果转换失败，记录错误但不中断程序
                print(f"Error updating progress: {str(e)}")
        
        # Run event study analysis
        results_df = event_study.run_event_study(
            days=args.days,
            timeframe=args.timeframe,
            drop_threshold=args.drop_threshold,
            window_size=args.window_size,
            top_n=args.top_n,
            end_date=args.end_date,
            use_cache=not args.no_cache,
            progress_callback=progress_wrapper,
            output_dir=output_dir
        )
    
    # Print summary of results
    if not results_df.empty:
        significant_results = results_df[results_df['significant']]
        
        results_panel = Panel(
            f"""[bold cyan]事件研究结果摘要[/bold cyan]
            
[green]分析的币种总数:[/green] {len(results_df)}
[green]具有统计显著性异常收益的币种:[/green] {len(significant_results)}
[green]当前样本量:[/green] {results_df['sample_size'].iloc[0] if not results_df.empty else 0}个事件""",
            border_style="green", 
            expand=False
        )
        console.print(results_panel)
        
        if not significant_results.empty:
            # Positive abnormal returns
            positive_sig = significant_results[significant_results['mean_abnormal_return'] > 0]
            negative_sig = significant_results[significant_results['mean_abnormal_return'] < 0]
            
            console.print(f"[green]具有显著正异常收益的币种:[/green] {len(positive_sig)}")
            console.print(f"[red]具有显著负异常收益的币种:[/red] {len(negative_sig)}")
            
            # Create table for top performers
            top_table = Table(title="前5名表现最佳币种")
            top_table.add_column("币种", style="cyan")
            top_table.add_column("异常收益", style="green")
            top_table.add_column("p值", style="blue")
            top_table.add_column("t统计量", style="yellow")
            top_table.add_column("样本量", style="magenta")
            
            # Sort by abnormal return (descending)
            top_5 = significant_results.sort_values('mean_abnormal_return', ascending=False).head(5)
            for _, row in top_5.iterrows():
                top_table.add_row(
                    row['symbol'],
                    f"{row['mean_abnormal_return']*100:.2f}%",
                    f"{row['p_value']:.4f}",
                    f"{row['t_statistic']:.2f}",
                    str(row['sample_size'])
                )
            console.print(top_table)
            
            # Create table for bottom performers
            bottom_table = Table(title="前5名表现最差币种")
            bottom_table.add_column("币种", style="cyan")
            bottom_table.add_column("异常收益", style="red")
            bottom_table.add_column("p值", style="blue")
            bottom_table.add_column("t统计量", style="yellow")
            bottom_table.add_column("样本量", style="magenta")
            
            # Sort by abnormal return (ascending)
            bottom_5 = significant_results.sort_values('mean_abnormal_return', ascending=True).head(5)
            for _, row in bottom_5.iterrows():
                bottom_table.add_row(
                    row['symbol'],
                    f"{row['mean_abnormal_return']*100:.2f}%",
                    f"{row['p_value']:.4f}",
                    f"{row['t_statistic']:.2f}",
                    str(row['sample_size'])
                )
            console.print(bottom_table)
            
            console.print(f"\n[green]详细结果已保存到:[/green] {output_dir}/")
            console.print(f"[green]交互式图表:[/green] {output_dir}/eth_drop_impact_interactive.html")
    else:
        console.print("[yellow]未找到有效的事件研究結果[/yellow]")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("[bold red]程序被用户中断[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]执行时错误: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1) 