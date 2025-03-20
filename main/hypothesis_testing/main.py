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

# 確保父目錄在模塊搜索路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 接著導入所需模塊
try:
    import config
    from data_fetcher import DataFetcher
    from hypothesis_testing import HypothesisTesting
except ImportError:
    # 如果在子目錄執行，嘗試使用相對導入
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
    """Run the hypothesis testing analysis"""
    args = parse_arguments()
    
    # Set up dates
    if args.start_date:
        start_date = args.start_date
    else:
        # Calculate start date based on days
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    
    # 創建輸出目錄，考慮到從不同位置執行的情況
    output_dir_name = f"{args.timeframe}_{args.days}days_drop{abs(args.drop_threshold*100)}pct_window{args.pre_event_window}_{args.post_event_window}min"
    
    # 檢測是否在子目錄執行
    if os.path.basename(os.getcwd()) == "hypothesis_testing":
        # 在子目錄執行
        if os.path.exists("results"):
            output_dir = f"results/{output_dir_name}"
        else:
            os.makedirs("results", exist_ok=True)
            output_dir = f"results/{output_dir_name}"
    else:
        # 從項目根目錄或其他位置執行
        base_results_dir = "results/hypothesis_testing"
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
        'pre_event_window': args.pre_event_window,
        'post_event_window': args.post_event_window,
        'top_n': args.top_n,
        'use_cache': not args.no_cache,
        'request_delay': args.request_delay,
        'runtime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{output_dir}/parameters.json", 'w') as f:
        json.dump(parameters, f, indent=4)
    
    # Display parameters
    param_panel = Panel(
        f"""[bold cyan]假设检验分析参数[/bold cyan]
        
[green]分析周期:[/green] {start_date} 至 {end_date}
[green]分析天数:[/green] {args.days} 天
[green]时间框架:[/green] {args.timeframe}
[green]下跌阈值:[/green] {args.drop_threshold * 100}%
[green]事件前窗口:[/green] {args.pre_event_window} 分钟
[green]事件后窗口:[/green] {args.post_event_window} 分钟
[green]缓存状态:[/green] {'禁用' if args.no_cache else '启用'}
[green]API请求延迟:[/green] {args.request_delay} 秒
[green]输出目录:[/green] {output_dir}""",
        border_style="blue",
        expand=False
    )
    console.print(param_panel)
    
    # Initialize data fetcher
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
        task = progress.add_task("[cyan]正在运行假设检验分析...", total=100)
        
        # Update progress to 10%
        progress.update(task, completed=10, description="[cyan]获取ETH历史数据...")
        
        # Create wrapper function for progress updates
        def progress_wrapper(percent_complete, description):
            try:
                # 确保percent_complete是整数或可以转换为整数的值
                percent_int = int(float(percent_complete) * 100)
                progress.update(task, completed=percent_int, description=f"[cyan]{description}")
            except Exception as e:
                # 如果转换失败，记录错误但不中断程序
                print(f"Error updating progress: {str(e)}")
        
        # Run hypothesis testing
        results_df = hypothesis_testing.run_hypothesis_testing(
            days=args.days,
            timeframe=args.timeframe,
            drop_threshold=args.drop_threshold,
            pre_event_window=args.pre_event_window,
            post_event_window=args.post_event_window,
            end_date=args.end_date,
            use_cache=not args.no_cache,
            progress_callback=progress_wrapper,
            output_dir=output_dir
        )
    
    # Print summary of results
    if not results_df.empty:
        # 計算顯著比例
        significant_results = results_df[results_df['p_value'] < 0.05]
        positive_sig = significant_results[significant_results['mean_diff'] > 0]
        negative_sig = significant_results[significant_results['mean_diff'] < 0]
        
        # 顯示效應大小分佈
        effect_size_counts = {}
        if 'effect_interpretation' in results_df.columns:
            effect_size_counts = results_df['effect_interpretation'].value_counts().to_dict()
        
        effect_size_summary = ""
        if effect_size_counts:
            effect_size_summary = "\n\n[bold cyan]效应大小分布:[/bold cyan]\n"
            for effect, count in effect_size_counts.items():
                percentage = (count / len(results_df)) * 100
                effect_size_summary += f"[green]{effect}:[/green] {count} 个 ({percentage:.1f}%)\n"
        
        results_panel = Panel(
            f"""[bold cyan]假设检验结果摘要[/bold cyan]
            
[green]分析的币种总数:[/green] {len(results_df)}
[green]统计显著的币种数:[/green] {len(significant_results)} ({len(significant_results)/len(results_df)*100:.1f}%)
[green]显著正向变化:[/green] {len(positive_sig)} ({len(positive_sig)/len(results_df)*100:.1f}%)
[green]显著负向变化:[/green] {len(negative_sig)} ({len(negative_sig)/len(results_df)*100:.1f}%){effect_size_summary}""",
            border_style="green", 
            expand=False
        )
        console.print(results_panel)
        
        # Create table for significant positive changes
        if not positive_sig.empty:
            positive_table = Table(title="显著正向变化的币种(前5名)")
            positive_table.add_column("币种", style="cyan")
            positive_table.add_column("均值差异", style="green")
            positive_table.add_column("p值", style="blue")
            positive_table.add_column("t统计量", style="yellow")
            
            # Add effect size columns if available
            has_effect_size = 'effect_size' in positive_sig.columns
            has_effect_interp = 'effect_interpretation' in positive_sig.columns
            
            if has_effect_size:
                positive_table.add_column("效应大小", style="magenta")
            if has_effect_interp:
                positive_table.add_column("效应程度", style="magenta")
            
            # Sort by mean difference (descending)
            top_5_positive = positive_sig.sort_values('mean_diff', ascending=False).head(5)
            for _, row in top_5_positive.iterrows():
                # 檢查使用哪個名稱的t統計量列
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
            negative_table = Table(title="显著负向变化的币种(前5名)")
            negative_table.add_column("币种", style="cyan")
            negative_table.add_column("均值差异", style="red")
            negative_table.add_column("p值", style="blue")
            negative_table.add_column("t统计量", style="yellow")
            
            # Add effect size columns if available
            if has_effect_size:
                negative_table.add_column("效应大小", style="magenta")
            if has_effect_interp:
                negative_table.add_column("效应程度", style="magenta")
            
            # Sort by mean difference (ascending)
            top_5_negative = negative_sig.sort_values('mean_diff', ascending=True).head(5)
            for _, row in top_5_negative.iterrows():
                # 檢查使用哪個名稱的t統計量列
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
        
        console.print(f"\n[green]详细结果已保存到:[/green] {output_dir}/")
    else:
        console.print("[yellow]未找到有效的假设检验結果[/yellow]")

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