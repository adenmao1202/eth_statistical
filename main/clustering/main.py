#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering Analysis Main Module
Entry point for the cryptocurrency clustering analysis
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
import time

# 確保父目錄在模塊搜索路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 設置默認緩存目錄
CACHE_DIR = os.path.join(parent_dir, 'data', 'cache')

# 接著導入所需模塊
try:
    import config
    from data_fetcher import DataFetcher
    from clustering import ClusteringAnalyzer
except ImportError:
    # 如果在子目錄執行，嘗試使用相對導入
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(parent_dir, ".."))
    import config
    from data_fetcher import DataFetcher
    from clustering import ClusteringAnalyzer

# 如果config中沒有定義CACHE_EXPIRY_DAYS，這裡設置一個默認值
if not hasattr(config, 'CACHE_EXPIRY_DAYS'):
    config.CACHE_EXPIRY_DAYS = 30

# Initialize Rich console
console = Console()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Clustering Analysis Using XGBoost')
    
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
    
    # Clustering parameters
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days to analyze (default: 30)')
    parser.add_argument('--timeframe', type=str, default='1h', choices=config.TIMEFRAMES.keys(),
                       help=f'Timeframe for analysis (default: 1h)')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--n_clusters', type=int, default=config.NUM_CLUSTERS, 
                       help=f'Number of clusters to form (default: {config.NUM_CLUSTERS})')
    parser.add_argument('--top_n', type=int, default=config.DEFAULT_TOP_N, 
                       help=f'Number of top cryptocurrencies to analyze (default: {config.DEFAULT_TOP_N})')
    
    # XGBoost parameters
    parser.add_argument('--max_depth', type=int, default=config.XGBOOST_MAX_DEPTH, 
                       help=f'XGBoost max depth (default: {config.XGBOOST_MAX_DEPTH})')
    parser.add_argument('--learning_rate', type=float, default=config.XGBOOST_LEARNING_RATE, 
                       help=f'XGBoost learning rate (default: {config.XGBOOST_LEARNING_RATE})')
    parser.add_argument('--n_estimators', type=int, default=config.XGBOOST_N_ESTIMATORS, 
                       help=f'XGBoost number of estimators (default: {config.XGBOOST_N_ESTIMATORS})')
    
    # Cache management parameters
    parser.add_argument('--show_cache_info', action='store_true', help='Display cache information')
    parser.add_argument('--clean_cache', action='store_true', help='Clean expired cache files')
    parser.add_argument('--force_clean', action='store_true', help='Force clean all cache files')
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR, help='Directory for data cache')
    parser.add_argument('--cache_days', type=int, default=None, help='Cache expiry days (default: config.CACHE_EXPIRY_DAYS)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    
    return parser.parse_args()

def display_cache_info(cache_dir, cache_days=None):
    """顯示緩存統計信息"""
    try:
        # 獲取緩存目錄中的所有文件
        cache_files = []
        total_size = 0
        expired_files = 0
        
        # 檢查緩存目錄是否存在
        if not os.path.exists(cache_dir):
            console.print(f"[bold red]緩存目錄不存在: {cache_dir}[/bold red]")
            return
            
        now = datetime.now()
        # 使用參數傳遞的過期天數，如果沒有則使用配置值
        cache_expiry_days = cache_days if cache_days is not None else config.CACHE_EXPIRY_DAYS
        
        # 獲取所有緩存文件
        for file in os.listdir(cache_dir):
            if file.endswith('.json'):
                file_path = os.path.join(cache_dir, file)
                file_stat = os.stat(file_path)
                file_modified = datetime.fromtimestamp(file_stat.st_mtime)
                file_size = file_stat.st_size
                
                # 檢查文件是否過期
                age_in_days = (now - file_modified).days
                is_expired = age_in_days > cache_expiry_days
                
                cache_files.append({
                    'path': file_path,
                    'size': file_size,
                    'modified': file_modified,
                    'expired': is_expired
                })
                
                total_size += file_size
                if is_expired:
                    expired_files += 1
                    
        # 對緩存文件進行排序
        cache_files.sort(key=lambda x: x['modified'])
        
        # 準備緩存統計信息
        cache_stats = {
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),  # 轉換為MB
            'expired_files': expired_files,
            'cache_expiry_days': cache_expiry_days,
            'cache_dir': cache_dir,
            'newest_cache': cache_files[-1] if cache_files else None,
            'oldest_cache': cache_files[0] if cache_files else None
        }
        
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
            
    except Exception as e:
        console.print(f"[bold red]獲取緩存信息時出錯: {str(e)}[/bold red]")

def clean_cache(cache_dir, cache_days=None, force=False):
    """清理緩存"""
    try:
        # 檢查緩存目錄是否存在
        if not os.path.exists(cache_dir):
            console.print(f"[bold red]緩存目錄不存在: {cache_dir}[/bold red]")
            return {'error': f"緩存目錄不存在: {cache_dir}"}
            
        # 初始化計數器
        removed_files = 0
        reclaimed_space = 0
        now = datetime.now()
        
        # 使用參數傳遞的過期天數，如果沒有則使用配置值
        expiry_days = cache_days if cache_days is not None else config.CACHE_EXPIRY_DAYS
        
        # 遍歷緩存文件
        for file in os.listdir(cache_dir):
            if file.endswith('.json'):
                file_path = os.path.join(cache_dir, file)
                file_stat = os.stat(file_path)
                file_modified = datetime.fromtimestamp(file_stat.st_mtime)
                
                # 決定是否刪除文件
                should_delete = False
                if force:
                    # 強制模式刪除所有緩存文件
                    should_delete = True
                else:
                    # 只刪除過期文件
                    age_in_days = (now - file_modified).days
                    if age_in_days > expiry_days:
                        should_delete = True
                        
                # 刪除文件
                if should_delete:
                    file_size = file_stat.st_size
                    try:
                        os.remove(file_path)
                        removed_files += 1
                        reclaimed_space += file_size
                        print(f"已刪除: {file}")
                    except Exception as e:
                        print(f"刪除文件 {file} 時出錯: {str(e)}")
        
        # 準備結果
        result = {
            'removed_files': removed_files,
            'reclaimed_space_mb': reclaimed_space / (1024 * 1024)  # 轉換為MB
        }
        
        if force:
            console.print(f"[bold green]強制清理完成！[/bold green]")
        else:
            console.print(f"[bold green]過期緩存清理完成！[/bold green]")
            
        console.print(f"[green]移除文件數:[/green] {result['removed_files']}")
        console.print(f"[green]釋放空間:[/green] {result['reclaimed_space_mb']:.2f} MB")
        
        return result
        
    except Exception as e:
        console.print(f"[bold red]清理緩存時出錯: {str(e)}[/bold red]")
        return {'error': str(e)}

def main():
    """
    入口函數 - 使用命令行參數運行加密貨幣聚類分析
    """
    # 解析命令行參數
    args = parse_arguments()
    
    # 處理緩存命令
    if args.show_cache_info:
        display_cache_info(CACHE_DIR, args.cache_days)
        return

    if args.clean_cache:
        clean_cache(CACHE_DIR, args.cache_days, args.force_clean)
        return
    
    # 創建輸出目錄
    if not args.output_dir:
        # 創建基於參數的默認輸出目錄名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"clustering_results_{args.timeframe}_{args.n_clusters}clusters_{args.days}days_{timestamp}"
    else:
        output_dir = args.output_dir
        
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"結果將保存到目錄: {output_dir}")
    
    # 保存分析參數到輸出目錄
    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
            
    # 計算分析的開始日期和結束日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    print(f"分析時間範圍: {start_date_str} 到 {end_date_str}")
    print(f"時間框架: {args.timeframe}")
    print(f"聚類數量: {args.n_clusters}")
    print(f"API請求延遲: {args.request_delay}秒")
    
    # 初始化數據獲取器
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    # 修改導入路徑，直接從main目錄導入
    from data_fetcher import DataFetcher
    data_fetcher = DataFetcher(
        api_key=args.api_key,
        api_secret=args.api_secret,
        max_klines=args.max_klines,
        request_delay=args.request_delay,
        max_workers=args.max_workers,
        use_proxies=args.use_proxies
    )
    
    # 獲取交易量最高的交易對
    print("獲取交易量最高的交易對...")
    top_symbols = data_fetcher.get_top_volume_symbols(n=args.top_n)
    print(f"選定的交易對: {', '.join(top_symbols)}")
    
    # 獲取所有交易對的數據
    print(f"獲取{len(top_symbols)}個交易對的數據...")
    coin_data = data_fetcher.get_all_data(
        timeframe=args.timeframe,
        days=args.days,
        top_n=args.top_n,
        end_date=end_date_str,
        use_cache=not args.no_cache
    )
    
    # 初始化聚類器
    # 從當前目錄導入
    from clustering.clustering import ClusteringAnalyzer
    clusterer = ClusteringAnalyzer(data_fetcher=data_fetcher)
    
    # 運行聚類分析
    print("開始聚類分析...")
    labels, feature_importance = clusterer.run_clustering_analysis(
        symbols=top_symbols,
        data_dict=coin_data,
        n_clusters=args.n_clusters,
        output_dir=output_dir
    )
    
    if labels is None:
        print("聚類分析失敗。請檢查錯誤信息並調整參數後重試。")
        return
    
    # 顯示聚類結果統計
    print("\n聚類結果統計:")
    cluster_counts = labels.value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"集群 {cluster}: {count} 個幣種 ({count/len(labels)*100:.1f}%)")
    
    print(f"\n聚類分析完成。結果已保存到 {output_dir}")
    
if __name__ == "__main__":
    main() 