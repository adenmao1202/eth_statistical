#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH Statistical Analysis Framework
Main entry point for all three research components:
1. Event Study
2. Clustering Analysis
3. Hypothesis Testing
"""

import os
import argparse
import sys
import json
import shutil
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# 初始化rich控制台
console = Console()

def print_header():
    """打印歡迎頭部信息"""
    header_panel = Panel(
        """[bold cyan]ETH 統計分析框架[/bold cyan]

[green]1.[/green] 事件研究: ETH價格下跌對其他加密貨幣的影響
[green]2.[/green] 聚類分析: 基於XGBoost的加密貨幣聚類
[green]3.[/green] 假設檢驗: 收益分佈分析""",
        border_style="blue",
        expand=False
    )
    console.print(header_panel)

def print_usage():
    """打印使用說明"""
    usage_md = """
# 使用說明

## 基本用法
```bash
python3 main.py --run [module_name] [options]
```

## 可用模塊
- `event_study`: 事件研究分析
- `clustering`: 加密貨幣聚類分析
- `hypothesis_testing`: 假設檢驗分析
- `cache_info`: 顯示緩存信息
- `clean_cache`: 清理過期緩存

## 常用選項
- `--timeframe`: 時間框架 (1m, 5m, 15m, 1h, 4h)
- `--days`: 分析天數
- `--drop_threshold`: ETH下跌閾值
- `--no_cache`: 禁用緩存
- `--request_delay`: API請求延遲

## 範例
```bash
# 運行事件研究
python3 main.py --run event_study --timeframe 1m --days 30 --drop_threshold -0.01

# 運行聚類分析
python3 main.py --run clustering --timeframe 1h --days 90 --clusters 5

# 運行假設檢驗
python3 main.py --run hypothesis_testing --timeframe 1m --days 60 --pre_event_window 30 --post_event_window 30
```
"""
    console.print(Markdown(usage_md))

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='ETH 統計分析框架')
    
    # 主要命令選項
    parser.add_argument('--run', type=str, required=True, 
                       choices=['event_study', 'clustering', 'hypothesis_testing', 'cache_info', 'clean_cache'],
                       help='要運行的分析模塊')
    
    # 通用數據獲取參數
    parser.add_argument('--api_key', type=str, help='Binance API密鑰')
    parser.add_argument('--api_secret', type=str, help='Binance API密碼')
    parser.add_argument('--request_delay', type=float, help='API請求延遲（秒）')
    parser.add_argument('--no_cache', action='store_true', help='禁用緩存')
    parser.add_argument('--force_clean', action='store_true', help='強制清理所有緩存文件')
    
    # 通用分析參數
    parser.add_argument('--days', type=int, help='分析天數')
    parser.add_argument('--timeframe', type=str, help='分析時間框架 (1m, 5m, 15m, 1h, 4h)')
    parser.add_argument('--top_n', type=int, help='分析排名前N的幣種')
    parser.add_argument('--start_date', type=str, help='開始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='結束日期 (YYYY-MM-DD)')
    
    # 事件研究和假設檢驗特定參數
    parser.add_argument('--drop_threshold', type=float, help='ETH下跌閾值 (負數)')
    parser.add_argument('--pre_event_window', type=int, help='事件前窗口（分鐘）')
    parser.add_argument('--post_event_window', type=int, help='事件後窗口（分鐘）')
    parser.add_argument('--window_size', type=int, help='檢測窗口大小')
    
    # 聚類分析特定參數
    parser.add_argument('--clusters', type=int, help='聚類數量')
    parser.add_argument('--corr_threshold', type=float, help='相關性閾值')
    
    return parser.parse_args()

def execute_module(module_name, args):
    """執行指定的分析模塊"""
    try:
        if module_name == 'event_study':
            # 使用sys.path添加路徑而不是動態導入包
            event_study_path = os.path.join(os.path.dirname(__file__), 'event_study')
            sys.path.insert(0, event_study_path)
            
            # 修改命令行參數
            original_argv = sys.argv
            sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[2:]]
            
            # 導入並執行
            import importlib.util
            spec = importlib.util.spec_from_file_location("event_study_main", os.path.join(event_study_path, "main.py"))
            event_study_main = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(event_study_main)
            event_study_main.main()
            
            # 恢復原始參數
            sys.argv = original_argv
            sys.path.pop(0)
            
        elif module_name == 'clustering':
            # 使用sys.path添加路徑而不是動態導入包
            clustering_path = os.path.join(os.path.dirname(__file__), 'clustering')
            sys.path.insert(0, clustering_path)
            
            # 修改命令行參數
            original_argv = sys.argv
            sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[2:]]
            
            # 導入並執行
            import importlib.util
            spec = importlib.util.spec_from_file_location("clustering_main", os.path.join(clustering_path, "main.py"))
            clustering_main = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(clustering_main)
            clustering_main.main()
            
            # 恢復原始參數
            sys.argv = original_argv
            sys.path.pop(0)
            
        elif module_name == 'hypothesis_testing':
            # 使用sys.path添加路徑而不是動態導入包
            hypothesis_path = os.path.join(os.path.dirname(__file__), 'hypothesis_testing')
            sys.path.insert(0, hypothesis_path)
            
            # 修改命令行參數
            original_argv = sys.argv
            sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[2:]]
            
            # 導入並執行
            import importlib.util
            spec = importlib.util.spec_from_file_location("hypothesis_main", os.path.join(hypothesis_path, "main.py"))
            hypothesis_main = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hypothesis_main)
            hypothesis_main.main()
            
            # 恢復原始參數
            sys.argv = original_argv
            sys.path.pop(0)
            
        elif module_name in ['cache_info', 'clean_cache']:
            # 導入數據獲取器
            from data_fetcher import DataFetcher
            
            data_fetcher = DataFetcher(
                api_key=args.api_key,
                api_secret=args.api_secret,
                request_delay=args.request_delay
            )
            
            if module_name == 'cache_info':
                display_cache_info(data_fetcher)
            else:
                clean_cache(data_fetcher, args.force_clean)
                
    except KeyboardInterrupt:
        console.print("[bold red]程序被用户中断[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]執行時錯誤: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

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

def clean_up_project():
    """清理項目中不需要的文件和目錄"""
    # 刪除根目錄下的舊文件和目錄
    old_dirs = ['eth_dist', 'old', 'coins', 'corr']
    
    for dir_name in old_dirs:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                console.print(f"[green]已刪除舊目錄: {dir_name}[/green]")
            except Exception as e:
                console.print(f"[red]無法刪除目錄 {dir_name}: {str(e)}[/red]")
    
    # 移動根目錄下的historical_data到main目錄
    if os.path.exists('historical_data') and not os.path.exists('main/historical_data'):
        try:
            shutil.move('historical_data', 'main/historical_data')
            console.print("[green]已移動historical_data目錄到main目錄下[/green]")
        except Exception as e:
            console.print(f"[red]無法移動historical_data目錄: {str(e)}[/red]")
    
    # 更新根目錄的README.md
    if os.path.exists('README.md') and os.path.exists('main/README.md'):
        try:
            shutil.copy2('main/README.md', 'README.md')
            console.print("[green]已更新根目錄的README.md[/green]")
        except Exception as e:
            console.print(f"[red]無法更新README.md: {str(e)}[/red]")

def main():
    """主函數入口點"""
    print_header()
    
    # 如果沒有參數，顯示使用說明
    if len(sys.argv) == 1:
        print_usage()
        sys.exit(0)
    
    args = parse_arguments()
    
    # 執行指定的模塊
    execute_module(args.run, args)

if __name__ == '__main__':
    main() 