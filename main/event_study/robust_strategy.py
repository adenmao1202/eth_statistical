#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
穩健交易策略模塊
基於跨時期驗證結果設計和回測交易策略
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_fetcher import DataFetcher


class RobustStrategy:
    """
    穩健交易策略
    基於跨時期驗證結果設計和回測交易策略
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        初始化交易策略類
        
        Args:
            data_fetcher: 數據獲取器實例
        """
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
        self.output_dir = os.path.join(os.path.dirname(__file__), 'results/strategy')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_validation_results(self, results_file: str) -> pd.DataFrame:
        """
        載入跨時期驗證的結果
        
        Args:
            results_file: 驗證結果文件路徑
            
        Returns:
            pd.DataFrame: 驗證結果DataFrame
        """
        try:
            results = pd.read_csv(results_file)
            self.logger.info(f"Loaded validation results from {results_file}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to load validation results: {e}")
            return pd.DataFrame()
    
    def select_robust_coins(self, 
                          validation_results: pd.DataFrame, 
                          min_stability: float = 0.6,
                          min_performance: float = 0.01,
                          max_std: float = 0.05,
                          top_n: int = 10) -> List[str]:
        """
        選擇穩健的幣種用於交易策略
        
        Args:
            validation_results: 驗證結果DataFrame
            min_stability: 最小穩定性閾值
            min_performance: 最小平均表現閾值
            max_std: 最大標準差閾值
            top_n: 選擇前N個符合條件的幣種
            
        Returns:
            List[str]: 選中的幣種列表
        """
        if validation_results.empty:
            return []
        
        # 應用篩選條件
        filtered = validation_results[
            (validation_results['stability_score'] >= min_stability) &
            (validation_results['avg_performance'] >= min_performance) &
            (validation_results['std_performance'] <= max_std)
        ]
        
        # 按穩定性得分排序並選取前N個
        selected = filtered.sort_values('stability_score', ascending=False).head(top_n)
        
        selected_symbols = selected['symbol'].tolist()
        self.logger.info(f"Selected {len(selected_symbols)} robust coins for strategy")
        
        return selected_symbols
    
    def backtest_strategy(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str,
                        timeframe: str = '5m',
                        eth_event_params: Optional[Dict] = None,
                        take_profit_pct: float = 0.03,
                        stop_loss_pct: float = 0.015,
                        max_holding_time: int = 60) -> Dict[str, Any]:
        """
        回測基於ETH下跌事件的交易策略
        
        Args:
            symbols: 要回測的幣種列表
            start_date: 開始日期
            end_date: 結束日期
            timeframe: 時間框架
            eth_event_params: ETH事件檢測參數
            take_profit_pct: 止盈百分比
            stop_loss_pct: 止損百分比
            max_holding_time: 最大持倉時間(分鐘)
            
        Returns:
            Dict[str, Any]: 回測結果
        """
        self.logger.info(f"Backtesting strategy from {start_date} to {end_date}")
        
        # 默認ETH事件參數
        if eth_event_params is None:
            eth_event_params = {
                'drop_threshold': -0.02,
                'consecutive_drops': 1,
                'volume_factor': 1.5,
                'pre_event_window': 15,
                'post_event_window': max_holding_time
            }
        
        # 獲取ETH數據和事件
        eth_data = self.data_fetcher.fetch_historical_data(
            symbol='ETHUSDT',
            interval=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if eth_data.empty:
            self.logger.warning("No ETH data available for backtest")
            return {'error': 'No ETH data available'}
        
        # 識別ETH下跌事件
        eth_events = self._identify_eth_events(
            eth_data,
            **eth_event_params
        )
        
        if not eth_events:
            self.logger.warning("No ETH events identified for backtest")
            return {'error': 'No ETH events identified'}
        
        self.logger.info(f"Identified {len(eth_events)} ETH drop events for backtest")
        
        # 獲取交易幣種的數據
        coin_data = {}
        for symbol in symbols:
            try:
                data = self.data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                if not data.empty:
                    coin_data[symbol] = data
            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {e}")
        
        # 執行回測
        backtest_results = self._run_backtest(
            coin_data=coin_data,
            eth_events=eth_events,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_time=max_holding_time
        )
        
        # 保存和可視化結果
        self._save_backtest_results(backtest_results)
        self._visualize_backtest_results(backtest_results)
        
        return backtest_results
    
    def _identify_eth_events(self, 
                           eth_data: pd.DataFrame, 
                           drop_threshold: float = -0.02,
                           consecutive_drops: int = 1,
                           volume_factor: float = 1.5,
                           **kwargs) -> List[Dict]:
        """
        從ETH數據識別下跌事件
        
        Args:
            eth_data: ETH價格數據
            drop_threshold: 下跌閾值
            consecutive_drops: 連續下跌K線數
            volume_factor: 成交量放大倍數
            
        Returns:
            List[Dict]: ETH事件列表
        """
        # 計算價格變化和成交量比率
        eth_data['pct_change'] = eth_data['close'].pct_change()
        eth_data['volume_ratio'] = eth_data['volume'] / eth_data['volume'].rolling(window=10).mean()
        
        # 標記下跌K線
        eth_data['is_drop'] = eth_data['pct_change'] < 0
        eth_data['consecutive_drops'] = eth_data['is_drop'].rolling(consecutive_drops).sum()
        
        # 查找符合條件的事件
        events = []
        
        for idx, row in eth_data.iterrows():
            if pd.isna(row['pct_change']) or pd.isna(row['volume_ratio']) or pd.isna(row['consecutive_drops']):
                continue
                
            # 檢查是否滿足所有條件
            if (row['pct_change'] <= drop_threshold and 
                row['volume_ratio'] >= volume_factor and 
                row['consecutive_drops'] >= consecutive_drops):
                
                events.append({
                    'event_time': idx,
                    'event_price': row['close'],
                    'drop_pct': row['pct_change'],
                    'volume_ratio': row['volume_ratio']
                })
        
        # 過濾重疊事件
        filtered_events = []
        
        if events:
            filtered_events = [events[0]]
            
            for event in events[1:]:
                overlapping = False
                
                for filtered_event in filtered_events:
                    # 確保事件之間至少間隔6小時
                    time_diff = abs((event['event_time'] - filtered_event['event_time']).total_seconds() / 3600)
                    if time_diff < 6:
                        overlapping = True
                        break
                
                if not overlapping:
                    filtered_events.append(event)
        
        return filtered_events
    
    def _run_backtest(self, 
                    coin_data: Dict[str, pd.DataFrame],
                    eth_events: List[Dict],
                    take_profit_pct: float = 0.03,
                    stop_loss_pct: float = 0.015,
                    max_holding_time: int = 60) -> Dict[str, Any]:
        """
        執行交易策略回測
        
        Args:
            coin_data: 幣種數據
            eth_events: ETH事件列表
            take_profit_pct: 止盈百分比
            stop_loss_pct: 止損百分比
            max_holding_time: 最大持倉時間(分鐘)
            
        Returns:
            Dict[str, Any]: 回測結果
        """
        # 交易結果
        trades = []
        symbol_results = {}
        
        for symbol, data in coin_data.items():
            symbol_trades = []
            
            for event in eth_events:
                event_time = event['event_time']
                
                # 檢查數據是否覆蓋事件時間
                if event_time not in data.index:
                    continue
                
                # 獲取進場價格
                entry_price = data.loc[event_time, 'close']
                
                # 計算止盈止損價格
                take_profit_price = entry_price * (1 + take_profit_pct)
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                
                # 定義退出時間窗口
                exit_window_end = event_time + timedelta(minutes=max_holding_time)
                
                # 獲取退出窗口數據
                exit_data = data[(data.index > event_time) & (data.index <= exit_window_end)]
                
                if exit_data.empty:
                    continue
                
                # 模擬交易
                exit_price = entry_price  # 默認退出價格
                exit_time = exit_window_end  # 默認退出時間
                exit_reason = "time_limit"  # 默認退出原因
                
                # 檢查是否觸發止盈/止損
                for t, row in exit_data.iterrows():
                    high = row['high']
                    low = row['low']
                    
                    if high >= take_profit_price:
                        exit_price = take_profit_price
                        exit_time = t
                        exit_reason = "take_profit"
                        break
                        
                    if low <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = t
                        exit_reason = "stop_loss"
                        break
                
                # 如果沒有觸發止盈/止損，使用時間窗口末尾的收盤價
                if exit_reason == "time_limit":
                    exit_price = exit_data.iloc[-1]['close']
                
                # 計算收益率
                profit_pct = (exit_price / entry_price) - 1
                
                # 計算持倉時間
                holding_time = (exit_time - event_time).total_seconds() / 60  # 分鐘
                
                # 記錄交易
                trade = {
                    'symbol': symbol,
                    'event_time': event_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'holding_time': holding_time,
                    'exit_reason': exit_reason
                }
                
                symbol_trades.append(trade)
                trades.append(trade)
            
            # 計算此幣種的回測結果
            if symbol_trades:
                win_trades = [t for t in symbol_trades if t['profit_pct'] > 0]
                
                symbol_results[symbol] = {
                    'total_trades': len(symbol_trades),
                    'win_trades': len(win_trades),
                    'win_rate': len(win_trades) / len(symbol_trades) if symbol_trades else 0,
                    'avg_profit': np.mean([t['profit_pct'] for t in symbol_trades]),
                    'total_profit': np.sum([t['profit_pct'] for t in symbol_trades]),
                    'max_profit': max([t['profit_pct'] for t in symbol_trades]) if symbol_trades else 0,
                    'max_loss': min([t['profit_pct'] for t in symbol_trades]) if symbol_trades else 0,
                    'avg_holding_time': np.mean([t['holding_time'] for t in symbol_trades]),
                    'take_profit_exits': len([t for t in symbol_trades if t['exit_reason'] == 'take_profit']),
                    'stop_loss_exits': len([t for t in symbol_trades if t['exit_reason'] == 'stop_loss']),
                    'time_limit_exits': len([t for t in symbol_trades if t['exit_reason'] == 'time_limit'])
                }
        
        # 計算整體回測結果
        if trades:
            win_trades = [t for t in trades if t['profit_pct'] > 0]
            
            overall_results = {
                'total_trades': len(trades),
                'win_trades': len(win_trades),
                'win_rate': len(win_trades) / len(trades) if trades else 0,
                'avg_profit': np.mean([t['profit_pct'] for t in trades]),
                'total_profit': np.sum([t['profit_pct'] for t in trades]),
                'max_profit': max([t['profit_pct'] for t in trades]) if trades else 0,
                'max_loss': min([t['profit_pct'] for t in trades]) if trades else 0,
                'avg_holding_time': np.mean([t['holding_time'] for t in trades]),
                'take_profit_exits': len([t for t in trades if t['exit_reason'] == 'take_profit']),
                'stop_loss_exits': len([t for t in trades if t['exit_reason'] == 'stop_loss']),
                'time_limit_exits': len([t for t in trades if t['exit_reason'] == 'time_limit'])
            }
        else:
            overall_results = {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0
            }
        
        # 組合完整結果
        backtest_results = {
            'overall': overall_results,
            'by_symbol': symbol_results,
            'trades': trades,
            'params': {
                'take_profit_pct': take_profit_pct,
                'stop_loss_pct': stop_loss_pct,
                'max_holding_time': max_holding_time
            }
        }
        
        return backtest_results
    
    def _save_backtest_results(self, results: Dict[str, Any]) -> None:
        """
        保存回測結果
        
        Args:
            results: 回測結果
        """
        # 創建時間戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存整體結果
        overall_file = os.path.join(self.output_dir, f"backtest_overall_{timestamp}.json")
        
        with open(overall_file, 'w') as f:
            # 僅保存可序列化的整體和參數數據
            json_data = {
                'overall': results['overall'],
                'params': results['params'],
                'timestamp': timestamp
            }
            json.dump(json_data, f, indent=4)
        
        # 保存按幣種的結果
        symbol_results_file = os.path.join(self.output_dir, f"backtest_by_symbol_{timestamp}.csv")
        
        symbol_df = pd.DataFrame([
            {**{'symbol': symbol}, **stats}
            for symbol, stats in results['by_symbol'].items()
        ])
        
        if not symbol_df.empty:
            symbol_df.to_csv(symbol_results_file, index=False)
        
        # 保存所有交易
        trades_file = os.path.join(self.output_dir, f"backtest_trades_{timestamp}.csv")
        
        trades_df = pd.DataFrame(results['trades'])
        
        if not trades_df.empty:
            trades_df.to_csv(trades_file, index=False)
        
        self.logger.info(f"Backtest results saved to {self.output_dir}")
    
    def _visualize_backtest_results(self, results: Dict[str, Any]) -> None:
        """
        可視化回測結果
        
        Args:
            results: 回測結果
        """
        if not results['trades']:
            self.logger.warning("No trades to visualize")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 創建交易結果數據框
        trades_df = pd.DataFrame(results['trades'])
        
        # 按幣種的表現對比
        plt.figure(figsize=(14, 8))
        
        # 準備按幣種的總利潤數據
        symbols = list(results['by_symbol'].keys())
        total_profits = [results['by_symbol'][s]['total_profit'] for s in symbols]
        
        # 按總利潤排序
        sorted_indices = np.argsort(total_profits)[::-1]
        sorted_symbols = [symbols[i] for i in sorted_indices]
        sorted_profits = [total_profits[i] for i in sorted_indices]
        
        # 繪制總利潤條形圖
        sns.barplot(x=sorted_symbols[:15], y=sorted_profits[:15], palette='viridis')
        plt.title('總利潤 - 按幣種', fontsize=14)
        plt.xlabel('幣種', fontsize=12)
        plt.ylabel('總利潤 (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存圖表
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"backtest_profit_by_symbol_{timestamp}.png"), dpi=300)
        plt.close()
        
        # 繪制勝率對比
        plt.figure(figsize=(14, 8))
        
        # 準備勝率數據
        win_rates = [results['by_symbol'][s]['win_rate'] * 100 for s in symbols]
        
        # 按勝率排序
        sorted_indices = np.argsort(win_rates)[::-1]
        sorted_symbols = [symbols[i] for i in sorted_indices]
        sorted_win_rates = [win_rates[i] for i in sorted_indices]
        
        # 繪制勝率條形圖
        sns.barplot(x=sorted_symbols[:15], y=sorted_win_rates[:15], palette='viridis')
        plt.title('勝率 - 按幣種', fontsize=14)
        plt.xlabel('幣種', fontsize=12)
        plt.ylabel('勝率 (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存圖表
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"backtest_win_rate_by_symbol_{timestamp}.png"), dpi=300)
        plt.close()
        
        # 繪制持倉時間分布
        plt.figure(figsize=(14, 8))
        
        # 按退出原因繪制持倉時間分布
        sns.histplot(
            data=trades_df, 
            x='holding_time', 
            hue='exit_reason', 
            multiple='stack',
            bins=20,
            palette='viridis'
        )
        plt.title('持倉時間分布 - 按退出原因', fontsize=14)
        plt.xlabel('持倉時間 (分鐘)', fontsize=12)
        plt.ylabel('交易次數', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存圖表
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"backtest_holding_time_{timestamp}.png"), dpi=300)
        plt.close()
        
        # 繪制盈虧散點圖
        plt.figure(figsize=(14, 8))
        
        # 按幣種和退出原因繪制盈虧散點圖
        scatter = plt.scatter(
            x=trades_df['holding_time'],
            y=trades_df['profit_pct'] * 100,  # 轉換為百分比
            c=trades_df['exit_reason'].map({'take_profit': 0, 'stop_loss': 1, 'time_limit': 2}),
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        # 添加圖例
        handles, labels = scatter.legend_elements()
        plt.legend(handles, ['止盈', '止損', '時間限制'])
        
        plt.title('盈虧分布 vs 持倉時間', fontsize=14)
        plt.xlabel('持倉時間 (分鐘)', fontsize=12)
        plt.ylabel('盈虧 (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 保存圖表
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"backtest_profit_vs_holding_time_{timestamp}.png"), dpi=300)
        plt.close()
        
    def optimize_strategy_parameters(self, 
                                  symbols: List[str],
                                  validation_period: Tuple[str, str],
                                  test_period: Tuple[str, str],
                                  timeframe: str = '5m') -> Dict[str, Any]:
        """
        優化交易策略參數
        
        Args:
            symbols: 要測試的幣種列表
            validation_period: 驗證期(開始日期, 結束日期)
            test_period: 測試期(開始日期, 結束日期)
            timeframe: 時間框架
            
        Returns:
            Dict[str, Any]: 優化結果
        """
        self.logger.info(f"Optimizing strategy parameters for {len(symbols)} coins")
        
        # 可選參數範圍
        drop_thresholds = [-0.01, -0.015, -0.02, -0.025, -0.03]
        volume_factors = [1.2, 1.5, 1.8, 2.0]
        take_profits = [0.02, 0.025, 0.03, 0.035, 0.04]
        stop_losses = [0.01, 0.015, 0.02, 0.025]
        holding_times = [30, 45, 60, 90, 120]
        
        # 存儲所有參數組合的結果
        all_results = []
        
        # 測試不同參數組合
        for drop_threshold in drop_thresholds:
            for volume_factor in volume_factors:
                for take_profit in take_profits:
                    for stop_loss in stop_losses:
                        for holding_time in holding_times:
                            # 創建參數組合
                            params = {
                                'drop_threshold': drop_threshold,
                                'volume_factor': volume_factor,
                                'take_profit_pct': take_profit,
                                'stop_loss_pct': stop_loss,
                                'max_holding_time': holding_time
                            }
                            
                            self.logger.info(f"Testing parameters: {params}")
                            
                            # 在驗證期測試參數
                            eth_event_params = {
                                'drop_threshold': drop_threshold,
                                'consecutive_drops': 1,
                                'volume_factor': volume_factor,
                                'pre_event_window': 15,
                                'post_event_window': holding_time
                            }
                            
                            validation_results = self.backtest_strategy(
                                symbols=symbols,
                                start_date=validation_period[0],
                                end_date=validation_period[1],
                                timeframe=timeframe,
                                eth_event_params=eth_event_params,
                                take_profit_pct=take_profit,
                                stop_loss_pct=stop_loss,
                                max_holding_time=holding_time
                            )
                            
                            # 記錄結果
                            result = {
                                **params,
                                'validation_win_rate': validation_results['overall']['win_rate'],
                                'validation_avg_profit': validation_results['overall']['avg_profit'],
                                'validation_total_profit': validation_results['overall']['total_profit'],
                                'validation_trades': validation_results['overall']['total_trades']
                            }
                            
                            all_results.append(result)
        
        # 找出驗證期表現最佳的參數
        if all_results:
            # 按總利潤排序
            sorted_results = sorted(all_results, key=lambda x: x['validation_total_profit'], reverse=True)
            best_params = sorted_results[0]
            
            self.logger.info(f"Best parameters: {best_params}")
            
            # 在測試期使用最佳參數
            eth_event_params = {
                'drop_threshold': best_params['drop_threshold'],
                'consecutive_drops': 1,
                'volume_factor': best_params['volume_factor'],
                'pre_event_window': 15,
                'post_event_window': best_params['max_holding_time']
            }
            
            test_results = self.backtest_strategy(
                symbols=symbols,
                start_date=test_period[0],
                end_date=test_period[1],
                timeframe=timeframe,
                eth_event_params=eth_event_params,
                take_profit_pct=best_params['take_profit_pct'],
                stop_loss_pct=best_params['stop_loss_pct'],
                max_holding_time=best_params['max_holding_time']
            )
            
            # 添加測試期結果
            best_params['test_win_rate'] = test_results['overall']['win_rate']
            best_params['test_avg_profit'] = test_results['overall']['avg_profit']
            best_params['test_total_profit'] = test_results['overall']['total_profit']
            best_params['test_trades'] = test_results['overall']['total_trades']
            
            # 保存所有參數測試結果
            results_df = pd.DataFrame(all_results)
            results_file = os.path.join(self.output_dir, "parameter_optimization_results.csv")
            results_df.to_csv(results_file, index=False)
            
            # 保存最佳參數
            best_params_file = os.path.join(self.output_dir, "best_strategy_parameters.json")
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f, indent=4)
            
            return {
                'best_params': best_params,
                'all_results': all_results
            }
        else:
            return {'error': 'No valid parameter combinations found'} 