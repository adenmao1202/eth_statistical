#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Event Analyzer Module
Integrates hypothesis testing module to perform rigorous event studies across multiple time periods
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import Progress
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure parent directory is in path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local imports
from hypothesis_testing import HypothesisTesting
import config
from data_fetcher import DataFetcher

# Initialize console for rich output
console = Console()

class RobustEventAnalyzer:
    """
    Robust Event Analyzer integrates hypothesis testing functionality 
    to perform rigorous event studies across multiple time periods
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the robust event analyzer
        
        Args:
            data_fetcher (DataFetcher): Data fetcher instance
        """
        self.data_fetcher = data_fetcher
        self.hypothesis_testing = HypothesisTesting(data_fetcher)
        self.reference_symbol = config.DEFAULT_REFERENCE_SYMBOL
        
        # Analysis results
        self.periods = []
        self.results_by_period = {}
        self.stability_scores = {}
        
        # Output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "results/robust_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_robust_analysis(self, 
                           symbols: List[str], 
                           total_days: int = 120,
                           period_length: int = 30,
                           overlap_days: int = 0,
                           timeframe: str = '5m',
                           eth_event_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run event study analysis across multiple time periods for robustness
        
        Args:
            symbols: List of cryptocurrency symbols to analyze
            total_days: Total number of days to analyze
            period_length: Length of each period in days
            overlap_days: Number of days overlap between periods
            timeframe: Data timeframe
            eth_event_params: Parameters for ETH event detection
            
        Returns:
            Dict: Analysis results
        """
        console.print("[bold blue]Running robust event analysis across multiple time periods[/bold blue]")
        
        # Default ETH event parameters if not provided
        if eth_event_params is None:
            eth_event_params = {
                'drop_threshold': -0.02,  # 2% drop
                'consecutive_drops': 1,
                'volume_factor': 1.5
            }
            
        # Calculate periods
        now = datetime.now()
        period_step = period_length - overlap_days
        
        # Generate non-overlapping time periods
        self.periods = []
        for i in range(0, total_days, period_step):
            if i + period_length > total_days:
                break
                
            end_date = now - timedelta(days=i)
            start_date = end_date - timedelta(days=period_length)
            
            self.periods.append({
                'period_id': len(self.periods) + 1,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
        
        console.print(f"[green]Analyzing {len(symbols)} coins across {len(self.periods)} time periods[/green]")
        
        # Analyze each period
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            period_task = progress.add_task("[cyan]Analyzing periods...", total=len(self.periods))
            
            for period in self.periods:
                period_id = period['period_id']
                start_date = period['start_date']
                end_date = period['end_date']
                
                progress.update(period_task, description=f"[cyan]Analyzing period {period_id}: {start_date} to {end_date}[/cyan]")
                
                # Run hypothesis testing for this period
                results = self._analyze_period(
                    period_id=period_id,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    eth_event_params=eth_event_params
                )
                
                self.results_by_period[period_id] = results
                progress.update(period_task, advance=1)
        
        # Calculate stability scores across periods
        stability_results = self._calculate_cross_period_stability()
        
        # Generate reports
        self._generate_reports()
        
        # Return results
        return {
            'periods': self.periods,
            'results_by_period': self.results_by_period,
            'stability_scores': self.stability_scores,
            'stability_results': stability_results
        }
    
    def _analyze_period(self,
                      period_id: int,
                      symbols: List[str],
                      start_date: str,
                      end_date: str,
                      timeframe: str,
                      eth_event_params: Dict) -> Dict[str, Any]:
        """
        Analyze a single time period using hypothesis testing
        
        Args:
            period_id: Period identifier
            symbols: List of cryptocurrency symbols
            start_date: Start date of the period
            end_date: End date of the period
            timeframe: Data timeframe
            eth_event_params: Parameters for ETH event detection
            
        Returns:
            Dict: Period analysis results
        """
        try:
            # Configure hypothesis testing parameters
            self.hypothesis_testing.pre_event_window = config.PRE_EVENT_WINDOW
            self.hypothesis_testing.post_event_window = config.POST_EVENT_WINDOW
            
            # Fetch ETH data for this period
            eth_symbol = "ETHUSDT"
            eth_data = self.data_fetcher.fetch_historical_data(
                symbol=eth_symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if eth_data.empty:
                console.print(f"[bold red]No ETH data available for period {period_id}[/bold red]")
                return {}
            
            # Identify ETH drop events
            events = self.hypothesis_testing.identify_eth_drop_events(
                eth_data=eth_data,
                drop_threshold=eth_event_params.get('drop_threshold', -0.02),
                window_size=config.DETECTION_WINDOW_SIZE,
                consecutive_drops=eth_event_params.get('consecutive_drops', 1),
                volume_factor=eth_event_params.get('volume_factor', 1.5)
            )
            
            if not events:
                console.print(f"[yellow]No ETH drop events found in period {period_id}[/yellow]")
                return {
                    'eth_events': [],
                    'test_results': pd.DataFrame(),
                    'abnormal_returns': {},
                    'summary': {
                        'total_events': 0,
                        'significant_coins': 0
                    }
                }
            
            # Fetch data for all symbols
            coin_data = {}
            for symbol in symbols:
                if symbol == self.reference_symbol:
                    continue
                
                self.logger.info(f"正在获取 {symbol} 的历史数据...")
                
                try:
                    # 使用symbolUSDT作为交易对
                    full_symbol = symbol if symbol.endswith('USDT') else f"{symbol}USDT"
                    data = self.data_fetcher.fetch_historical_data(
                        symbol=full_symbol,
                        interval=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not data.empty:
                        coin_data[symbol] = data
                except Exception as e:
                    console.print(f"[bold red]Error fetching data for {symbol}: {e}[/bold red]")
            
            # Calculate returns
            pre_returns, post_returns = self.hypothesis_testing.calculate_returns(
                coin_data=coin_data,
                event_periods=events
            )
            
            # Conduct statistical tests
            test_results = self.hypothesis_testing.conduct_statistical_tests(
                pre_event_returns=pre_returns,
                post_event_returns=post_returns
            )
            
            # Calculate average abnormal returns
            abnormal_returns = {}
            for symbol in symbols:
                if symbol in pre_returns and symbol in post_returns:
                    # Calculate abnormal return (post - pre)
                    abnormal_returns[symbol] = np.mean(post_returns[symbol]) - np.mean(pre_returns[symbol])
            
            # Return period results
            return {
                'eth_events': events,
                'test_results': test_results,
                'abnormal_returns': abnormal_returns,
                'summary': {
                    'total_events': len(events),
                    'significant_coins': len(test_results[test_results['is_significant'] == True])
                }
            }
            
        except Exception as e:
            console.print(f"[bold red]Error analyzing period {period_id}: {e}[/bold red]")
            return {}
    
    def _calculate_cross_period_stability(self) -> pd.DataFrame:
        """
        Calculate stability scores for coins across multiple periods
        
        Returns:
            pd.DataFrame: Stability analysis results
        """
        console.print("[bold blue]Calculating cross-period stability scores[/bold blue]")
        
        # Collect all unique symbols
        all_symbols = set()
        for period_id, results in self.results_by_period.items():
            if 'test_results' in results and not results['test_results'].empty:
                all_symbols.update(results['test_results']['symbol'].unique())
        
        # Initialize data structures
        stability_data = []
        
        # For each symbol, calculate stability metrics
        for symbol in all_symbols:
            # Collect data across periods
            periods_present = 0
            periods_significant = 0
            abnormal_returns = []
            t_values = []
            p_values = []
            effect_sizes = []
            
            for period_id, results in self.results_by_period.items():
                if 'test_results' not in results or results['test_results'].empty:
                    continue
                    
                # Check if symbol is present in this period
                symbol_results = results['test_results'][results['test_results']['symbol'] == symbol]
                
                if len(symbol_results) > 0:
                    periods_present += 1
                    
                    # Check if significant in this period
                    if symbol_results['is_significant'].values[0]:
                        periods_significant += 1
                    
                    # Collect statistics
                    t_values.append(symbol_results['t_statistic'].values[0])
                    p_values.append(symbol_results['p_value'].values[0])
                    
                    if 'effect_size' in symbol_results.columns:
                        effect_sizes.append(symbol_results['effect_size'].values[0])
                
                # Collect abnormal returns
                if 'abnormal_returns' in results and symbol in results['abnormal_returns']:
                    abnormal_returns.append(results['abnormal_returns'][symbol])
            
            # Calculate stability metrics
            if periods_present > 0:
                # Significance stability: percentage of periods where coin showed significant results
                significance_stability = periods_significant / periods_present
                
                # T-value stability: standard deviation of t-values across periods (lower is better)
                t_value_stability = np.std(t_values) if len(t_values) > 1 else np.nan
                
                # Direction stability: consistency of abnormal return direction
                if len(abnormal_returns) > 1:
                    positive_returns = sum(1 for r in abnormal_returns if r > 0)
                    negative_returns = sum(1 for r in abnormal_returns if r < 0)
                    direction_stability = max(positive_returns, negative_returns) / len(abnormal_returns)
                else:
                    direction_stability = np.nan
                
                # Magnitude stability: coefficient of variation of absolute abnormal returns
                if len(abnormal_returns) > 1:
                    abs_returns = [abs(r) for r in abnormal_returns]
                    magnitude_stability = 1 - (np.std(abs_returns) / np.mean(abs_returns)) if np.mean(abs_returns) > 0 else 0
                else:
                    magnitude_stability = np.nan
                
                # Combined stability score
                weights = {
                    'significance': 0.3,
                    'direction': 0.3,
                    'magnitude': 0.2,
                    't_value': 0.2
                }
                
                # Normalize t-value stability (lower is better)
                norm_t_value_stability = 1 / (1 + t_value_stability) if not np.isnan(t_value_stability) else 0
                
                # Calculate weighted score
                weighted_score = (
                    weights['significance'] * significance_stability +
                    weights['direction'] * direction_stability +
                    weights['magnitude'] * magnitude_stability +
                    weights['t_value'] * norm_t_value_stability
                )
                
                # Store data
                stability_data.append({
                    'symbol': symbol,
                    'periods_present': periods_present,
                    'periods_significant': periods_significant,
                    'significance_stability': significance_stability,
                    'direction_stability': direction_stability,
                    'magnitude_stability': magnitude_stability,
                    't_value_stability': t_value_stability,
                    'mean_abnormal_return': np.mean(abnormal_returns) if abnormal_returns else np.nan,
                    'std_abnormal_return': np.std(abnormal_returns) if len(abnormal_returns) > 1 else np.nan,
                    'weighted_score': weighted_score,
                    'period_coverage': periods_present / len(self.periods)
                })
        
        # Create DataFrame and sort by weighted score
        stability_df = pd.DataFrame(stability_data)
        if not stability_df.empty:
            stability_df = stability_df.sort_values('weighted_score', ascending=False)
            
            # Store in instance
            self.stability_scores = {row['symbol']: row['weighted_score'] for _, row in stability_df.iterrows()}
        
        return stability_df
    
    def _generate_reports(self) -> None:
        """Generate detailed analysis reports and visualizations"""
        console.print("[bold blue]Generating analysis reports and visualizations[/bold blue]")
        
        try:
            # Create period summary report
            period_summary = []
            for period in self.periods:
                period_id = period['period_id']
                
                if period_id in self.results_by_period:
                    results = self.results_by_period[period_id]
                    
                    summary = {
                        'period_id': period_id,
                        'start_date': period['start_date'],
                        'end_date': period['end_date'],
                        'eth_events': len(results.get('eth_events', [])),
                        'coins_analyzed': len(results.get('abnormal_returns', {})),
                        'significant_coins': results.get('summary', {}).get('significant_coins', 0)
                    }
                    
                    period_summary.append(summary)
            
            # Save period summary
            period_df = pd.DataFrame(period_summary)
            period_df.to_csv(os.path.join(self.output_dir, 'period_summary.csv'), index=False)
            
            # Generate detailed report of stability analysis
            stability_df = self._calculate_cross_period_stability()
            if not stability_df.empty:
                stability_df.to_csv(os.path.join(self.output_dir, 'stability_analysis.csv'), index=False)
                
                # Generate plots
                self._generate_stability_visualizations(stability_df)
            
            # Generate comprehensive report in JSON format
            report = {
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'periods_analyzed': len(self.periods),
                'periods': [{'id': p['period_id'], 'start': p['start_date'], 'end': p['end_date']} for p in self.periods],
                'top_stable_coins': self._get_top_stable_coins(10),
                'period_stats': period_summary
            }
            
            with open(os.path.join(self.output_dir, 'analysis_report.json'), 'w') as f:
                json.dump(report, f, indent=4)
                
            console.print(f"[bold green]Reports saved to {self.output_dir}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Error generating reports: {e}[/bold red]")
    
    def _generate_stability_visualizations(self, stability_df: pd.DataFrame) -> None:
        """
        Generate visualizations for stability analysis
        
        Args:
            stability_df: DataFrame with stability analysis results
        """
        if stability_df.empty:
            return
            
        try:
            # 1. Top coins by stability score
            plt.figure(figsize=(10, 6))
            top_n = min(20, len(stability_df))
            sns.barplot(
                data=stability_df.head(top_n),
                x='weighted_score',
                y='symbol',
                palette='viridis'
            )
            plt.title('Top Coins by Stability Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'top_stable_coins.png'))
            plt.close()
            
            # 2. Stability components comparison for top coins
            top_10 = stability_df.head(10)
            
            plt.figure(figsize=(12, 8))
            
            # Create a normalized stacked bar chart
            stability_components = ['significance_stability', 'direction_stability', 
                                   'magnitude_stability', 't_value_stability']
            
            # Normalize t_value_stability
            top_10_norm = top_10.copy()
            top_10_norm['t_value_stability'] = 1 / (1 + top_10['t_value_stability'])
            
            # Create stacked bar
            top_10_norm[stability_components].plot(
                kind='bar', 
                stacked=True, 
                figsize=(12, 6),
                colormap='viridis',
                width=0.8
            )
            
            plt.title('Stability Components for Top 10 Coins')
            plt.xlabel('Coin')
            plt.ylabel('Stability Score Components')
            plt.xticks(range(len(top_10)), top_10['symbol'], rotation=45)
            plt.legend(title='Stability Components')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'stability_components.png'))
            plt.close()
            
            # 3. Scatter plot of stability vs. mean abnormal return
            plt.figure(figsize=(10, 6))
            scatter = sns.scatterplot(
                data=stability_df,
                x='mean_abnormal_return',
                y='weighted_score',
                size='period_coverage',
                sizes=(20, 200),
                alpha=0.7,
                palette='viridis'
            )
            
            # Annotate top coins
            for idx, row in stability_df.head(10).iterrows():
                plt.annotate(
                    row['symbol'],
                    (row['mean_abnormal_return'], row['weighted_score']),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
            
            plt.title('Stability Score vs. Mean Abnormal Return')
            plt.xlabel('Mean Abnormal Return')
            plt.ylabel('Stability Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'stability_vs_return.png'))
            plt.close()
            
        except Exception as e:
            console.print(f"[bold yellow]Error generating visualizations: {e}[/bold yellow]")
    
    def _get_top_stable_coins(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top stable coins based on weighted stability score
        
        Args:
            top_n: Number of top coins to return
            
        Returns:
            List[Dict]: List of top stable coins with their metrics
        """
        stability_df = self._calculate_cross_period_stability()
        
        if stability_df.empty:
            return []
            
        top_coins = []
        for _, row in stability_df.head(top_n).iterrows():
            top_coins.append({
                'symbol': row['symbol'],
                'stability_score': row['weighted_score'],
                'mean_abnormal_return': row['mean_abnormal_return'],
                'period_coverage': row['period_coverage'],
                'significance_stability': row['significance_stability']
            })
            
        return top_coins 

    def _analyze_single_period(self, 
                             symbols: List[str], 
                             period: Dict[str, Any],
                             timeframe: str,
                             eth_event_params: Dict[str, Any],
                             progress: Progress = None,
                             task_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze a single validation period
        
        Args:
            symbols: List of symbols to analyze
            period: Period details (period_id, start_date, end_date)
            timeframe: Data timeframe
            eth_event_params: Parameters for ETH event detection
            progress: Progress object
            task_id: Task ID for updating progress
            
        Returns:
            Dict: Period analysis results
        """
        period_id = period['period_id']
        start_date = period['start_date']
        end_date = period['end_date']
        
        if progress:
            progress.update(task_id, description=f"[cyan]Period {period_id}: {start_date} to {end_date}[/cyan]")
        
        try:
            # 計算日期間隔天數
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days + 1  # 加1包含結束日期
            
            # Fetch ETH data for this period
            eth_symbol = "ETHUSDT"
            eth_data = self.data_fetcher.fetch_historical_data(
                symbol=eth_symbol,
                interval=timeframe,
                days=days_diff,
                end_date=end_date
            )
            
            if eth_data.empty:
                return {'error': f"No ETH data available for period {period_id} ({start_date} to {end_date})"}
                
            # Identify ETH events
            events = self._identify_eth_events(
                eth_data=eth_data,
                drop_threshold=eth_event_params.get('drop_threshold', -0.01),
                consecutive_drops=eth_event_params.get('consecutive_drops', 1),
                volume_factor=eth_event_params.get('volume_factor', 1.5)
            )
            
            if not events:
                return {
                    'eth_events': [],
                    'coin_responses': {},
                    'response_stats': {}
                }
            
            # Get data for all symbols
            coin_data = {}
            for symbol in symbols:
                try:
                    data = self.data_fetcher.fetch_historical_data(
                        symbol=f"{symbol}USDT",
                        interval=timeframe,
                        days=days_diff,
                        end_date=end_date
                    )
                    if not data.empty:
                        coin_data[symbol] = data
                except Exception as e:
                    console.print(f"[bold red]Error fetching data for {symbol}: {e}[/bold red]")
            
            # Calculate returns
            pre_returns, post_returns = self.hypothesis_testing.calculate_returns(
                coin_data=coin_data,
                event_periods=events
            )
            
            # Conduct statistical tests
            test_results = self.hypothesis_testing.conduct_statistical_tests(
                pre_event_returns=pre_returns,
                post_event_returns=post_returns
            )
            
            # Calculate average abnormal returns
            abnormal_returns = {}
            for symbol in symbols:
                if symbol in pre_returns and symbol in post_returns:
                    # Calculate abnormal return (post - pre)
                    abnormal_returns[symbol] = np.mean(post_returns[symbol]) - np.mean(pre_returns[symbol])
            
            # Return period results
            return {
                'eth_events': events,
                'test_results': test_results,
                'abnormal_returns': abnormal_returns,
                'summary': {
                    'total_events': len(events),
                    'significant_coins': len(test_results[test_results['is_significant'] == True])
                }
            }
            
        except Exception as e:
            console.print(f"[bold red]Error analyzing period {period_id}: {e}[/bold red]")
            return {} 