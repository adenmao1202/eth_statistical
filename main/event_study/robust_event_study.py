#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Event Study Module
Integrates coin classification and time series validation for a 
comprehensive event study without look-ahead bias
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
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import time
import math
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

# Import coin classifier and time series validator
from coin_classifier import CoinClassifier
from time_series_validator import TimeSeriesValidator

# Initialize Rich console
console = Console()

class RobustEventStudy:
    """
    Robust Event Study integrates hypothesis testing, coin classification, 
    and time series validation to perform robust event studies
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the robust event study
        
        Args:
            data_fetcher (DataFetcher): Data fetcher instance
        """
        self.data_fetcher = data_fetcher
        self.hypothesis_testing = HypothesisTesting(data_fetcher)
        self.coin_classifier = CoinClassifier(data_fetcher)
        self.time_validator = TimeSeriesValidator(data_fetcher)
        
        # Reference symbol
        self.reference_symbol = config.DEFAULT_REFERENCE_SYMBOL
        
        # Output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "results/event_study")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_analysis(self,
                    classification_methods: List[str] = ["market_cap", "volume", "volatility"],
                    total_days: int = 120,
                    period_length: int = 30,
                    timeframe: str = '5m',
                    eth_event_params: Optional[Dict] = None,
                    output_suffix: str = "") -> Dict[str, Any]:
        """
        Run a comprehensive event study analysis across multiple time periods
        for different coin classifications
        
        Args:
            classification_methods: Methods to classify coins ("market_cap", "volume", "volatility")
            total_days: Total number of days to analyze
            period_length: Length of each period in days
            timeframe: Data timeframe
            eth_event_params: Parameters for ETH event detection
            output_suffix: Suffix for output directory name
            
        Returns:
            Dict: Analysis results
        """
        console.print(Panel(
            "[bold blue]Robust Event Study Analysis[/bold blue]\n\n"
            f"Classification methods: {', '.join(classification_methods)}\n"
            f"Analysis period: {total_days} days in {period_length}-day segments\n"
            f"Timeframe: {timeframe}",
            border_style="blue",
            expand=False
        ))
        
        # Default ETH event parameters if not provided
        if eth_event_params is None:
            eth_event_params = {
                'drop_threshold': -0.01,  # 1% drop
                'consecutive_drops': 2,
                'volume_factor': 1.5
            }
        
        # Prepare output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_suffix:
            output_dir = os.path.join(self.output_dir, f"{timestamp}_{output_suffix}")
        else:
            output_dir = os.path.join(self.output_dir, timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Fetch all available symbols
        console.print("[cyan]Fetching all available symbols...[/cyan]")
        all_symbols = self.data_fetcher._get_symbols()
        
        if not all_symbols:
            console.print("[bold red]Failed to fetch symbols. Check your connection and API settings.[/bold red]")
            return {}
        
        console.print(f"[green]Found {len(all_symbols)} tradable symbols[/green]")
        
        # Step 2: Classify coins based on specified methods
        coin_classifications = self._classify_coins(all_symbols, classification_methods, period_length, timeframe)
        
        # Save classification results
        with open(os.path.join(output_dir, "coin_classifications.json"), "w") as f:
            # Convert sets to lists for JSON serialization
            serializable_classifications = {}
            for method, tiers in coin_classifications.items():
                serializable_classifications[method] = {
                    tier: list(symbols) for tier, symbols in tiers.items()
                }
            json.dump(serializable_classifications, f, indent=4)
        
        # Step 3: Calculate validation periods
        validation_periods = self._calculate_validation_periods(total_days, period_length)
        
        # Save periods information
        with open(os.path.join(output_dir, "validation_periods.json"), "w") as f:
            json.dump(validation_periods, f, indent=4)
        
        # Step 4: Run event study for each classification method and period
        analysis_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            # Calculate total tasks
            total_tasks = sum(len(tiers) for method, tiers in coin_classifications.items()) * len(validation_periods)
            analysis_task = progress.add_task("[cyan]Running event study analysis...", total=total_tasks)
            
            for method, tiers in coin_classifications.items():
                analysis_results[method] = {}
                
                for tier_name, symbols in tiers.items():
                    analysis_results[method][tier_name] = {}
                    
                    progress.update(analysis_task, description=f"[cyan]Analyzing {method} - {tier_name}[/cyan]")
                    
                    # Run analysis for this classification across all periods
                    tier_results = self._analyze_classification_across_periods(
                        symbols=list(symbols),
                        periods=validation_periods,
                        timeframe=timeframe,
                        eth_event_params=eth_event_params,
                        progress=progress,
                        analysis_task=analysis_task
                    )
                    
                    analysis_results[method][tier_name] = tier_results
                    
                    # Save tier results to file
                    with open(os.path.join(output_dir, f"{method}_{tier_name}_results.json"), "w") as f:
                        json.dump(tier_results, f, indent=4)
        
        # Step 5: Calculate cross-period stability and consolidated results
        stability_results = self._calculate_stability_across_periods(analysis_results, validation_periods)
        
        # Save stability results
        with open(os.path.join(output_dir, "stability_results.json"), "w") as f:
            json.dump(stability_results, f, indent=4)
        
        # Step 6: Generate comprehensive report
        self._generate_summary_report(analysis_results, stability_results, validation_periods, 
                                     eth_event_params, output_dir)
        
        # Return results
        return {
            'analysis_results': analysis_results,
            'stability_results': stability_results,
            'validation_periods': validation_periods,
            'output_dir': output_dir
        }
    
    def _classify_coins(self, 
                       all_symbols: List[str], 
                       methods: List[str],
                       period_length: int,
                       timeframe: str) -> Dict[str, Dict[str, set]]:
        """
        Classify coins using specified methods
        
        Args:
            all_symbols: List of all available symbols
            methods: List of classification methods
            period_length: Period length in days for relevant classifications
            timeframe: Data timeframe
            
        Returns:
            Dict: Classification results by method and tier
        """
        classifications = {}
        
        # Filter out stablecoins
        stablecoins = self.coin_classifier.get_stable_coins(all_symbols)
        filtered_symbols = [s for s in all_symbols if s not in stablecoins]
        
        console.print(f"[yellow]Excluded {len(stablecoins)} stablecoins from analysis[/yellow]")
        
        # Process each classification method
        for method in methods:
            classifications[method] = {}
            
            if method == "market_cap":
                console.print("[cyan]Classifying coins by market cap...[/cyan]")
                try:
                    market_cap_tiers = self.coin_classifier.classify_by_market_cap(filtered_symbols)
                    for tier, symbols in market_cap_tiers.items():
                        classifications[method][tier] = set(symbols)
                    console.print(f"[green]Market cap classification completed with {len(market_cap_tiers)} tiers[/green]")
                except Exception as e:
                    console.print(f"[bold red]Market cap classification failed: {e}[/bold red]")
                    # Use simple grouping as fallback
                    classifications[method]["all"] = set(filtered_symbols[:100]) 
            
            elif method == "volume":
                console.print("[cyan]Classifying coins by trading volume...[/cyan]")
                try:
                    volume_tiers = self.coin_classifier.classify_by_volume(
                        filtered_symbols,
                        days=min(30, period_length),
                        timeframe=timeframe
                    )
                    for tier, symbols in volume_tiers.items():
                        classifications[method][tier] = set(symbols)
                    console.print(f"[green]Volume classification completed with {len(volume_tiers)} tiers[/green]")
                except Exception as e:
                    console.print(f"[bold red]Volume classification failed: {e}[/bold red]")
            
            elif method == "volatility":
                console.print("[cyan]Classifying coins by price volatility...[/cyan]")
                try:
                    volatility_tiers = self.coin_classifier.classify_by_volatility(
                        filtered_symbols,
                        days=min(30, period_length),
                        timeframe=timeframe
                    )
                    for tier, symbols in volatility_tiers.items():
                        classifications[method][tier] = set(symbols)
                    console.print(f"[green]Volatility classification completed with {len(volatility_tiers)} tiers[/green]")
                except Exception as e:
                    console.print(f"[bold red]Volatility classification failed: {e}[/bold red]")
            
            elif method == "random":
                console.print("[cyan]Creating random coin baskets...[/cyan]")
                try:
                    random_baskets = self.coin_classifier.create_index_baskets(filtered_symbols)
                    for basket, symbols in random_baskets.items():
                        classifications[method][basket] = set(symbols)
                    console.print(f"[green]Random baskets created: {len(random_baskets)}[/green]")
                except Exception as e:
                    console.print(f"[bold red]Random basket creation failed: {e}[/bold red]")
        
        return classifications
    
    def _calculate_validation_periods(self, total_days: int, period_length: int) -> List[Dict[str, Any]]:
        """
        Calculate validation periods
        
        Args:
            total_days: Total number of days to analyze
            period_length: Length of each period in days
            
        Returns:
            List[Dict]: List of validation periods
        """
        # Calculate periods
        now = datetime.now()
        periods = []
        
        for i in range(0, total_days, period_length):
            if i + period_length > total_days:
                break
                
            end_date = now - timedelta(days=i)
            start_date = end_date - timedelta(days=period_length)
            
            periods.append({
                'period_id': len(periods) + 1,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
        
        console.print(f"[green]Created {len(periods)} validation periods of {period_length} days each[/green]")
        
        return periods
    
    def _analyze_classification_across_periods(self,
                                             symbols: List[str],
                                             periods: List[Dict[str, Any]],
                                             timeframe: str,
                                             eth_event_params: Dict[str, Any],
                                             progress: Progress,
                                             analysis_task: int) -> Dict[str, Any]:
        """
        Analyze a specific classification across all validation periods
        
        Args:
            symbols: List of symbols in this classification
            periods: List of validation periods
            timeframe: Data timeframe
            eth_event_params: Parameters for ETH event detection
            progress: Progress object for tracking
            analysis_task: Task ID for updating progress
            
        Returns:
            Dict: Analysis results for this classification
        """
        results_by_period = {}
        all_abnormal_returns = {}
        
        for period in periods:
            period_id = period['period_id']
            start_date = period['start_date']
            end_date = period['end_date']
            
            progress.update(analysis_task, 
                          description=f"[cyan]Period {period_id}: {start_date} to {end_date}[/cyan]")
            
            # Run hypothesis testing for this period
            period_results = self._analyze_single_period(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                eth_event_params=eth_event_params
            )
            
            results_by_period[period_id] = period_results
            
            # Collect abnormal returns for stability calculation
            if 'abnormal_returns' in period_results:
                for symbol, abnormal_return in period_results['abnormal_returns'].items():
                    if symbol not in all_abnormal_returns:
                        all_abnormal_returns[symbol] = {}
                    
                    all_abnormal_returns[symbol][period_id] = abnormal_return
            
            progress.update(analysis_task, advance=1)
        
        # Compile results
        return {
            'results_by_period': results_by_period,
            'all_abnormal_returns': all_abnormal_returns
        }
    
    def _analyze_single_period(self,
                              symbols: List[str],
                              start_date: str,
                              end_date: str,
                              timeframe: str,
                              eth_event_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single period using hypothesis testing
        
        Args:
            symbols: List of symbols to analyze
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
            
            # 計算開始日期到結束日期的天數
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
                return {'error': 'No ETH data available for this period'}
            
            # Identify ETH drop events
            events = self.hypothesis_testing.identify_eth_drop_events(
                eth_data=eth_data,
                drop_threshold=eth_event_params.get('drop_threshold', -0.01),
                window_size=config.DETECTION_WINDOW_SIZE,
                consecutive_drops=eth_event_params.get('consecutive_drops', 1),
                volume_factor=eth_event_params.get('volume_factor', 1.5)
            )
            
            if not events:
                return {
                    'eth_events': [],
                    'abnormal_returns': {},
                    'summary': {'total_events': 0, 'significant_coins': 0}
                }
            
            # Fetch data for all symbols
            coin_data = {}
            for symbol in symbols:
                # 检查symbol是否已经包含USDT后缀
                full_symbol = symbol if symbol.endswith('USDT') else f"{symbol}USDT"
                try:
                    data = self.data_fetcher.fetch_historical_data(
                        symbol=full_symbol,
                        interval=timeframe,
                        days=days_diff,
                        end_date=end_date
                    )
                    
                    if not data.empty:
                        coin_data[symbol] = data
                except Exception as e:
                    self.logger.warning(f"Error fetching data for {symbol}: {e}")
            
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
                    if len(pre_returns[symbol]) > 0 and len(post_returns[symbol]) > 0:
                        abnormal_returns[symbol] = np.mean(post_returns[symbol]) - np.mean(pre_returns[symbol])
            
            # Return period results
            return {
                'eth_events': [{'time': e['event_time'].isoformat(), 'drop_pct': e['drop_pct']} for e in events],
                'test_results': test_results.to_dict('records') if not test_results.empty else [],
                'abnormal_returns': abnormal_returns,
                'summary': {
                    'total_events': len(events),
                    'coins_analyzed': len(abnormal_returns),
                    'significant_coins': len(test_results[test_results['is_significant'] == True]) if not test_results.empty else 0
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_stability_across_periods(self,
                                           analysis_results: Dict[str, Dict[str, Dict]],
                                           periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate stability metrics for coins across multiple periods
        
        Args:
            analysis_results: Results for each classification and period
            periods: List of validation periods
            
        Returns:
            Dict: Stability analysis results
        """
        console.print("[bold blue]Calculating stability metrics across periods[/bold blue]")
        
        stability_results = {}
        
        for method, tiers in analysis_results.items():
            stability_results[method] = {}
            
            for tier_name, tier_results in tiers.items():
                # Extract abnormal returns data
                all_abnormal_returns = tier_results.get('all_abnormal_returns', {})
                
                if not all_abnormal_returns:
                    stability_results[method][tier_name] = []
                    continue
                
                # Calculate stability metrics for each coin
                stability_data = []
                
                for symbol, period_returns in all_abnormal_returns.items():
                    # Skip if not enough data
                    if len(period_returns) < 2:
                        continue
                        
                    # Calculate stability metrics
                    periods_present = len(period_returns)
                    period_coverage = periods_present / len(periods)
                    
                    # Calculate mean and standard deviation of abnormal returns
                    returns_values = list(period_returns.values())
                    mean_return = np.mean(returns_values)
                    std_return = np.std(returns_values)
                    
                    # Calculate direction stability
                    positive_periods = sum(1 for r in returns_values if r > 0)
                    negative_periods = periods_present - positive_returns
                    direction_stability = max(positive_periods, negative_periods) / periods_present
                    
                    # Calculate magnitude stability (coefficient of variation)
                    abs_returns = [abs(r) for r in returns_values]
                    if np.mean(abs_returns) > 0:
                        magnitude_stability = 1 - np.std(abs_returns) / np.mean(abs_returns)
                    else:
                        magnitude_stability = 0
                        
                    # Calculate weighted stability score
                    weights = {
                        'direction': 0.4,
                        'magnitude': 0.3,
                        'coverage': 0.3
                    }
                    
                    weighted_score = (
                        weights['direction'] * direction_stability +
                        weights['magnitude'] * magnitude_stability +
                        weights['coverage'] * period_coverage
                    )
                    
                    # Store stability metrics
                    stability_data.append({
                        'symbol': symbol,
                        'periods_present': periods_present,
                        'period_coverage': period_coverage,
                        'mean_abnormal_return': mean_return,
                        'std_abnormal_return': std_return,
                        'direction_stability': direction_stability,
                        'magnitude_stability': magnitude_stability,
                        'weighted_score': weighted_score
                    })
                
                # Sort by weighted score
                stability_data.sort(key=lambda x: x['weighted_score'], reverse=True)
                
                # Store in results
                stability_results[method][tier_name] = stability_data
        
        return stability_results
    
    def _generate_summary_report(self,
                                analysis_results: Dict,
                                stability_results: Dict,
                                validation_periods: List[Dict],
                                eth_event_params: Dict,
                                output_dir: str) -> None:
        """
        Generate a comprehensive summary report
        
        Args:
            analysis_results: Results for each classification and period
            stability_results: Stability analysis results
            validation_periods: List of validation periods
            eth_event_params: Parameters for ETH event detection
            output_dir: Output directory for reports
        """
        console.print("[bold blue]Generating comprehensive summary report[/bold blue]")
        
        # Create report dictionary
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'validation_periods': len(validation_periods),
                'period_length': (datetime.strptime(validation_periods[0]['end_date'], '%Y-%m-%d') - 
                                 datetime.strptime(validation_periods[0]['start_date'], '%Y-%m-%d')).days,
                'eth_event_params': eth_event_params
            },
            'top_stable_coins_by_classification': {}
        }
        
        # Extract top stable coins for each classification
        for method, tiers in stability_results.items():
            report['top_stable_coins_by_classification'][method] = {}
            
            for tier_name, stability_data in tiers.items():
                # Get top 10 coins
                top_10 = stability_data[:10]
                
                report['top_stable_coins_by_classification'][method][tier_name] = [
                    {
                        'symbol': coin['symbol'],
                        'stability_score': coin['weighted_score'],
                        'mean_abnormal_return': coin['mean_abnormal_return'],
                        'period_coverage': coin['period_coverage']
                    } for coin in top_10
                ]
        
        # Write report to file
        with open(os.path.join(output_dir, "summary_report.json"), "w") as f:
            json.dump(report, f, indent=4)
        
        # Generate tables for console display
        console.print("\n[bold green]Top Stable Coins by Classification[/bold green]")
        
        for method, tiers in report['top_stable_coins_by_classification'].items():
            for tier_name, coins in tiers.items():
                if not coins:
                    continue
                    
                console.print(f"\n[bold blue]{method.upper()} - {tier_name}[/bold blue]")
                
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Rank")
                table.add_column("Symbol")
                table.add_column("Stability Score")
                table.add_column("Mean Abnormal Return")
                table.add_column("Period Coverage")
                
                for i, coin in enumerate(coins):
                    table.add_row(
                        str(i+1),
                        coin['symbol'],
                        f"{coin['stability_score']:.4f}",
                        f"{coin['mean_abnormal_return']:.4f}",
                        f"{coin['period_coverage']:.2f}"
                    )
                
                console.print(table)
        
        # Generate visualizations
        self._generate_visualizations(stability_results, output_dir)
    
    def _generate_visualizations(self, stability_results: Dict, output_dir: str) -> None:
        """
        Generate visualizations for stability analysis
        
        Args:
            stability_results: Stability analysis results
            output_dir: Output directory for visualizations
        """
        try:
            for method, tiers in stability_results.items():
                for tier_name, stability_data in tiers.items():
                    if not stability_data:
                        continue
                        
                    # Convert to DataFrame for easier visualization
                    df = pd.DataFrame(stability_data)
                    
                    # Create visualization directory
                    viz_dir = os.path.join(output_dir, f"visualizations/{method}/{tier_name}")
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    # 1. Top coins by stability score
                    plt.figure(figsize=(10, 6))
                    top_n = min(20, len(df))
                    sns.barplot(
                        data=df.head(top_n),
                        x='weighted_score',
                        y='symbol',
                        palette='viridis'
                    )
                    plt.title(f'Top Stable Coins by Score - {method} - {tier_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'top_stable_coins.png'))
                    plt.close()
                    
                    # 2. Scatter plot of stability vs. abnormal returns
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(
                        data=df,
                        x='mean_abnormal_return',
                        y='weighted_score',
                        size='period_coverage', 
                        sizes=(20, 200),
                        alpha=0.7
                    )
                    
                    # Annotate top coins
                    for i, row in df.head(10).iterrows():
                        plt.annotate(
                            row['symbol'],
                            (row['mean_abnormal_return'], row['weighted_score']),
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
                    
                    plt.title(f'Stability vs Return - {method} - {tier_name}')
                    plt.xlabel('Mean Abnormal Return')
                    plt.ylabel('Stability Score')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'stability_vs_return.png'))
                    plt.close()
                    
                    # 3. Distribution of stability scores
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df['weighted_score'], kde=True)
                    plt.title(f'Distribution of Stability Scores - {method} - {tier_name}')
                    plt.xlabel('Stability Score')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'stability_distribution.png'))
                    plt.close()
        
        except Exception as e:
            console.print(f"[bold red]Error generating visualizations: {e}[/bold red]") 