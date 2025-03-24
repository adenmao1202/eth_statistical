#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hypothesis Testing Module
import from hypothesis_testing file --> hypothesis_testing.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Try multiple import methods for statistics-related modules
try:
    from scipy.stats.multitest import multipletests
except ImportError:
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None
        print("Warning: Could not import multipletests function, multiple testing correction will not be performed")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_fetcher import DataFetcher
from utils import calculate_abnormal_returns, calculate_effect_size


class HypothesisTesting:
    """Statistical hypothesis testing for altcoin returns before/after ETH price drops"""
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the hypothesis testing analyzer
        
        Args:
            data_fetcher (DataFetcher): Data fetcher instance
        """
        self.data_fetcher = data_fetcher
        self.reference_symbol = config.DEFAULT_REFERENCE_SYMBOL
        
        # Analysis results
        self.event_periods = []
        self.pre_event_returns = {}
        self.post_event_returns = {}
        self.test_results = pd.DataFrame()
        
        # Parameters
        self.pre_event_window = config.PRE_EVENT_WINDOW
        self.post_event_window = config.POST_EVENT_WINDOW
        self.significance_level = config.SIGNIFICANCE_LEVEL
        
        # Output directory
        self.output_dir = "results/hypothesis_testing/default"
    
    def identify_eth_drop_events(self, eth_data: pd.DataFrame, 
                                drop_threshold: float = config.MIN_DROP_PCT,
                                window_size: int = config.DETECTION_WINDOW_SIZE,
                                consecutive_drops: int = 1,  # Added parameter: consecutive K-line drops
                                volume_factor: float = 1.5) -> List[Dict[str, Any]]:
        """
        Identify significant ETH price drop events, considering volume and continuity
        
        Args:
            eth_data: ETH price data
            drop_threshold: Threshold to identify significant drops (negative value)
            window_size: Size of window for calculating price changes
            consecutive_drops: Number of consecutive K-line drops required
            volume_factor: Volume multiple threshold (relative to average volume)
            
        Returns:
            List[Dict[str, Any]]: List of event periods, including start/end time and indicators
        """
        if 'close' not in eth_data.columns or 'volume' not in eth_data.columns:
            raise ValueError("Data must contain 'close' and 'volume' columns")
        
        print(f"Analyzing {len(eth_data)} ETH price data points for drop events")
        print(f"Date range: {eth_data.index.min()} to {eth_data.index.max()}")
        print(f"Drop threshold: {drop_threshold}, window size: {window_size}, consecutive drops: {consecutive_drops}, volume multiple: {volume_factor}")
        
        # Calculate percentage change within window
        eth_data['pct_change'] = eth_data['close'].pct_change(window_size)
        
        # Calculate moving average volume (20 periods)
        eth_data['volume_ma'] = eth_data['volume'].rolling(20).mean()
        
        # Calculate volume ratio
        eth_data['volume_ratio'] = eth_data['volume'] / eth_data['volume_ma']
        
        # Identify consecutive K-line drops
        eth_data['is_drop'] = eth_data['close'].pct_change() < 0
        eth_data['consecutive_drops'] = eth_data['is_drop'].rolling(consecutive_drops).sum()
        
        # Identify potential event periods
        event_periods = []
        
        for idx, row in eth_data.iterrows():
            if pd.isna(row['pct_change']) or pd.isna(row['volume_ratio']) or pd.isna(row['consecutive_drops']):
                continue
                
            # Check if all conditions are met: significant drop, volume amplification, consecutive drops
            if (row['pct_change'] <= drop_threshold and 
                row['volume_ratio'] >= volume_factor and 
                row['consecutive_drops'] >= consecutive_drops):
                
                # Find drop event
                event_time = idx
                
                # Define event window boundaries
                pre_event_start = event_time - timedelta(minutes=self.pre_event_window)
                post_event_end = event_time + timedelta(minutes=self.post_event_window)
                
                # Calculate event indicators
                event_data = eth_data[(eth_data.index >= pre_event_start) & (eth_data.index <= post_event_end)]
                if not event_data.empty:
                    pre_event_price = eth_data.loc[pre_event_start:event_time]['close'].iloc[0]
                    event_price = eth_data.loc[event_time]['close']
                    post_event_price = eth_data.loc[event_time:post_event_end]['close'].iloc[-1]
                    drop_pct = row['pct_change'] * 100  # Convert to percentage
                    
                    # Add to event list
                    event_periods.append({
                        'event_time': event_time,
                        'pre_event_start': pre_event_start,
                        'post_event_end': post_event_end,
                        'pre_event_price': pre_event_price,
                        'event_price': event_price,
                        'post_event_price': post_event_price,
                        'drop_pct': drop_pct,
                        'volume_ratio': row['volume_ratio'],
                        'consecutive_drops': row['consecutive_drops']
                    })
        
        # Filter overlapping events
        filtered_events = self._filter_overlapping_events(event_periods)
        
        print(f"Identified {len(filtered_events)} ETH drop events")
        
        # Sort by drop percentage (most severe first)
        filtered_events.sort(key=lambda x: x['drop_pct'])
        
        return filtered_events
    
    def _filter_overlapping_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out overlapping events, keeping the most severe drops
        
        Args:
            events (List[Dict[str, Any]]): List of event periods
            
        Returns:
            List[Dict[str, Any]]: Filtered list of events
        """
        if not events:
            return []
            
        # Sort events by drop percentage (ascending, most severe drops first)
        sorted_events = sorted(events, key=lambda x: x['drop_pct'])
        
        filtered_events = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            overlapping = False
            for filtered_event in filtered_events:
                # Check if this event overlaps with any already filtered event
                if ((event['pre_event_start'] <= filtered_event['post_event_end'] and 
                     event['post_event_end'] >= filtered_event['pre_event_start']) or
                    (filtered_event['pre_event_start'] <= event['post_event_end'] and 
                     filtered_event['post_event_end'] >= event['pre_event_start'])):
                    overlapping = True
                    break
            
            if not overlapping:
                filtered_events.append(event)
        
        return filtered_events
    
    def calculate_returns(self, coin_data: Dict[str, pd.DataFrame], 
                         event_periods: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Calculate pre-event and post-event returns for all coins
        
        Args:
            coin_data (Dict[str, pd.DataFrame]): Dictionary of coin data
            event_periods (List[Dict[str, Any]]): List of event periods
            
        Returns:
            Tuple[Dict[str, List[float]], Dict[str, List[float]]]: Pre-event and post-event returns
        """
        pre_event_returns = {}
        post_event_returns = {}
        
        for symbol, df in coin_data.items():
            pre_event_returns[symbol] = []
            post_event_returns[symbol] = []
            
            for event in event_periods:
                event_time = event['event_time']
                pre_start = event['pre_event_start']
                post_end = event['post_event_end']
                
                # Get data for pre-event and post-event windows
                pre_event_data = df[(df.index >= pre_start) & (df.index <= event_time)]
                post_event_data = df[(df.index >= event_time) & (df.index <= post_end)]
                
                # Calculate returns if enough data is available
                if len(pre_event_data) >= 2 and len(post_event_data) >= 2:
                    # Pre-event return
                    pre_return = (pre_event_data['close'].iloc[-1] / pre_event_data['close'].iloc[0] - 1) * 100
                    pre_event_returns[symbol].append(pre_return)
                    
                    # Post-event return
                    post_return = (post_event_data['close'].iloc[-1] / post_event_data['close'].iloc[0] - 1) * 100
                    post_event_returns[symbol].append(post_return)
        
        return pre_event_returns, post_event_returns
    
    def conduct_statistical_tests(self, pre_event_returns: Dict[str, List[float]], 
                                post_event_returns: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Conduct statistical tests on pre and post event returns
        
        Args:
            pre_event_returns (Dict[str, List[float]]): Pre-event returns
            post_event_returns (Dict[str, List[float]]): Post-event returns
            
        Returns:
            pd.DataFrame: DataFrame with test results
        """
        results = []
        
        for symbol in pre_event_returns.keys():
            if symbol not in post_event_returns:
                continue
                
            pre_returns = pre_event_returns[symbol]
            post_returns = post_event_returns[symbol]
            
            # Minimum sample size requirement
            min_sample_size = 5
            if len(pre_returns) < min_sample_size or len(post_returns) < min_sample_size:
                continue
            
            # Outlier handling - use percentile truncation for extreme values
            lower_percentile = 1
            upper_percentile = 99
            
            pre_lower = np.percentile(pre_returns, lower_percentile)
            pre_upper = np.percentile(pre_returns, upper_percentile)
            post_lower = np.percentile(post_returns, lower_percentile)
            post_upper = np.percentile(post_returns, upper_percentile)
            
            pre_returns_filtered = [x for x in pre_returns if pre_lower <= x <= pre_upper]
            post_returns_filtered = [x for x in post_returns if post_lower <= x <= post_upper]
            
            # At least need a certain number of valid observations
            if len(pre_returns_filtered) < min_sample_size or len(post_returns_filtered) < min_sample_size:
                continue
                
            # Calculate basic statistics
            pre_mean = np.mean(pre_returns_filtered)
            post_mean = np.mean(post_returns_filtered)
            pre_std = np.std(pre_returns_filtered)
            post_std = np.std(post_returns_filtered)
            mean_diff = post_mean - pre_mean
            
            # Ensure two arrays have the same length to prevent broadcasting error
            min_length = min(len(pre_returns_filtered), len(post_returns_filtered))
            length_diff = abs(len(pre_returns_filtered) - len(post_returns_filtered))
            
            # If length difference is too large, use independent sample t-test
            if length_diff > min_length * 0.3:  # If difference exceeds 30%
                try:
                    t_stat, p_value = stats.ttest_ind(post_returns_filtered, pre_returns_filtered, equal_var=False)
                    test_type = "Independent t-test (unequal length)"
                except Exception as e:
                    print(f"Independent t-test failed for {symbol}: {e}")
                    t_stat, p_value = np.nan, np.nan
                    test_type = "Failed"
            else:
                # For paired samples, ensure lengths are the same
                pre_returns_cut = np.array(pre_returns_filtered[:min_length])
                post_returns_cut = np.array(post_returns_filtered[:min_length])
                
                # Paired t-test
                try:
                    t_stat, p_value = stats.ttest_rel(post_returns_cut, pre_returns_cut)
                    test_type = "Paired t-test"
                except Exception as e:
                    print(f"Paired t-test failed for {symbol}: {e}")
                    # Fallback to independent sample t-test
                    try:
                        t_stat, p_value = stats.ttest_ind(post_returns_filtered, pre_returns_filtered, equal_var=False)
                        test_type = "Independent t-test (fallback)"
                    except Exception as e2:
                        print(f"Independent t-test also failed for {symbol}: {e2}")
                        t_stat, p_value = np.nan, np.nan
                        test_type = "Failed"
            
            # Calculate effect size (Cohen's d)
            # For paired samples, use standardized average difference
            pooled_std = np.sqrt((pre_std**2 + post_std**2) / 2)
            if pooled_std != 0:  # Avoid division by zero
                effect_size, effect_interpretation = calculate_effect_size(mean_diff, pooled_std)
            else:
                effect_size, effect_interpretation = 0, "Cannot calculate"
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                # For Wilcoxon test also ensure lengths are the same
                pre_returns_cut = np.array(pre_returns_filtered[:min_length])
                post_returns_cut = np.array(post_returns_filtered[:min_length])
                w_stat, w_p_value = stats.wilcoxon(post_returns_cut, pre_returns_cut)
            except Exception as e:
                print(f"Wilcoxon test failed for {symbol}: {e}")
                w_stat, w_p_value = np.nan, np.nan
            
            # Kolmogorov-Smirnov test for distribution difference
            try:
                ks_stat, ks_p_value = stats.ks_2samp(post_returns_filtered, pre_returns_filtered)
            except Exception as e:
                print(f"KS test failed for {symbol}: {e}")
                ks_stat, ks_p_value = np.nan, np.nan
            
            # Store results
            results.append({
                'symbol': symbol,
                'sample_size': min(len(pre_returns), len(post_returns)),
                'pre_mean': pre_mean,
                'post_mean': post_mean,
                'pre_std': pre_std,
                'post_std': post_std,
                'mean_diff': mean_diff,
                't_stat': t_stat,
                'p_value': p_value,
                'test_type': test_type,
                'effect_size': effect_size,
                'effect_interpretation': effect_interpretation,
                'significant': p_value < self.significance_level,
                'wilcoxon_stat': w_stat,
                'wilcoxon_p': w_p_value,
                'ks_stat': ks_stat,
                'ks_p': ks_p_value
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Apply multiple testing correction (Benjamini-Hochberg)
            if multipletests is not None:
                _, corrected_p_values, _, _ = multipletests(results_df['p_value'], alpha=self.significance_level, method='fdr_bh')
                results_df['corrected_p_value'] = corrected_p_values
                results_df['significant_corrected'] = results_df['corrected_p_value'] < self.significance_level
            else:
                # If multipletests is not available, use original p-values
                results_df['corrected_p_value'] = results_df['p_value']
                results_df['significant_corrected'] = results_df['significant']
            
            # Sort by absolute mean difference
            results_df['abs_mean_diff'] = results_df['mean_diff'].abs()
            results_df.sort_values('abs_mean_diff', ascending=False, inplace=True)
            results_df.drop('abs_mean_diff', axis=1, inplace=True)
        
        return results_df
    
    def visualize_results(self, results_df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Visualize hypothesis testing results
        
        Args:
            results_df (pd.DataFrame): DataFrame with test results
            top_n (int): Number of top results to display
        """
        if results_df.empty:
            print("No data to visualize")
            return
            
        # Get top coins by absolute mean difference
        results_df['abs_mean_diff'] = results_df['mean_diff'].abs()
        top_results = results_df.sort_values('abs_mean_diff', ascending=False).head(top_n)
        results_df.drop('abs_mean_diff', axis=1, inplace=True)
        
        # Create bar plot
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_results['symbol'], top_results['mean_diff'], 
                     color=[('green' if x else 'gray') for x in top_results['significant_corrected']])
        
        # Add red line at y=0
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Symbol')
        plt.ylabel('Mean Return Difference (Post - Pre) %')
        plt.title(f'Top {top_n} Coins by Return Difference After ETH Price Drops')
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color='green'), plt.Rectangle((0,0),1,1, color='gray')]
        labels = ['Statistically Significant', 'Not Significant']
        plt.legend(handles, labels)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/return_difference_top.png', dpi=300)
        plt.close()
        
        # Create scatter plot of pre vs post returns
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='pre_mean', y='post_mean', hue='significant_corrected', 
                       size='sample_size', alpha=0.7, data=results_df)
        
        # Add diagonal line (y=x)
        min_val = min(results_df['pre_mean'].min(), results_df['post_mean'].min())
        max_val = max(results_df['pre_mean'].max(), results_df['post_mean'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # Add labels for selected coins
        for _, row in top_results.head(10).iterrows():
            plt.annotate(row['symbol'], (row['pre_mean'], row['post_mean']), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.xlabel('Pre-Event Mean Return (%)')
        plt.ylabel('Post-Event Mean Return (%)')
        plt.title('Pre-Event vs Post-Event Mean Returns')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pre_post_returns_scatter.png', dpi=300)
        plt.close()
    
    def create_distribution_plots(self, pre_event_returns: Dict[str, List[float]], 
                                post_event_returns: Dict[str, List[float]], 
                                symbols: List[str]) -> None:
        """
        Create distribution plots comparing pre and post event returns
        
        Args:
            pre_event_returns (Dict[str, List[float]]): Pre-event returns for each coin
            post_event_returns (Dict[str, List[float]]): Post-event returns for each coin
            symbols (List[str]): List of symbols to plot
        """
        for symbol in symbols:
            if symbol not in pre_event_returns or symbol not in post_event_returns:
                continue
                
            pre_returns = pre_event_returns[symbol]
            post_returns = post_event_returns[symbol]
            
            if len(pre_returns) < 5 or len(post_returns) < 5:
                continue
                
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot distributions
            sns.histplot(pre_returns, kde=True, ax=ax1, color='blue', label='Pre-Event')
            sns.histplot(post_returns, kde=True, ax=ax2, color='red', label='Post-Event')
            
            # Add statistics to the plot
            pre_mean = np.mean(pre_returns)
            post_mean = np.mean(post_returns)
            pre_std = np.std(pre_returns)
            post_std = np.std(post_returns)
            
            ax1.axvline(pre_mean, color='black', linestyle='--')
            ax2.axvline(post_mean, color='black', linestyle='--')
            
            ax1.set_title(f'Pre-Event Returns for {symbol}\nMean: {pre_mean:.2f}%, Std: {pre_std:.2f}%')
            ax2.set_title(f'Post-Event Returns for {symbol}\nMean: {post_mean:.2f}%, Std: {post_std:.2f}%')
            
            ax1.set_xlabel('Return (%)')
            ax2.set_xlabel('Return (%)')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/distribution_{symbol}.png', dpi=300)
            plt.close()
    
    def summarize_results(self, results_df: pd.DataFrame, rebound_df: pd.DataFrame, strategy_df: pd.DataFrame) -> None:
        """
        Generate result summary
        
        Args:
            results_df: Hypothesis testing results
            rebound_df: Rebound time analysis results
            strategy_df: Trading strategy evaluation results
        """
        summary_file = f"{self.output_dir}/analysis_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("===== Crypto Market Analysis Summary =====\n\n")
            
            # Hypothesis testing summary
            f.write("1. Return Difference Analysis\n")
            f.write("-----------------------\n")
            
            if not results_df.empty:
                significant_results = results_df[results_df['significant_corrected']]
                positive_sig = significant_results[significant_results['mean_diff'] > 0]
                negative_sig = significant_results[significant_results['mean_diff'] < 0]
                
                f.write(f"Analyzed {len(results_df)} cryptocurrencies\n")
                f.write(f"Statistically significant results: {len(significant_results)} ({len(significant_results)/len(results_df)*100:.1f}%)\n")
                f.write(f"Positive significant difference: {len(positive_sig)} ({len(positive_sig)/len(significant_results)*100:.1f}% of significant results)\n")
                f.write(f"Negative significant difference: {len(negative_sig)} ({len(negative_sig)/len(significant_results)*100:.1f}% of significant results)\n\n")
                
                f.write("Top 5 cryptocurrencies with largest return difference:\n")
                for _, row in results_df.sort_values('mean_diff', ascending=False).head(5).iterrows():
                    f.write(f"  {row['symbol']}: {row['mean_diff']:.2f}% (p={row['corrected_p_value']:.4f})\n")
                
                f.write("\nTop 5 cryptocurrencies with smallest return difference:\n")
                for _, row in results_df.sort_values('mean_diff').head(5).iterrows():
                    f.write(f"  {row['symbol']}: {row['mean_diff']:.2f}% (p={row['corrected_p_value']:.4f})\n")
            else:
                f.write("No usable hypothesis testing results\n")
            
            # Rebound time analysis summary
            f.write("\n\n2. Rebound Time Analysis\n")
            f.write("-----------------------\n")
            
            if not rebound_df.empty:
                faster_coins = rebound_df[rebound_df['faster_pct'] > 50]
                
                f.write(f"Analyzed {len(rebound_df)} cryptocurrencies' rebound time\n")
                f.write(f"Coins that rebound earlier than ETH: {len(faster_coins)} ({len(faster_coins)/len(rebound_df)*100:.1f}%)\n\n")
                
                f.write("Top 5 cryptocurrencies that rebound fastest:\n")
                for symbol, row in rebound_df.head(5).iterrows():
                    f.write(f"  {symbol}: Earlier rebound percentage compared to ETH {row['faster_pct']:.1f}%, Average time difference {row['avg_time_diff']:.1f} minutes\n")
            else:
                f.write("No usable rebound time analysis results\n")
            
            # Trading strategy evaluation summary
            f.write("\n\n3. Trading Strategy Evaluation\n")
            f.write("-----------------------\n")
            
            if not strategy_df.empty:
                profitable_coins = strategy_df[strategy_df['total_profit'] > 0]
                
                f.write(f"Evaluated {len(strategy_df)} cryptocurrencies' trading strategies\n")
                f.write(f"Profitable coins: {len(profitable_coins)} ({len(profitable_coins)/len(strategy_df)*100:.1f}%)\n")
                
                if not profitable_coins.empty:
                    avg_win_rate = profitable_coins['win_rate'].mean()
                    avg_profit = profitable_coins['avg_profit'].mean()
                    f.write(f"Average win rate of profitable coins: {avg_win_rate:.1f}%\n")
                    f.write(f"Average profit of profitable coins: {avg_profit:.2f}%\n\n")
                
                f.write("Top 5 cryptocurrencies with best strategy performance:\n")
                for symbol, row in strategy_df.head(5).iterrows():
                    f.write(f"  {symbol}: Total profit {row['total_profit']:.1f}%, Win rate {row['win_rate']:.1f}%, Average holding time {row['avg_holding']:.1f} minutes\n")
            else:
                f.write("No usable trading strategy evaluation results\n")
            
            # Comprehensive advice
            f.write("\n\n4. Comprehensive Advice\n")
            f.write("-----------------------\n")
            
            if not results_df.empty and not rebound_df.empty and not strategy_df.empty:
                # Find coins that meet all three conditions:
                # 1. Significant positive return difference
                # 2. Rebound earlier than ETH
                # 3. Trading strategy profitable
                
                significant_symbols = set(results_df[results_df['significant_corrected'] & (results_df['mean_diff'] > 0)]['symbol'])
                faster_symbols = set(rebound_df[rebound_df['faster_pct'] > 50].index)
                profitable_symbols = set(strategy_df[strategy_df['total_profit'] > 0].index)
                
                recommended_symbols = significant_symbols.intersection(faster_symbols).intersection(profitable_symbols)
                
                if recommended_symbols:
                    f.write("Comprehensive analysis results, the following coins performed best after ETH drop, recommended to consider first:\n")
                    
                    # Sort by strategy total profit
                    recommended_df = strategy_df.loc[list(recommended_symbols)].sort_values('total_profit', ascending=False)
                    
                    for symbol, row in recommended_df.iterrows():
                        result_row = results_df[results_df['symbol'] == symbol].iloc[0]
                        rebound_row = rebound_df.loc[symbol]
                        
                        f.write(f"  {symbol}:\n")
                        f.write(f"    - Return difference: +{result_row['mean_diff']:.2f}% (p={result_row['corrected_p_value']:.4f})\n")
                        f.write(f"    - Rebound earlier than ETH: {rebound_row['faster_pct']:.1f}% in events\n")
                        f.write(f"    - Average rebound time difference: {rebound_row['avg_time_diff']:.1f} minutes\n")
                        f.write(f"    - Trading strategy win rate: {row['win_rate']:.1f}%\n")
                        f.write(f"    - Trading strategy total profit: {row['total_profit']:.1f}%\n")
                        f.write(f"    - Average holding time: {row['avg_holding']:.1f} minutes\n\n")
                else:
                    f.write("No coins that meet all conditions, recommended:\n")
                    f.write("1. Consider loosening selection criteria\n")
                    f.write("2. Increase historical data range for analysis\n")
                    f.write("3. Adjust ETH drop threshold or event window size\n")
            else:
                f.write("Insufficient data, cannot provide comprehensive advice\n")
        
        print(f"Results summary saved to {summary_file}")
    
    def run_hypothesis_testing(self, days: int = 30, timeframe: str = '1m', 
                             drop_threshold: float = config.MIN_DROP_PCT, 
                             window_size: int = config.DETECTION_WINDOW_SIZE,
                             top_n: int = config.DEFAULT_TOP_N, 
                             end_date: Optional[str] = None,
                             use_cache: bool = True,
                             pre_event_window: int = None,
                             post_event_window: int = None,
                             consecutive_drops: int = 1,
                             volume_factor: float = 1.5,
                             rebound_threshold: float = 0.005,
                             take_profit_pct: float = 0.03,
                             stop_loss_pct: float = 0.02,
                             progress_callback: Optional[Callable] = None,
                             output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run hypothesis testing analysis and trading strategy evaluation

        Args:
            days: Analysis days
            timeframe: Time frame ('1m', '5m', '15m', '1h', '4h')
            drop_threshold: Threshold to identify significant ETH price drop (negative value, e.g., -0.01 for 1% drop)
            window_size: Detection window size
            top_n: Analyze top N coins
            end_date: End date, format 'YYYY-MM-DD'
            use_cache: Whether to use cached data
            pre_event_window: Event pre-window (minutes)
            post_event_window: Event post-window (minutes)
            consecutive_drops: Number of consecutive K-line drops
            volume_factor: Volume multiple threshold
            rebound_threshold: Define rebound threshold
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            progress_callback: Progress callback function
            output_dir: Output directory

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                - Hypothesis testing results
                - Rebound time analysis results
                - Trading strategy evaluation results
        """
        # Update event window size (if provided)
        if pre_event_window is not None:
            self.pre_event_window = pre_event_window
        if post_event_window is not None:
            self.post_event_window = post_event_window
            
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
            
        print(f"Running hypothesis testing analysis, time range: {days} days, time frame: {timeframe}")
        
        # Calculate date
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Add extra days to allow event window
        extra_days = max(2, (self.pre_event_window + self.post_event_window) // (60 * 24) + 1)
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days+extra_days)).strftime('%Y-%m-%d')
        
        print(f"Analysis period: {start_date} to {end_date}")
        print(f"Event pre-window: {self.pre_event_window} minutes, Event post-window: {self.post_event_window} minutes")
        
        # Get ETH data
        if progress_callback:
            progress_callback(0.10, "Getting ETH historical data...")
            
        eth_data = self.data_fetcher.fetch_historical_data(
            symbol=self.reference_symbol, 
            interval=timeframe, 
            days=days,
            end_date=end_date,
            use_cache=use_cache
        )
        
        if eth_data.empty:
            print(f"Could not get data for {self.reference_symbol}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Identify ETH drop events
        if progress_callback:
            progress_callback(0.20, "Identifying ETH drop events...")
            
        self.event_periods = self.identify_eth_drop_events(
            eth_data, 
            drop_threshold, 
            window_size,
            consecutive_drops,
            volume_factor
        )
        
        if not self.event_periods:
            print("No significant ETH drop events identified")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Get all coin data
        if progress_callback:
            progress_callback(0.30, "Getting cryptocurrency data...")
            
        coin_data = self.data_fetcher.get_all_data(
            timeframe=timeframe,
            days=days,
            end_date=end_date,
            use_cache=use_cache,
            include_reference=True  # Ensure include ETH
        )
        
        # Calculate returns
        if progress_callback:
            progress_callback(0.40, "Calculating event pre-post returns...")
            
        self.pre_event_returns, self.post_event_returns = self.calculate_returns(coin_data, self.event_periods)
        
        # Conduct statistical tests
        if progress_callback:
            progress_callback(0.50, "Conducting statistical tests...")
            
        results_df = self.conduct_statistical_tests(self.pre_event_returns, self.post_event_returns)
        
        # Analyze rebound time
        if progress_callback:
            progress_callback(0.60, "Analyzing rebound time difference...")
        
        rebound_df = self.analyze_rebound_timing(coin_data, self.event_periods, rebound_threshold)
        
        # Evaluate trading strategy
        if progress_callback:
            progress_callback(0.70, "Evaluating trading strategy...")
        
        strategy_df = self.evaluate_trading_strategy(
            coin_data, 
            self.event_periods, 
            rebound_df, 
            top_n,
            take_profit_pct,
            stop_loss_pct
        )
        
        # Save results
        results_file = f"{self.output_dir}/hypothesis_testing_results.csv"
        results_df.to_csv(results_file)
        
        rebound_file = f"{self.output_dir}/rebound_analysis_results.csv"
        rebound_df.to_csv(rebound_file)
        
        strategy_file = f"{self.output_dir}/strategy_evaluation_results.csv"
        strategy_df.to_csv(strategy_file)
        
        print(f"Results saved to {self.output_dir}")
        
        # Visualize results
        if progress_callback:
            progress_callback(0.80, "Generating visualization charts...")
            
        self.visualize_results(results_df, top_n)
        
        # Generate rebound time and trading strategy visualization
        self.visualize_rebound_analysis(rebound_df, top_n)
        self.visualize_strategy_results(strategy_df, top_n)
        
        # Generate summary
        if progress_callback:
            progress_callback(0.90, "Generating result summary...")
            
        self.summarize_results(results_df, rebound_df, strategy_df)
        
        if progress_callback:
            progress_callback(1.0, "Analysis completed!")
            
        return results_df, rebound_df, strategy_df

    def analyze_rebound_timing(self, coin_data: Dict[str, pd.DataFrame], 
                              event_periods: List[Dict[str, Any]],
                              rebound_threshold: float = 0.005) -> pd.DataFrame:
        """
        Analyze ETH and altcoin rebound time difference after ETH price drops
        
        Args:
            coin_data: Coin data dictionary
            event_periods: List of event periods
            rebound_threshold: Define rebound threshold (e.g., 0.5% for 0.5% rebound)
            
        Returns:
            pd.DataFrame: DataFrame with rebound time analysis results
        """
        rebound_results = []
        
        for event in event_periods:
            event_time = event['event_time']
            post_end = event['post_event_end']
            
            # Get ETH data
            eth_df = coin_data[self.reference_symbol]
            eth_post_event = eth_df[(eth_df.index >= event_time) & (eth_df.index <= post_end)]
            
            if eth_post_event.empty:
                continue
                
            # Get ETH price at event time
            eth_event_price = eth_post_event.iloc[0]['close']
            
            # Calculate ETH rebound time
            eth_rebound_time = None
            for idx, row in eth_post_event.iterrows():
                if row['close'] >= eth_event_price * (1 + rebound_threshold):
                    eth_rebound_time = idx
                    break
            
            # If ETH didn't rebound within observation period, skip this event
            if eth_rebound_time is None:
                continue
            
            # Calculate ETH rebound required minutes
            eth_rebound_minutes = (eth_rebound_time - event_time).total_seconds() / 60
            
            # Analyze each altcoin
            for symbol, df in coin_data.items():
                if symbol == self.reference_symbol:
                    continue
                    
                # Get altcoin event post-data
                coin_post_event = df[(df.index >= event_time) & (df.index <= post_end)]
                
                if coin_post_event.empty:
                    continue
                    
                # Get altcoin price at event time
                coin_event_price = coin_post_event.iloc[0]['close']
                
                # Calculate altcoin rebound time
                coin_rebound_time = None
                for idx, row in coin_post_event.iterrows():
                    if row['close'] >= coin_event_price * (1 + rebound_threshold):
                        coin_rebound_time = idx
                        break
                
                # If altcoin rebounded within observation period
                if coin_rebound_time is not None:
                    # Calculate altcoin rebound required minutes
                    coin_rebound_minutes = (coin_rebound_time - event_time).total_seconds() / 60
                    
                    # Calculate time difference with ETH (negative value means earlier rebound)
                    time_difference = coin_rebound_minutes - eth_rebound_minutes
                    
                    rebound_results.append({
                        'event_time': event_time,
                        'symbol': symbol,
                        'eth_rebound_minutes': eth_rebound_minutes,
                        'coin_rebound_minutes': coin_rebound_minutes,
                        'time_difference': time_difference,
                        'faster_than_eth': time_difference < 0,
                        'drop_pct': event['drop_pct']
                    })
        
        # Convert to DataFrame
        rebound_df = pd.DataFrame(rebound_results)
        
        if not rebound_df.empty:
            # Calculate each coin's percentage of events rebounding earlier than ETH
            rebound_summary = rebound_df.groupby('symbol').agg(
                total_events=('event_time', 'count'),
                faster_events=('faster_than_eth', 'sum'),
                avg_time_diff=('time_difference', 'mean'),
                min_time_diff=('time_difference', 'min'),
                max_time_diff=('time_difference', 'max')
            )
            
            rebound_summary['faster_pct'] = (rebound_summary['faster_events'] / rebound_summary['total_events']) * 100
            
            # Sort by earlier rebound percentage
            rebound_summary = rebound_summary.sort_values('faster_pct', ascending=False)
        else:
            rebound_summary = pd.DataFrame()
        
        return rebound_summary 

    def evaluate_trading_strategy(self, coin_data: Dict[str, pd.DataFrame], 
                                 event_periods: List[Dict[str, Any]],
                                 rebound_summary: pd.DataFrame,
                                 top_n: int = 5,
                                 take_profit_pct: float = 0.03,
                                 stop_loss_pct: float = 0.02) -> pd.DataFrame:
        """
        Evaluate trading strategy based on ETH drop events
        
        Args:
            coin_data: Coin data dictionary
            event_periods: List of event periods
            rebound_summary: Rebound time analysis results
            top_n: Select top N coins that rebound fastest
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            
        Returns:
            pd.DataFrame: Strategy evaluation results
        """
        if rebound_summary.empty:
            return pd.DataFrame()
        
        # Select top N coins that rebound fastest
        top_coins = rebound_summary.head(top_n).index.tolist()
        
        strategy_results = []
        
        for event in event_periods:
            event_time = event['event_time']
            post_end = event['post_event_end']
            
            for symbol in top_coins:
                if symbol not in coin_data:
                    continue
                    
                df = coin_data[symbol]
                post_event_data = df[(df.index >= event_time) & (df.index <= post_end)]
                
                if post_event_data.empty:
                    continue
                    
                # Simulate entering at ETH drop event
                entry_price = post_event_data.iloc[0]['close']
                entry_time = post_event_data.index[0]
                
                # Calculate take profit and stop loss prices
                take_profit_price = entry_price * (1 + take_profit_pct)
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                
                # Simulate trading results
                exit_price = None
                exit_time = None
                profit_pct = None
                hit_target = None
                
                for idx, row in post_event_data.iterrows():
                    if idx == entry_time:
                        continue
                        
                    # Check if reach take profit or stop loss
                    if row['high'] >= take_profit_price:
                        exit_price = take_profit_price
                        exit_time = idx
                        profit_pct = take_profit_pct * 100
                        hit_target = True
                        break
                    elif row['low'] <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = idx
                        profit_pct = -stop_loss_pct * 100
                        hit_target = False
                        break
                
                # If didn't reach take profit or stop loss, use observation period end price
                if exit_price is None:
                    exit_price = post_event_data.iloc[-1]['close']
                    exit_time = post_event_data.index[-1]
                    profit_pct = ((exit_price / entry_price) - 1) * 100
                    hit_target = profit_pct > 0
                
                # Calculate holding time (minutes)
                holding_period = (exit_time - entry_time).total_seconds() / 60
                
                strategy_results.append({
                    'event_time': event_time,
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'holding_period': holding_period,
                    'hit_target': hit_target
                })
        
        # Convert to DataFrame
        strategy_df = pd.DataFrame(strategy_results)
        
        if not strategy_df.empty:
            # Calculate each coin's strategy performance
            strategy_summary = strategy_df.groupby('symbol').agg(
                total_trades=('event_time', 'count'),
                win_trades=('hit_target', 'sum'),
                avg_profit=('profit_pct', 'mean'),
                total_profit=('profit_pct', 'sum'),
                max_profit=('profit_pct', 'max'),
                min_profit=('profit_pct', 'min'),
                avg_holding=('holding_period', 'mean')
            )
            
            strategy_summary['win_rate'] = (strategy_summary['win_trades'] / strategy_summary['total_trades']) * 100
            
            # Sort by total profit
            strategy_summary = strategy_summary.sort_values('total_profit', ascending=False)
        else:
            strategy_summary = pd.DataFrame()
        
        return strategy_summary 

    def visualize_rebound_analysis(self, rebound_df: pd.DataFrame, top_n: int = 10) -> None:
        """
        Visualize rebound time analysis results
        
        Args:
            rebound_df: Rebound time analysis results
            top_n: Display top N results
        """
        if rebound_df.empty:
            print("No data to visualize")
            return
        
        # Get top coins that rebound fastest
        top_results = rebound_df.head(top_n)
        
        # Create bar chart
        plt.figure(figsize=(14, 8))
        
        # Use color to differentiate percentage of earlier rebound compared to ETH
        bars = plt.bar(top_results.index, top_results['faster_pct'], 
                     color=plt.cm.RdYlGn(top_results['faster_pct']/100))
        
        # Add labels and title
        plt.xlabel('Coin')
        plt.ylabel('Percentage of Earlier Rebound Compared to ETH (%)')
        plt.title(f'Top {top_n} Coins that Rebound Faster (Compared to ETH)')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/rebound_analysis_top{top_n}.png", dpi=300)
        plt.close()
        
        # Create average time difference chart
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_results.index, top_results['avg_time_diff'], 
                     color=['green' if x < 0 else 'red' for x in top_results['avg_time_diff']])
        
        plt.xlabel('Coin')
        plt.ylabel('Average Rebound Time Difference (Minutes)')
        plt.title(f'Top {top_n} Coins Average Rebound Time Difference (Compared to ETH, Negative Value Means Earlier Rebound)')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3 if height >= 0 else height - 0.8,
                    f'{height:.1f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/rebound_time_diff_top{top_n}.png", dpi=300)
        plt.close()

    def visualize_strategy_results(self, strategy_df: pd.DataFrame, top_n: int = 10) -> None:
        """
        Visualize trading strategy evaluation results
        
        Args:
            strategy_df: Trading strategy evaluation results
            top_n: Display top N results
        """
        if strategy_df.empty:
            print("No data to visualize")
            return
        
        # Get top coins with best performance
        top_results = strategy_df.head(top_n)
        
        # Create win rate chart
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_results.index, top_results['win_rate'], 
                     color=plt.cm.RdYlGn(top_results['win_rate']/100))
        
        plt.xlabel('Coin')
        plt.ylabel('Win Rate (%)')
        plt.title(f'Top {top_n} Coins Trading Strategy Win Rate')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/strategy_win_rate_top{top_n}.png", dpi=300)
        plt.close()
        
        # Create average profit chart
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_results.index, top_results['avg_profit'], 
                     color=['green' if x > 0 else 'red' for x in top_results['avg_profit']])
        
        plt.xlabel('Coin')
        plt.ylabel('Average Profit (%)')
        plt.title(f'Top {top_n} Coins Trading Strategy Average Profit')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.3,
                    f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/strategy_avg_profit_top{top_n}.png", dpi=300)
        plt.close()
        
        # Create total profit chart
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_results.index, top_results['total_profit'], 
                     color=['green' if x > 0 else 'red' for x in top_results['total_profit']])
        
        plt.xlabel('Coin')
        plt.ylabel('Total Profit (%)')
        plt.title(f'Top {top_n} Coins Trading Strategy Total Profit')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1 if height >= 0 else height - 3,
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/strategy_total_profit_top{top_n}.png", dpi=300)
        plt.close() 