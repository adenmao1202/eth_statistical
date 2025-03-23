#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter Analysis and Optimization Tool
Automatically recommend optimal parameter settings based on ETH historical data
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
from sklearn.model_selection import ParameterGrid

# Ensure main directory is in module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grand_parent_dir)

# Import local modules
try:
    import config
    from data_fetcher import DataFetcher
except ImportError as e:
    print(f"Cannot import necessary modules: {e}")
    sys.exit(1)

class ParamAnalyzer:
    """Parameter analysis tool based on ETH historical data"""
    
    def __init__(self, data_fetcher: Optional[DataFetcher] = None):
        """
        Initialize parameter analyzer
        
        Args:
            data_fetcher: Data fetcher, will create automatically if not provided
        """
        self.data_fetcher = data_fetcher or DataFetcher()
        self.eth_data = None
        self.output_dir = os.path.join(parent_dir, "parameter_suggestions")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fetch_eth_data(self, days: int = 60, timeframe: str = '5m', 
                      end_date: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch ETH historical data
        
        Args:
            days: Analysis period in days
            timeframe: Time frame
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cache
            
        Returns:
            pd.DataFrame: ETH historical data
        """
        symbol = config.DEFAULT_REFERENCE_SYMBOL
        print(f"Fetching {days} days of historical data for {symbol} (timeframe: {timeframe})...")
        
        self.eth_data = self.data_fetcher.fetch_historical_data(
            symbol=symbol,
            interval=timeframe,
            days=days,
            end_date=end_date,
            use_cache=use_cache
        )
        
        print(f"Retrieved {len(self.eth_data)} data points")
        return self.eth_data
    
    def analyze_price_drops(self, eth_data: pd.DataFrame, percentiles: List[int] = [1, 3, 5, 10, 15]) -> Dict[str, Any]:
        """
        Analyze ETH price drop distribution
        
        Args:
            eth_data: ETH price data
            percentiles: Percentiles to analyze
            
        Returns:
            Dict[str, Any]: Price drop analysis results
        """
        print("Analyzing ETH price drop distribution...")
        
        # Calculate price changes for various time windows
        results = {}
        windows = [1, 3, 5, 10, 15, 30, 60]
        
        for window in windows:
            eth_data[f'pct_change_{window}'] = eth_data['close'].pct_change(window)
            
            # Focus on negative values (drops)
            drops = eth_data[f'pct_change_{window}'][eth_data[f'pct_change_{window}'] < 0]
            
            # Calculate percentiles for drop magnitudes
            percentile_values = {}
            for p in percentiles:
                value = np.percentile(drops, p)
                percentile_values[p] = value
            
            results[window] = {
                'count': len(drops),
                'mean': drops.mean(),
                'std': drops.std(),
                'min': drops.min(),
                'percentiles': percentile_values
            }
        
        # Analyze consecutive drop candles statistics
        eth_data['is_drop'] = eth_data['close'].pct_change() < 0
        
        # Calculate distribution of consecutive drop candles
        continuous_drops = []
        current_count = 0
        
        for is_drop in eth_data['is_drop']:
            if pd.isna(is_drop):
                continue
                
            if is_drop:
                current_count += 1
            else:
                if current_count > 0:
                    continuous_drops.append(current_count)
                current_count = 0
                
        if current_count > 0:
            continuous_drops.append(current_count)
            
        # Calculate statistics for consecutive drops
        continuous_stats = {
            'count': len(continuous_drops),
            'mean': np.mean(continuous_drops),
            'median': np.median(continuous_drops),
            'max': np.max(continuous_drops) if continuous_drops else 0,
            'percentiles': {p: np.percentile(continuous_drops, p) for p in [25, 50, 75, 90, 95]} if continuous_drops else {}
        }
        
        results['continuous_drops'] = continuous_stats
        
        # Analyze volume distribution
        eth_data['volume_ratio'] = eth_data['volume'] / eth_data['volume'].rolling(20).mean()
        volume_ratios = eth_data['volume_ratio'].dropna()
        
        volume_stats = {
            'mean': volume_ratios.mean(),
            'median': volume_ratios.median(),
            'percentiles': {p: np.percentile(volume_ratios, p) for p in [50, 75, 90, 95, 99]}
        }
        
        results['volume_stats'] = volume_stats
        
        self._visualize_drop_analysis(results, eth_data)
        
        return results
    
    def _visualize_drop_analysis(self, results: Dict[str, Any], eth_data: pd.DataFrame) -> None:
        """
        Visualize drop analysis results
        
        Args:
            results: Drop analysis results
            eth_data: ETH price data
        """
        # Improved drop distribution visualization - separate charts
        windows = [1, 3, 5, 15, 30]
        percentiles = [1, 5, 10, 25]
        
        # Create subplot for each window instead of overlapping
        fig, axes = plt.subplots(len(windows), 1, figsize=(14, 4*len(windows)))
        fig.suptitle('ETH Price Drop Distribution by Timeframe', fontsize=16)
        
        for i, window in enumerate(windows):
            ax = axes[i]
            drops = eth_data[f'pct_change_{window}'][eth_data[f'pct_change_{window}'] < 0] * 100  # Convert to percentage
            
            # Use more bins for better resolution
            counts, bins, _ = ax.hist(drops, bins=100, alpha=0.7, color=f'C{i}')
            
            # Add percentile markers
            percentile_values = {}
            for p in percentiles:
                percentile_values[p] = np.percentile(drops, p)
                ax.axvline(x=percentile_values[p], color='red', linestyle='--', alpha=0.7)
                ax.text(percentile_values[p], ax.get_ylim()[1]*0.9, f'{p}%: {percentile_values[p]:.2f}%', 
                        rotation=90, verticalalignment='top')
            
            # Calculate and display statistics
            mean_val = drops.mean()
            median_val = drops.median()
            ax.axvline(x=mean_val, color='black', linestyle='-', alpha=0.7)
            ax.text(mean_val, ax.get_ylim()[1]*0.8, f'Mean: {mean_val:.2f}%', 
                    rotation=90, verticalalignment='top')
            
            ax.set_title(f'{window} Period Window Drop Distribution')
            ax.set_xlabel('Drop Percentage (%)')
            ax.set_ylabel('Frequency')
            
            # Use log scale for y-axis to better see the distribution
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Zoom in on the relevant range
            min_x = percentile_values[1] * 1.5 if percentile_values[1] < -5 else -5
            ax.set_xlim([min_x, 0])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(self.output_dir, 'eth_drop_distribution.png'), dpi=300)
        plt.close()
        
        # Generate a quantitative summary table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for the table
        table_data = []
        for window in windows:
            row_data = [f"{window} periods"]
            
            drops = eth_data[f'pct_change_{window}'][eth_data[f'pct_change_{window}'] < 0] * 100
            
            for p in [1, 3, 5, 10, 25, 50]:
                row_data.append(f"{np.percentile(drops, p):.2f}%")
            
            row_data.append(f"{drops.mean():.2f}%")
            row_data.append(f"{drops.std():.2f}%")
            row_data.append(f"{len(drops)}")
            
            table_data.append(row_data)
        
        column_labels = ['Window', '1%', '3%', '5%', '10%', '25%', '50%', 'Mean', 'Std Dev', 'Count']
        
        table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.title('ETH Drop Percentile Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eth_drop_quantitative_summary.png'), dpi=300)
        plt.close()
        
        # Visualize continuous drop candles distribution
        if 'continuous_drops' in results and results['continuous_drops']['count'] > 0:
            continuous_drops = results['continuous_drops']
            continuous_drop_values = []
            
            # Ensure we're working with a list of integers
            if isinstance(continuous_drops, dict) and 'percentiles' in continuous_drops:
                # Need to extract the actual list from the results
                # This is likely a nested list within eth_data or was generated earlier
                current_count = 0  # Initialize current_count before using it
                
                for _, row in eth_data.iterrows():
                    if row.get('is_drop', False) and not pd.isna(row.get('is_drop', None)):
                        current_count += 1
                    elif current_count > 0:
                        continuous_drop_values.append(current_count)
                        current_count = 0
                
                # Add the last count if there was one
                if current_count > 0:
                    continuous_drop_values.append(current_count)
                
                if not continuous_drop_values:  # If still empty, use simple range
                    max_drops = int(continuous_drops.get('max', 10))
                    continuous_drop_values = list(range(1, max_drops+1))
            else:
                # If continuous_drops is not a dict with percentiles, assume it's the actual list
                continuous_drop_values = continuous_drops
            
            plt.figure(figsize=(10, 6))
            
            # Make sure we have data to plot and it's in the right format
            if continuous_drop_values and isinstance(continuous_drop_values, list):
                # Calculate percentages instead of raw counts
                continuous_drop_values = np.array(continuous_drop_values)
                max_drops = max(continuous_drop_values)
                
                # Create histogram
                counts, bins, _ = plt.hist(continuous_drop_values, bins=range(1, max_drops + 2), alpha=0.7, 
                                          color='skyblue', edgecolor='black')
                
                # Add percentage labels above each bar
                total = len(continuous_drop_values)
                for i, count in enumerate(counts):
                    percentage = 100 * count / total
                    plt.text(bins[i] + 0.5, count + (plt.ylim()[1] * 0.01), f'{percentage:.1f}%', 
                            ha='center', va='bottom', fontsize=9)
                
                # Add cumulative distribution curve
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                cumulative = np.cumsum(counts) / total * 100
                ax2.plot(bins[:-1] + 0.5, cumulative, 'r-', marker='o', markersize=4)
                ax2.set_ylabel('Cumulative Percentage (%)', color='r')
                ax2.tick_params(axis='y', colors='r')
                ax2.set_ylim([0, 105])
                
                for i, cum_pct in enumerate(cumulative):
                    if i % 2 == 0 or cum_pct > 95:  # Show fewer labels to avoid crowding
                        ax2.text(bins[i] + 0.5, cum_pct + 3, f'{cum_pct:.1f}%', 
                                color='r', ha='center', va='bottom', fontsize=8)
                
                plt.title('ETH Consecutive Drop Candles Distribution')
                ax1.set_xlabel('Number of Consecutive Drop Candles')
                ax1.set_ylabel('Frequency')
                ax1.grid(True, alpha=0.3)
                plt.xticks(bins[:-1] + 0.5)
                plt.savefig(os.path.join(self.output_dir, 'continuous_drops_distribution.png'), dpi=300)
            plt.close()
        
        # Visualize volume ratio distribution
        plt.figure(figsize=(10, 6))
        volume_ratios = eth_data['volume_ratio'].dropna()
        
        # Create more detailed histogram with percentile markers
        counts, bins, _ = plt.hist(volume_ratios, bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
        
        # Add percentile markers
        volume_percentiles = [50, 75, 90, 95, 99]
        for p in volume_percentiles:
            pct_val = np.percentile(volume_ratios, p)
            plt.axvline(x=pct_val, color='red', linestyle='--', alpha=0.7)
            plt.text(pct_val, plt.ylim()[1]*0.9, f'{p}%: {pct_val:.2f}x', 
                    rotation=90, verticalalignment='top')
        
        plt.axvline(x=1.0, color='black', linestyle='-', linewidth=1.5)
        plt.text(1.0, plt.ylim()[1]*0.8, 'Baseline (1.0x)', 
                rotation=90, verticalalignment='top')
        
        # Add mean line
        mean_val = volume_ratios.mean()
        plt.axvline(x=mean_val, color='blue', linestyle='-', alpha=0.7)
        plt.text(mean_val, plt.ylim()[1]*0.7, f'Mean: {mean_val:.2f}x', 
                rotation=90, verticalalignment='top')
        
        plt.title('Volume Ratio Distribution (Relative to 20-period Moving Average)')
        plt.xlabel('Volume Ratio')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Set sensible x-axis limits
        plt.xlim([0, min(10, np.percentile(volume_ratios, 99) * 1.2)])
        
        plt.savefig(os.path.join(self.output_dir, 'volume_ratio_distribution.png'), dpi=300)
        plt.close()
    
    def analyze_rebound_patterns(self, eth_data: pd.DataFrame, drop_thresholds: List[float] = [-0.01, -0.015, -0.02]) -> Dict[str, Any]:
        """
        Analyze ETH rebound patterns after drops
        
        Args:
            eth_data: ETH price data
            drop_thresholds: List of drop thresholds to analyze
            
        Returns:
            Dict[str, Any]: Rebound pattern analysis results
        """
        print("Analyzing ETH rebound patterns...")
        
        results = {}
        
        for threshold in drop_thresholds:
            # Identify drop events that meet the threshold
            eth_data['significant_drop'] = eth_data['close'].pct_change(5) <= threshold
            
            drop_events = []
            for idx, row in eth_data.iterrows():
                if row['significant_drop'] and not pd.isna(row['significant_drop']):
                    drop_events.append(idx)
            
            # Filter events that are too close, keep minimum 30 periods apart
            filtered_events = [drop_events[0]] if drop_events else []
            for event in drop_events[1:]:
                if (event - filtered_events[-1]).total_seconds() / 60 >= 30:  # At least 30 minutes apart
                    filtered_events.append(event)
            
            rebound_stats = []
            for event_time in filtered_events:
                # Get data after the event
                post_event_data = eth_data.loc[event_time:].iloc[:100]  # Take 100 periods
                
                if len(post_event_data) < 5:  # Need at least 5 periods
                    continue
                
                drop_price = post_event_data.iloc[0]['close']
                
                # Calculate time needed for various rebound thresholds
                rebound_times = {}
                for rebound_pct in [0.005, 0.01, 0.015, 0.02, 0.03]:
                    rebound_target = drop_price * (1 + rebound_pct)
                    
                    # Find time to reach target price
                    rebound_time = None
                    for i, (time, row) in enumerate(post_event_data.iterrows()):
                        if row['high'] >= rebound_target:
                            rebound_time = i
                            break
                    
                    rebound_times[rebound_pct] = rebound_time
                
                rebound_stats.append({
                    'event_time': event_time,
                    'rebound_times': rebound_times
                })
            
            # Calculate statistics for each rebound level
            rebound_summary = {}
            for rebound_pct in [0.005, 0.01, 0.015, 0.02, 0.03]:
                # Collect all times to reach this rebound level
                times = [s['rebound_times'][rebound_pct] for s in rebound_stats 
                        if s['rebound_times'][rebound_pct] is not None]
                
                if times:
                    rebound_summary[rebound_pct] = {
                        'count': len(times),
                        'mean': np.mean(times),
                        'median': np.median(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'percentiles': {p: np.percentile(times, p) for p in [25, 50, 75, 90]}
                    }
                else:
                    rebound_summary[rebound_pct] = {'count': 0}
            
            results[threshold] = {
                'drop_count': len(filtered_events),
                'rebound_summary': rebound_summary
            }
        
        self._visualize_rebound_analysis(results)
        
        return results
    
    def _visualize_rebound_analysis(self, results: Dict[str, Any]) -> None:
        """
        Visualize rebound analysis results
        
        Args:
            results: Rebound analysis results
        """
        # Visualize rebound times for different drop thresholds
        plt.figure(figsize=(14, 10))
        
        rebound_pcts = [0.005, 0.01, 0.015, 0.02, 0.03]
        thresholds = list(results.keys())
        
        # Create subplot for each threshold
        for i, threshold in enumerate(thresholds, 1):
            plt.subplot(len(thresholds), 1, i)
            
            rebound_data = []
            rebound_labels = []
            
            for pct in rebound_pcts:
                summary = results[threshold]['rebound_summary'][pct]
                if summary['count'] > 0:
                    rebound_data.append(summary['median'])
                    rebound_labels.append(f'{pct*100:.1f}%')
            
            if rebound_data:
                bars = plt.bar(rebound_labels, rebound_data, color='skyblue')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.1f}', ha='center', va='bottom')
            
            plt.title(f'Median Time (Candles) to Reach Rebound Levels after {threshold*100:.1f}% Drop')
            plt.ylabel('Number of Candles')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rebound_times_by_threshold.png'), dpi=300)
        plt.close()
    
    def generate_parameter_recommendations(self, drop_analysis: Dict[str, Any], 
                                          rebound_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate parameter recommendations
        
        Args:
            drop_analysis: Price drop analysis results
            rebound_analysis: Rebound pattern analysis results
            
        Returns:
            Dict[str, Any]: Parameter recommendations
        """
        print("Generating parameter recommendations...")
        
        recommendations = {}
        
        # Drop threshold recommendation (using 5-period window's 5th percentile)
        window_5_drops = drop_analysis[5]['percentiles']
        recommendations['drop_threshold'] = window_5_drops[5]  # 5th percentile
        
        # Consecutive drop candles recommendation (using median)
        continuous_stats = drop_analysis['continuous_drops']
        recommendations['consecutive_drops'] = round(continuous_stats['median'])
        
        # Volume factor recommendation (using 75th percentile)
        volume_stats = drop_analysis['volume_stats']
        recommendations['volume_factor'] = volume_stats['percentiles'][75]
        
        # Rebound threshold recommendation (average across thresholds)
        rebound_times = {}
        for threshold, data in rebound_analysis.items():
            rebound_summary = data['rebound_summary']
            
            # Find largest rebound percentage with median time within 15 candles
            target_pct = 0.005  # Default minimum
            for pct, stats in rebound_summary.items():
                if stats['count'] > 0 and stats['median'] <= 15 and pct > target_pct:
                    target_pct = pct
            
            rebound_times[threshold] = target_pct
        
        # Take average as recommended rebound threshold
        recommendations['rebound_threshold'] = np.mean(list(rebound_times.values()))
        
        # Take profit recommendation (2-3x rebound threshold)
        recommendations['take_profit_pct'] = recommendations['rebound_threshold'] * 2.5
        
        # Stop loss recommendation (absolute value of drop threshold, but not more than 3%)
        recommendations['stop_loss_pct'] = min(abs(recommendations['drop_threshold']), 0.03)
        
        # Round all float values
        for key, value in recommendations.items():
            if isinstance(value, float):
                recommendations[key] = round(value, 4)
        
        # Event window recommendations
        recommendations['pre_event_window'] = 15  # 15 minutes
        recommendations['post_event_window'] = 45  # 45 minutes
        
        # Default timeframe recommendations
        recommendations['timeframe'] = '5m'
        recommendations['days'] = 30
        
        # Save recommendations to file
        self._save_recommendations(recommendations)
        
        return recommendations
    
    def _save_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """
        Save parameter recommendations to file
        
        Args:
            recommendations: Parameter recommendations
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        recommendation_file = os.path.join(self.output_dir, f'param_recommendations_{timestamp}.json')
        
        with open(recommendation_file, 'w') as f:
            json.dump(recommendations, f, indent=4)
        
        print(f"Parameter recommendations saved to {recommendation_file}")
        
        # Generate command line parameter string
        cmd_params = []
        for key, value in recommendations.items():
            if key not in ['days', 'timeframe']:  # These are already included in main parameters
                if isinstance(value, bool):
                    if value:
                        cmd_params.append(f"--{key}")
                else:
                    cmd_params.append(f"--{key} {value}")
        
        cmd_line = f"python main.py --days {recommendations['days']} --timeframe {recommendations['timeframe']} " + " ".join(cmd_params)
        
        with open(os.path.join(self.output_dir, f'run_command_{timestamp}.txt'), 'w') as f:
            f.write(cmd_line)
        
        print(f"Run command saved to {os.path.join(self.output_dir, f'run_command_{timestamp}.txt')}")

    def run_analysis(self, days: int = 60, timeframe: str = '5m', 
                    end_date: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis workflow
        
        Args:
            days: Analysis period in days
            timeframe: Time frame
            end_date: End date
            use_cache: Whether to use cache
            
        Returns:
            Dict[str, Any]: Parameter recommendations
        """
        # Get ETH data
        eth_data = self.fetch_eth_data(days, timeframe, end_date, use_cache)
        
        if eth_data is None or eth_data.empty:
            print("Could not fetch ETH data, analysis terminated")
            return {}
        
        # Analyze price drops
        drop_analysis = self.analyze_price_drops(eth_data)
        
        # Analyze rebound patterns
        rebound_analysis = self.analyze_rebound_patterns(eth_data)
        
        # Generate parameter recommendations
        recommendations = self.generate_parameter_recommendations(drop_analysis, rebound_analysis)
        
        return recommendations

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ETH Parameter Analysis and Optimization Tool')
    
    parser.add_argument('--days', type=int, default=60, 
                        help='Analysis period in days (default: 60)')
    parser.add_argument('--timeframe', type=str, default='5m', 
                        choices=['1m', '5m', '15m', '1h', '4h'],
                        help='Time frame (default: 5m)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date in format YYYY-MM-DD')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable data cache')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    print("="*50)
    print("ETH Parameter Analysis and Optimization Tool")
    print("="*50)
    
    # Initialize data fetcher and parameter analyzer
    data_fetcher = DataFetcher()
    analyzer = ParamAnalyzer(data_fetcher)
    
    # Run analysis
    recommendations = analyzer.run_analysis(
        days=args.days,
        timeframe=args.timeframe,
        end_date=args.end_date,
        use_cache=not args.no_cache
    )
    
    if recommendations:
        print("\nRecommended Parameters:")
        print("-"*30)
        for key, value in recommendations.items():
            print(f"{key}: {value}")
    
    print("\nAnalysis complete!")
    print(f"Charts and parameter recommendations saved to: {analyzer.output_dir}")

if __name__ == "__main__":
    main() 