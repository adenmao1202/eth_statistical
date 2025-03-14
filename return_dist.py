#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH Return Distribution Analysis Tool
This script analyzes the return distribution of ETH over different timeframes (1m, 5m, 15m, 1h, 4h)
and generates visual charts to help set reasonable thresholds for downtrends and rebounds.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze ETH return distribution across different timeframes')
    
    parser.add_argument('--data_dir', type=str, default='historical_data',
                        help='Local data directory (default: historical_data)')
    
    parser.add_argument('--days', type=int, default=60,
                        help='Number of days to analyze (default: 60)')
    
    parser.add_argument('--windows', type=str, default='1,3,5,10,20',
                        help='Window sizes to analyze, separated by commas (default: 1,3,5,10,20)')
    
    parser.add_argument('--percentiles', type=str, default='1,5,10,25,50,75,90,95,99',
                        help='Percentiles to display, separated by commas (default: 1,5,10,25,50,75,90,95,99)')
    
    parser.add_argument('--timeframes', type=str, default='1m,5m,15m,1h,4h',
                        help='Timeframes to analyze, separated by commas (default: 1m,5m,15m,1h,4h)')
    
    return parser.parse_args()


def load_local_data(data_dir, symbol, timeframe):
    """Load data from a local file"""
    filename = f"{data_dir}/{symbol}_{timeframe}.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    return None


def analyze_eth_returns(data_dir, timeframes, start_date, end_date, windows):
    """
    Analyze the return distribution of ETH over different timeframes and window sizes.
    
    Args:
        data_dir (str): Data directory.
        timeframes (list): List of timeframes.
        start_date (str): Start date.
        end_date (str): End date.
        windows (list): List of window sizes.
        
    Returns:
        dict: A dictionary containing the return distribution for each timeframe and window size.
    """
    eth_symbol = 'ETHUSDT'
    results = {}
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for timeframe in timeframes:
        print(f"\nAnalyzing ETH return distribution for the {timeframe} timeframe...")
        
        # Load ETH data
        eth_data = load_local_data(data_dir, eth_symbol, timeframe)
        
        if eth_data is None:
            print(f"Warning: Could not find ETH {timeframe} data file {data_dir}/{eth_symbol}_{timeframe}.csv")
            continue
        
        # Filter by date range
        eth_data = eth_data[(eth_data.index >= start_dt) & (eth_data.index <= end_dt)]
        
        if len(eth_data) < max(windows) + 1:
            print(f"Warning: Insufficient data for ETH {timeframe}. At least {max(windows) + 1} data points are required.")
            continue
        
        # Calculate returns for different window sizes
        timeframe_results = {}
        
        for window in windows:
            # Calculate returns
            eth_data[f'return_{window}'] = eth_data['close'].pct_change(window) * 100
            
            # Remove NaN values
            returns = eth_data[f'return_{window}'].dropna()
            
            timeframe_results[window] = returns
            
        results[timeframe] = timeframe_results
    
    return results


def calculate_statistics(returns_data, percentiles):
    """
    Calculate statistical metrics for the returns.
    
    Args:
        returns_data (dict): Dictionary containing returns data for different timeframes and window sizes.
        percentiles (list): List of percentiles to calculate.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the statistical metrics.
    """
    stats = []
    
    for timeframe, timeframe_data in returns_data.items():
        for window, returns in timeframe_data.items():
            row = {
                'Timeframe': timeframe,
                'Window Size': window,
                'Sample Size': len(returns),
                'Min': returns.min(),
                'Max': returns.max(),
                'Mean': returns.mean(),
                'Median': returns.median(),
                'Std Dev': returns.std()
            }
            
            # Add percentiles
            for p in percentiles:
                row[f'{p}% Percentile'] = np.percentile(returns, p)
            
            stats.append(row)
    
    # Create DataFrame and sort
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values(['Timeframe', 'Window Size'])
    
    return stats_df


def plot_return_distributions(returns_data):
    """
    Plot return distribution charts.
    
    Args:
        returns_data (dict): Dictionary containing returns data for different timeframes and window sizes.
    """
    # Create a figure for each timeframe
    for timeframe, timeframe_data in returns_data.items():
        # Determine the number of subplots needed
        n_windows = len(timeframe_data)
        n_cols = min(3, n_windows)
        n_rows = (n_windows + n_cols - 1) // n_cols
        
        # Create the figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        # Plot the distribution for each window size
        for i, (window, returns) in enumerate(sorted(timeframe_data.items())):
            ax = axes[i // n_cols, i % n_cols] if n_rows > 1 and n_cols > 1 else axes[i]
            
            # Plot histogram and KDE
            sns.histplot(returns, kde=True, ax=ax, bins=50)
            
            # Add vertical lines for different thresholds
            ax.axvline(x=-5, color='r', linestyle='--', alpha=0.7, label='-5%')
            ax.axvline(x=-3, color='orange', linestyle='--', alpha=0.7, label='-3%')
            ax.axvline(x=3, color='g', linestyle='--', alpha=0.7, label='3%')
            ax.axvline(x=5, color='blue', linestyle='--', alpha=0.7, label='5%')
            
            # Set title and labels
            ax.set_title(f'{timeframe} timeframe, Window Size = {window}')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            
            # Add statistical information text
            stats_text = f"Mean: {returns.mean():.2f}%\n"
            stats_text += f"Median: {returns.median():.2f}%\n"
            stats_text += f"Std Dev: {returns.std():.2f}%\n"
            stats_text += f"Min: {returns.min():.2f}%\n"
            stats_text += f"Max: {returns.max():.2f}%"
            
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove extra subplots if any
        for i in range(len(timeframe_data), n_rows * n_cols):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.savefig(f'eth_return_distribution_{timeframe}.png', dpi=300)
        plt.show()


def plot_percentile_heatmap(stats_df, percentiles):
    """
    Plot Percentile Heatmap
    
    Args:
        stats_df (pandas.DataFrame): DataFrame containing statistical metrics.
        percentiles (list): List of percentiles.
    """
    # Create heatmaps for downtrend and uptrend separately
    
    # 1. Downtrend Percentiles (1%, 5%, 10%, 25%)
    downtrend_percentiles = [p for p in percentiles if p <= 25]
    downtrend_cols = [f'{p}% Percentile' for p in downtrend_percentiles]
    
    # Check if columns exist
    print("Available columns in stats_df:")
    print(stats_df.columns.tolist())
    
    # Ensure that all required columns exist
    missing_cols = [col for col in downtrend_cols if col not in stats_df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing: {missing_cols}")
        # Use only the existing columns
        downtrend_cols = [col for col in downtrend_cols if col in stats_df.columns]
        if not downtrend_cols:
            print("Error: No valid downtrend percentile columns found")
            return
    
    pivot_down = stats_df.pivot_table(
        index='Window Size', 
        columns='Timeframe',
        values=downtrend_cols
    )
    
    # Check pivot table structure
    print("Pivot table structure:")
    print(pivot_down.columns.names)
    print(pivot_down.columns.levels)
    
    # Create multi-index heatmaps
    fig, axes = plt.subplots(len(downtrend_cols), 1, figsize=(12, 4 * len(downtrend_cols)))
    if len(downtrend_cols) == 1:
        axes = [axes]  # Ensure axes is always a list
    
    for i, col in enumerate(downtrend_cols):
        ax = axes[i]
        
        try:
            # Try extracting data using xs
            data = pivot_down.xs(col, axis=1, level=1)
        except KeyError:
            # If it fails, try an alternative approach
            print(f"Could not find column '{col}' in level 1. Trying alternative approach.")
            print(f"Pivot table columns: {pivot_down.columns}")
            
            # Attempt to find all columns containing the percentile
            cols_with_percentile = [c for c in pivot_down.columns if col in str(c)]
            if not cols_with_percentile:
                print(f"No columns containing '{col}' found. Skipping this percentile.")
                continue
            
            # Create a new DataFrame with only these columns
            data = pd.DataFrame()
            for c in cols_with_percentile:
                tf = c[0]  # Timeframe
                data[tf] = pivot_down[c]
        
        # Plot heatmap
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        
        p = int(col.split('%')[0])
        ax.set_title(f'ETH {p}% Percentile Return (%) - Possible Downtrend Threshold')
        ax.set_ylabel('Window Size')
    
    plt.tight_layout()
    plt.savefig('eth_downtrend_percentiles.png', dpi=300)
    plt.show()
    
    # 2. Uptrend Percentiles (75%, 90%, 95%, 99%)
    uptrend_percentiles = [p for p in percentiles if p >= 75]
    uptrend_cols = [f'{p}% Percentile' for p in uptrend_percentiles]
    
    # Ensure that all required columns exist
    missing_cols = [col for col in uptrend_cols if col not in stats_df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing: {missing_cols}")
        # Use only the existing columns
        uptrend_cols = [col for col in uptrend_cols if col in stats_df.columns]
        if not uptrend_cols:
            print("Error: No valid uptrend percentile columns found")
            return
    
    pivot_up = stats_df.pivot_table(
        index='Window Size', 
        columns='Timeframe',
        values=uptrend_cols
    )
    
    # Create multi-index heatmaps
    fig, axes = plt.subplots(len(uptrend_cols), 1, figsize=(12, 4 * len(uptrend_cols)))
    if len(uptrend_cols) == 1:
        axes = [axes]  # Ensure axes is always a list
    
    for i, col in enumerate(uptrend_cols):
        ax = axes[i]
        
        try:
            # Try extracting data using xs
            data = pivot_up.xs(col, axis=1, level=1)
        except KeyError:
            # If it fails, try an alternative approach
            print(f"Could not find column '{col}' in level 1. Trying alternative approach.")
            
            # Attempt to find all columns containing the percentile
            cols_with_percentile = [c for c in pivot_up.columns if col in str(c)]
            if not cols_with_percentile:
                print(f"No columns containing '{col}' found. Skipping this percentile.")
                continue
            
            # Create a new DataFrame with only these columns
            data = pd.DataFrame()
            for c in cols_with_percentile:
                tf = c[0]  # Timeframe
                data[tf] = pivot_up[c]
        
        # Plot heatmap
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm_r', ax=ax)
        
        p = int(col.split('%')[0])
        ax.set_title(f'ETH {p}% Percentile Return (%) - Possible Rebound Threshold')
        ax.set_ylabel('Window Size')
    
    plt.tight_layout()
    plt.savefig('eth_rebound_percentiles.png', dpi=300)
    plt.show()


def plot_threshold_exceedance(returns_data):
    """
    Plot frequency charts for threshold exceedances.
    
    Args:
        returns_data (dict): Dictionary containing returns data for different timeframes and window sizes.
    """
    # Define thresholds
    downtrend_thresholds = [-10, -7, -5, -3, -2, -1]
    rebound_thresholds = [1, 2, 3, 5, 7, 10]
    
    # Store results
    downtrend_results = []
    rebound_results = []
    
    for timeframe, timeframe_data in returns_data.items():
        for window, returns in timeframe_data.items():
            # Calculate frequency of exceedance for downtrend thresholds
            for threshold in downtrend_thresholds:
                frequency = (returns <= threshold).mean() * 100
                downtrend_results.append({
                    'Timeframe': timeframe,
                    'Window Size': window,
                    'Threshold': threshold,
                    'Exceedance Frequency (%)': frequency
                })
            
            # Calculate frequency of exceedance for rebound thresholds
            for threshold in rebound_thresholds:
                frequency = (returns >= threshold).mean() * 100
                rebound_results.append({
                    'Timeframe': timeframe,
                    'Window Size': window,
                    'Threshold': threshold,
                    'Exceedance Frequency (%)': frequency
                })
    
    # Create DataFrames
    downtrend_df = pd.DataFrame(downtrend_results)
    rebound_df = pd.DataFrame(rebound_results)
    
    # Plot heatmap for downtrend thresholds
    plt.figure(figsize=(15, 10))
    pivot_down = downtrend_df.pivot_table(
        index=['Timeframe', 'Window Size'],
        columns='Threshold',
        values='Exceedance Frequency (%)'
    )
    
    sns.heatmap(pivot_down, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('ETH Downtrend Threshold Exceedance Frequency (%)')
    plt.tight_layout()
    plt.savefig('eth_downtrend_threshold_frequency.png', dpi=300)
    plt.show()
    
    # Plot heatmap for rebound thresholds
    plt.figure(figsize=(15, 10))
    pivot_up = rebound_df.pivot_table(
        index=['Timeframe', 'Window Size'],
        columns='Threshold',
        values='Exceedance Frequency (%)'
    )
    
    sns.heatmap(pivot_up, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('ETH Rebound Threshold Exceedance Frequency (%)')
    plt.tight_layout()
    plt.savefig('eth_rebound_threshold_frequency.png', dpi=300)
    plt.show()


def generate_parameter_recommendations(stats_df, percentiles):
    """
    Generate parameter recommendations based on statistical data.
    
    Args:
        stats_df (pandas.DataFrame): DataFrame containing statistical metrics.
        percentiles (list): List of percentiles.
        
    Returns:
        pandas.DataFrame: A DataFrame containing parameter recommendations.
    """
    recommendations = []
    
    # Get all combinations of timeframes and window sizes
    timeframes = stats_df['Timeframe'].unique()
    windows = stats_df['Window Size'].unique()
    
    # Define percentiles for different risk profiles
    risk_profiles = {
        'Conservative': {'downtrend': 10, 'rebound': 90},  # 10th percentile as downtrend threshold, 90th percentile as rebound threshold
        'Moderate': {'downtrend': 5, 'rebound': 95},         # 5th percentile as downtrend threshold, 95th percentile as rebound threshold
        'Aggressive': {'downtrend': 1, 'rebound': 99}          # 1st percentile as downtrend threshold, 99th percentile as rebound threshold
    }
    
    for timeframe in timeframes:
        for window in windows:
            # Get the statistical data for the current timeframe and window size
            row = stats_df[(stats_df['Timeframe'] == timeframe) & (stats_df['Window Size'] == window)]
            
            if len(row) == 0:
                continue
            
            # Generate recommendations for each risk profile
            for profile, thresholds in risk_profiles.items():
                downtrend_percentile = thresholds['downtrend']
                rebound_percentile = thresholds['rebound']
                
                # Check if the corresponding percentile columns exist
                if f'{downtrend_percentile}% Percentile' in row.columns and f'{rebound_percentile}% Percentile' in row.columns:
                    downtrend_value = row[f'{downtrend_percentile}% Percentile'].values[0]
                    rebound_value = row[f'{rebound_percentile}% Percentile'].values[0]
                    
                    # Round to the nearest integer or half-integer
                    downtrend_rounded = round(downtrend_value * 2) / 2
                    rebound_rounded = round(rebound_value * 2) / 2
                    
                    recommendations.append({
                        'Timeframe': timeframe,
                        'Window Size': window,
                        'Risk Profile': profile,
                        'Suggested Downtrend Threshold': downtrend_rounded,
                        'Suggested Rebound Threshold': rebound_rounded,
                        'Downtrend Percentile': downtrend_percentile,
                        'Rebound Percentile': rebound_percentile
                    })
    
    # Create DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    
    return recommendations_df


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # Parse window sizes and percentiles
    windows = [int(w) for w in args.windows.split(',')]
    percentiles = [int(p) for p in args.percentiles.split(',')]
    timeframes = args.timeframes.split(',')
    
    print(f"Analyzing ETH return distribution across different timeframes")
    print(f"Data directory: {args.data_dir}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Timeframes: {timeframes}")
    print(f"Window sizes: {windows}")
    print(f"Percentiles: {percentiles}")
    
    # Analyze ETH returns
    returns_data = analyze_eth_returns(args.data_dir, timeframes, start_date, end_date, windows)
    
    if not returns_data:
        print("Error: No ETH data found")
        return
    
    # Calculate statistical metrics
    stats_df = calculate_statistics(returns_data, percentiles)
    
    # Save statistical metrics to CSV
    stats_df.to_csv('eth_return_statistics.csv', index=False)
    print("Statistical metrics saved to 'eth_return_statistics.csv'")
    
    # Plot return distribution charts
    plot_return_distributions(returns_data)
    
    # Plot percentile heatmaps
    plot_percentile_heatmap(stats_df, percentiles)
    
    # Plot threshold exceedance charts
    plot_threshold_exceedance(returns_data)
    
    # Generate parameter recommendations
    recommendations_df = generate_parameter_recommendations(stats_df, percentiles)
    
    # Save parameter recommendations to CSV
    recommendations_df.to_csv('eth_parameter_recommendations.csv', index=False)
    print("Parameter recommendations saved to 'eth_parameter_recommendations.csv'")
    
    # Print parameter recommendations
    print("\nParameter Recommendations:")
    for timeframe in timeframes:
        print(f"\nTimeframe: {timeframe}")
        for window in windows:
            tf_window_recommendations = recommendations_df[
                (recommendations_df['Timeframe'] == timeframe) & 
                (recommendations_df['Window Size'] == window)
            ]
            
            if len(tf_window_recommendations) > 0:
                print(f"  Window Size: {window}")
                for _, row in tf_window_recommendations.iterrows():
                    print(f"    {row['Risk Profile']} strategy: Downtrend Threshold = {row['Suggested Downtrend Threshold']:.1f}%, Rebound Threshold = {row['Suggested Rebound Threshold']:.1f}%")
    
    print("\nAnalysis complete!")
    print("Charts have been saved as:")
    print("- 'eth_return_distribution_*.png'")
    print("- 'eth_downtrend_percentiles.png'")
    print("- 'eth_rebound_percentiles.png'")
    print("- 'eth_downtrend_threshold_frequency.png'")
    print("- 'eth_rebound_threshold_frequency.png'")


if __name__ == "__main__":
    main()
