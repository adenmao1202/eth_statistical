#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter Analysis Startup Script
Simplifies the startup process for the parameter analysis tool
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    """Main function"""
    print("="*60)
    print("ETH Hypothesis Testing Parameter Analysis - Easy Startup Tool")
    print("="*60)
    
    # Setup default options
    days = 60
    timeframe = '5m'
    use_cache = True
    
    # Ask user for configuration
    print("\nPlease configure analysis parameters:")
    
    while True:
        days_input = input(f"Analysis days [{days}]: ").strip()
        if days_input:
            try:
                days = int(days_input)
                if days <= 0:
                    print("Days must be a positive integer, please try again")
                    continue
                break
            except ValueError:
                print("Please enter a valid integer")
                continue
        else:
            break
    
    valid_timeframes = ['1m', '5m', '15m', '1h', '4h']
    while True:
        timeframe_input = input(f"Timeframe [{timeframe}] (options: {', '.join(valid_timeframes)}): ").strip()
        if timeframe_input:
            if timeframe_input in valid_timeframes:
                timeframe = timeframe_input
                break
            else:
                print(f"Please enter a valid timeframe: {', '.join(valid_timeframes)}")
                continue
        else:
            break
    
    cache_input = input(f"Use data cache [Y/n]: ").strip().lower()
    if cache_input and cache_input[0] == 'n':
        use_cache = False
    
    # Construct command
    cmd = [sys.executable, "param_analyzer.py", "--days", str(days), "--timeframe", timeframe]
    
    if not use_cache:
        cmd.append("--no_cache")
    
    # Execute command
    print("\nStarting parameter analysis...")
    print(f"Command: {' '.join(cmd)}")
    print("-"*60)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed, error code: {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\nUser interrupted execution")
        return
    
    print("\nParameter analysis complete!")
    print("You can now use the recommended parameters for ETH hypothesis testing in the main program.")

if __name__ == "__main__":
    main() 