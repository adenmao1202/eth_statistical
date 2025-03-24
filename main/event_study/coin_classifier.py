#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coin Classifier Module
Classifies cryptocurrencies based on objective pre-event characteristics to avoid look-ahead bias
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime, timedelta
import logging

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_fetcher import DataFetcher


class CoinClassifier:
    """
    Cryptocurrency Classifier
    Classifies coins based on pre-event characteristics to avoid look-ahead bias
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the coin classifier
        
        Args:
            data_fetcher: DataFetcher instance
        """
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
    
    def classify_by_volume(self, symbols: List[str], days: int = 30, timeframe: str = '1h', tiers: int = 3) -> Dict[str, List[str]]:
        """
        Classify coins by trading volume
        
        Args:
            symbols: List of symbols to classify
            days: Number of days to analyze
            timeframe: Data timeframe
            tiers: Number of tiers to create
            
        Returns:
            Dict[str, List[str]]: Classification results
        """
        self.logger.info(f"Classifying {len(symbols)} symbols by trading volume")
        
        # Calculate average volume for each symbol
        volumes = {}
        for symbol in symbols:
            # Check if symbol already includes USDT suffix
            full_symbol = symbol if symbol.endswith('USDT') else f"{symbol}USDT"
            try:
                data = self.data_fetcher.fetch_historical_data(
                    symbol=full_symbol,
                    interval=timeframe,  # Use interval parameter instead of timeframe
                    days=days
                )
                
                if data is not None and not data.empty:
                    # Calculate average volume
                    avg_volume = data['volume'].mean()
                    volumes[symbol] = avg_volume
            except Exception as e:
                self.logger.warning(f"Error fetching data for {symbol}: {e}")
        
        # Sort symbols by volume
        sorted_symbols = sorted(volumes.keys(), key=lambda s: volumes[s], reverse=True)
        
        # Create tiers
        tier_size = len(sorted_symbols) // tiers
        result = {}
        
        for i in range(tiers):
            tier_name = f"high_volume" if i == 0 else f"mid_volume_{i}" if i < tiers - 1 else "low_volume"
            start_idx = i * tier_size
            end_idx = (i + 1) * tier_size if i < tiers - 1 else len(sorted_symbols)
            result[tier_name] = sorted_symbols[start_idx:end_idx]
        
        self.logger.info(f"Classified symbols into {len(result)} volume tiers")
        return result
    
    def classify_by_volatility(self, symbols: List[str], days: int = 30, timeframe: str = '1h', tiers: int = 3) -> Dict[str, List[str]]:
        """
        Classify coins by price volatility
        
        Args:
            symbols: List of symbols to classify
            days: Number of days to analyze
            timeframe: Data timeframe
            tiers: Number of tiers to create
            
        Returns:
            Dict[str, List[str]]: Classification results
        """
        self.logger.info(f"Classifying {len(symbols)} symbols by price volatility")
        
        # Calculate price volatility for each symbol
        volatilities = {}
        for symbol in symbols:
            # Check if symbol already includes USDT suffix
            full_symbol = symbol if symbol.endswith('USDT') else f"{symbol}USDT"
            try:
                data = self.data_fetcher.fetch_historical_data(
                    symbol=full_symbol,
                    interval=timeframe,  # Use interval parameter instead of timeframe
                    days=days
                )
                
                if data is not None and not data.empty:
                    # Calculate returns
                    data['returns'] = data['close'].pct_change()
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility = data['returns'].std()
                    volatilities[symbol] = volatility
            except Exception as e:
                self.logger.warning(f"Error fetching data for {symbol}: {e}")
        
        # Sort symbols by volatility
        sorted_symbols = sorted(volatilities.keys(), key=lambda s: volatilities[s], reverse=True)
        
        # Create tiers
        tier_size = len(sorted_symbols) // tiers
        result = {}
        
        for i in range(tiers):
            tier_name = f"high_volatility" if i == 0 else f"mid_volatility_{i}" if i < tiers - 1 else "low_volatility"
            start_idx = i * tier_size
            end_idx = (i + 1) * tier_size if i < tiers - 1 else len(sorted_symbols)
            result[tier_name] = sorted_symbols[start_idx:end_idx]
        
        self.logger.info(f"Classified symbols into {len(result)} volatility tiers")
        return result
    
    def create_index_baskets(self, symbols: List[str], num_baskets: int = 5, basket_size: int = 20) -> Dict[str, List[str]]:
        """
        Create random coin baskets for comparative analysis
        
        Args:
            symbols: List of coin symbols
            num_baskets: Number of baskets to create
            basket_size: Number of coins in each basket
            
        Returns:
            Dict[str, List[str]]: Dictionary of random baskets
        """
        import random
        self.logger.info(f"Creating {num_baskets} random baskets with {basket_size} coins each")
        
        # Ensure there are enough symbols
        if len(symbols) < basket_size:
            self.logger.warning(f"Not enough symbols ({len(symbols)}) to create baskets of size {basket_size}")
            # Return single basket with all symbols
            return {"random_basket_1": symbols}
        
        # Create multiple random baskets
        baskets = {}
        for i in range(num_baskets):
            # Randomly select symbols
            basket_symbols = random.sample(symbols, min(basket_size, len(symbols)))
            baskets[f"random_basket_{i+1}"] = basket_symbols
            
        self.logger.info(f"Successfully created {len(baskets)} random baskets")
        return baskets
    
    def classify_by_correlation(self, symbols: List[str], reference_symbols: List[str] = ['ETH', 'BTC'], 
                                days: int = 30, timeframe: str = '1h', tiers: int = 3) -> Dict[str, Dict[str, List[str]]]:
        """
        Classify coins by their price correlation with reference symbols (ETH and BTC)
        
        Args:
            symbols: List of symbols to classify
            reference_symbols: List of reference symbols to calculate correlation against
            days: Number of days to analyze
            timeframe: Data timeframe
            tiers: Number of tiers to create
            
        Returns:
            Dict[str, Dict[str, List[str]]]: Classification results for each reference symbol
        """
        self.logger.info(f"Classifying {len(symbols)} symbols by correlation with {reference_symbols}")
        
        # Get reference data first
        reference_data = {}
        for ref_symbol in reference_symbols:
            full_ref_symbol = ref_symbol if ref_symbol.endswith('USDT') else f"{ref_symbol}USDT"
            try:
                ref_data = self.data_fetcher.fetch_historical_data(
                    symbol=full_ref_symbol,
                    interval=timeframe,
                    days=days
                )
                
                if ref_data is not None and not ref_data.empty:
                    # Calculate returns
                    ref_data['returns'] = ref_data['close'].pct_change().dropna()
                    reference_data[ref_symbol] = ref_data
            except Exception as e:
                self.logger.warning(f"Error fetching data for reference symbol {ref_symbol}: {e}")
                
        if not reference_data:
            self.logger.error("Could not fetch any reference data, cannot calculate correlations")
            return {}
            
        # Calculate correlations for each symbol with each reference symbol
        correlations = {ref: {} for ref in reference_data.keys()}
        
        for symbol in symbols:
            if symbol in reference_symbols:
                continue  # Skip self-correlation
                
            full_symbol = symbol if symbol.endswith('USDT') else f"{symbol}USDT"
            try:
                data = self.data_fetcher.fetch_historical_data(
                    symbol=full_symbol,
                    interval=timeframe,
                    days=days
                )
                
                if data is not None and not data.empty:
                    # Calculate returns
                    data['returns'] = data['close'].pct_change().dropna()
                    
                    # Calculate correlation with each reference symbol
                    for ref_symbol, ref_data in reference_data.items():
                        # Align timestamps
                        merged = pd.merge(
                            data['returns'], 
                            ref_data['returns'], 
                            left_index=True, 
                            right_index=True,
                            how='inner',
                            suffixes=('', f'_{ref_symbol}')
                        )
                        
                        if not merged.empty and len(merged) > 5:  # Ensure enough data points
                            # Calculate Pearson correlation
                            correlation = merged['returns'].corr(merged[f'returns_{ref_symbol}'])
                            correlations[ref_symbol][symbol] = correlation
            except Exception as e:
                self.logger.warning(f"Error calculating correlation for {symbol}: {e}")
        
        # Create classification for each reference
        result = {}
        
        for ref_symbol, corr_dict in correlations.items():
            # Sort symbols by correlation (highest first)
            sorted_symbols = sorted(corr_dict.keys(), key=lambda s: corr_dict[s], reverse=True)
            
            # Create tiers
            if not sorted_symbols:
                continue
                
            tier_size = len(sorted_symbols) // tiers
            ref_result = {}
            
            for i in range(tiers):
                tier_name = f"high_correlation_{ref_symbol}" if i == 0 else \
                            f"mid_correlation_{ref_symbol}_{i}" if i < tiers - 1 else \
                            f"low_correlation_{ref_symbol}"
                start_idx = i * tier_size
                end_idx = (i + 1) * tier_size if i < tiers - 1 else len(sorted_symbols)
                ref_result[tier_name] = sorted_symbols[start_idx:end_idx]
            
            result[ref_symbol] = ref_result
            
            # Log actual correlation values for each tier
            for tier_name, tier_symbols in ref_result.items():
                if tier_symbols:
                    tier_corrs = [corr_dict[s] for s in tier_symbols]
                    self.logger.info(f"{tier_name}: Avg Correlation = {np.mean(tier_corrs):.4f}, " +
                                    f"Range = [{min(tier_corrs):.4f}, {max(tier_corrs):.4f}]")
        
        self.logger.info(f"Classified symbols by correlation with {len(result)} reference symbols")
        return result
    
    def get_stable_coins(self, all_symbols: List[str]) -> List[str]:
        """
        Identify stablecoins
        
        Args:
            all_symbols: List of all coin symbols
            
        Returns:
            List[str]: List of stablecoins
        """
        # Stablecoin name patterns
        stable_coin_patterns = [
            'USD', 'USDT', 'USDC', 'DAI', 'TUSD', 'PAX', 'BUSD', 'GUSD', 'USDK', 'USDP', 'UST', 'USDN', 
            'HUSD', 'SUSD', 'EURS', 'EURT', 'EURN', 'EUROC', 'EURU'
        ]
        
        # Identify stablecoins
        stable_coins = []
        for symbol in all_symbols:
            # Remove suffix (e.g., USDTUSDT -> USDT)
            base = symbol.replace('USDT', '')
            
            # Check if it's a stablecoin
            if base in stable_coin_patterns:
                stable_coins.append(symbol)
        
        self.logger.info(f"Identified {len(stable_coins)} stablecoins out of {len(all_symbols)} symbols")
        return stable_coins 