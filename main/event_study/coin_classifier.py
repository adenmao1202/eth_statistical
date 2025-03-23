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
        
    def classify_by_market_cap(self, symbols: List[str], tiers: int = 3) -> Dict[str, List[str]]:
        """
        Classify coins by market capitalization
        
        Args:
            symbols: List of coin symbols
            tiers: Number of classification tiers
            
        Returns:
            Dict[str, List[str]]: Dictionary of coins classified by market cap
        """
        self.logger.info(f"Classifying {len(symbols)} symbols by market cap into {tiers} tiers")
        
        # Since we lack direct market cap API, we use a simple proxy method:
        # 1. Use get_top_volume_symbols to get volume ranking
        # 2. Assume volume correlates with market cap
        try:
            # Use standard top 100 pairs as proxy
            ranked_symbols = self.data_fetcher.get_top_volume_symbols(100)
            
            # Build ranking dictionary (symbol -> rank)
            symbol_ranks = {s: i for i, s in enumerate(ranked_symbols)}
            
            # Sort input symbols by "market cap" (actually volume proxy)
            symbols_with_ranks = []
            for symbol in symbols:
                if symbol in symbol_ranks:
                    symbols_with_ranks.append((symbol, symbol_ranks[symbol]))
                else:
                    # For symbols not in ranking list, give lower priority
                    symbols_with_ranks.append((symbol, 999))
            
            # Sort by rank
            symbols_with_ranks.sort(key=lambda x: x[1])
            
            # Get symbols-only sorted list
            sorted_symbols = [item[0] for item in symbols_with_ranks]
            
        except Exception as e:
            self.logger.warning(f"Failed to use volume ranking as market cap proxy: {e}, using input symbol list directly")
            sorted_symbols = symbols.copy()
        
        # Divide into N tiers
        result = {}
        if not sorted_symbols:
            return {"large_cap": []}
            
        symbols_per_tier = max(1, len(sorted_symbols) // tiers)
        
        for i in range(tiers):
            start_idx = i * symbols_per_tier
            # For the last tier, take all remaining
            end_idx = None if i == tiers - 1 else (i + 1) * symbols_per_tier
            
            tier_name = ""
            if i == 0:
                tier_name = "large_cap"
            elif i == tiers - 1:
                tier_name = "small_cap"
            else:
                tier_name = f"mid_cap_{i}"
                
            result[tier_name] = sorted_symbols[start_idx:end_idx]
            
        return result
    
    def classify_by_volume(self, symbols: List[str], days: int = 30, timeframe: str = '1d', tiers: int = 3) -> Dict[str, List[str]]:
        """
        Classify coins by trading volume
        
        Args:
            symbols: List of coin symbols
            days: Analysis period in days
            timeframe: Time interval
            tiers: Number of classification tiers
            
        Returns:
            Dict[str, List[str]]: Dictionary of coins classified by volume
        """
        self.logger.info(f"Classifying {len(symbols)} symbols by volume into {tiers} tiers")
        
        # Only analyze a subset of coins to avoid API overload
        max_symbols_to_analyze = 100
        symbols_to_analyze = symbols[:max_symbols_to_analyze] if len(symbols) > max_symbols_to_analyze else symbols
        
        # Get volume data
        volumes = {}
        for symbol in symbols_to_analyze:
            try:
                # Get historical data
                data = self.data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval=timeframe,
                    days=days,
                    use_cache=True
                )
                
                if not data.empty:
                    # Calculate average daily volume
                    avg_volume = data['volume'].mean()
                    volumes[symbol] = avg_volume
            except Exception as e:
                self.logger.warning(f"Failed to get volume data for {symbol}: {e}")
        
        # Handle unanalyzed symbols
        if len(symbols_to_analyze) < len(symbols):
            self.logger.warning(f"Only analyzed volume for {len(symbols_to_analyze)} out of {len(symbols)} symbols, remaining will be placed in low volume category")
            
            # Unanalyzed symbols are assumed to be low volume
            for symbol in symbols[max_symbols_to_analyze:]:
                volumes[symbol] = 0
        
        # Sort
        sorted_volumes = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
        sorted_symbols = [item[0] for item in sorted_volumes]
        
        # Divide into N tiers
        result = {}
        if not sorted_symbols:
            return {"high_volume": []}
            
        symbols_per_tier = max(1, len(sorted_symbols) // tiers)
        
        for i in range(tiers):
            start_idx = i * symbols_per_tier
            # For the last tier, take all remaining
            end_idx = None if i == tiers - 1 else (i + 1) * symbols_per_tier
            
            tier_name = ""
            if i == 0:
                tier_name = "high_volume"
            elif i == tiers - 1:
                tier_name = "low_volume"
            else:
                tier_name = f"medium_volume_{i}"
                
            result[tier_name] = sorted_symbols[start_idx:end_idx]
            
        return result
    
    def classify_by_volatility(self, symbols: List[str], days: int = 30, timeframe: str = '1d', tiers: int = 3) -> Dict[str, List[str]]:
        """
        Classify coins by price volatility
        
        Args:
            symbols: List of coin symbols
            days: Analysis period in days
            timeframe: Time interval
            tiers: Number of classification tiers
            
        Returns:
            Dict[str, List[str]]: Dictionary of coins classified by volatility
        """
        self.logger.info(f"Classifying {len(symbols)} symbols by volatility into {tiers} tiers")
        
        # Only analyze a subset of coins to avoid API overload
        max_symbols_to_analyze = 100
        symbols_to_analyze = symbols[:max_symbols_to_analyze] if len(symbols) > max_symbols_to_analyze else symbols
        
        # Get volatility data
        volatilities = {}
        for symbol in symbols_to_analyze:
            try:
                # Get historical data
                data = self.data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval=timeframe,
                    days=days,
                    use_cache=True
                )
                
                if not data.empty and len(data) > 1:
                    # Calculate daily returns
                    data['returns'] = data['close'].pct_change()
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility = data['returns'].std()
                    volatilities[symbol] = volatility
            except Exception as e:
                self.logger.warning(f"Failed to calculate volatility for {symbol}: {e}")
        
        # Handle unanalyzed symbols
        if len(symbols_to_analyze) < len(symbols):
            self.logger.warning(f"Only analyzed volatility for {len(symbols_to_analyze)} out of {len(symbols)} symbols, remaining will be placed in medium volatility category")
            
            # Unanalyzed symbols are assumed to have medium volatility
            for symbol in symbols[max_symbols_to_analyze:]:
                if symbol not in volatilities:
                    volatilities[symbol] = 0.01  # Default value for medium volatility
        
        # Sort
        sorted_volatilities = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
        sorted_symbols = [item[0] for item in sorted_volatilities]
        
        # Divide into N tiers
        result = {}
        if not sorted_symbols:
            return {"high_volatility": []}
            
        symbols_per_tier = max(1, len(sorted_symbols) // tiers)
        
        for i in range(tiers):
            start_idx = i * symbols_per_tier
            # For the last tier, take all remaining
            end_idx = None if i == tiers - 1 else (i + 1) * symbols_per_tier
            
            tier_name = ""
            if i == 0:
                tier_name = "high_volatility"
            elif i == tiers - 1:
                tier_name = "low_volatility"
            else:
                tier_name = f"medium_volatility_{i}"
                
            result[tier_name] = sorted_symbols[start_idx:end_idx]
            
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
    
    def classify_by_sector(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Classify coins by industry sector (simulated implementation)
        
        Args:
            symbols: List of coin symbols
            
        Returns:
            Dict[str, List[str]]: Dictionary of coins classified by sector
        """
        self.logger.info(f"Classifying {len(symbols)} symbols by sector (simulated)")
        
        # Due to lack of direct sector classification data, we create simulated sector classifications
        # In real applications, this should be replaced with actual industry classification data
        import random
        
        # Define basic industry sectors
        sectors = {
            "defi": [],
            "gaming": [],
            "infrastructure": [],
            "exchange": [],
            "privacy": [],
            "storage": [],
            "other": []
        }
        
        # Simple keyword matching (very basic example)
        defi_keywords = ['uni', 'sushi', 'cake', 'comp', 'aave', 'mkr', 'crv', 'ldo', 'snx', 'rune']
        gaming_keywords = ['sand', 'mana', 'axs', 'ape', 'gala', 'ilv', 'enj', 'alice']
        infra_keywords = ['link', 'grt', 'fil', 'rndr', 'ar', 'api3', 'band', 'rlc']
        exchange_keywords = ['bnb', 'okb', 'cro', 'ftm', 'kcs', 'gt', 'ht', 'ftt']
        privacy_keywords = ['xmr', 'zcash', 'dash', 'scrt', 'rose', 'keep', 'mina']
        storage_keywords = ['fil', 'sc', 'storj', 'ar', 'ocean']
        
        # Simple rule matching, in real applications there should be more complex logic
        for symbol in symbols:
            base = symbol.replace('USDT', '').lower()
            
            if any(kw in base for kw in defi_keywords):
                sectors['defi'].append(symbol)
            elif any(kw in base for kw in gaming_keywords):
                sectors['gaming'].append(symbol)
            elif any(kw in base for kw in infra_keywords):
                sectors['infrastructure'].append(symbol)
            elif any(kw in base for kw in exchange_keywords):
                sectors['exchange'].append(symbol)
            elif any(kw in base for kw in privacy_keywords):
                sectors['privacy'].append(symbol)
            elif any(kw in base for kw in storage_keywords):
                sectors['storage'].append(symbol)
            else:
                sectors['other'].append(symbol)
        
        # Remove empty sectors
        sectors = {k: v for k, v in sectors.items() if v}
        
        # If all sectors are empty, create an "unclassified" category
        if not sectors:
            sectors['unclassified'] = symbols
        
        # Log classification results
        for sector, sector_symbols in sectors.items():
            self.logger.info(f"Sector '{sector}' contains {len(sector_symbols)} coins")
            
        return sectors
    
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