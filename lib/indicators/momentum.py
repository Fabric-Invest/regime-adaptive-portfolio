"""
Momentum Indicator Calculations

Implements various momentum indicators used for regime detection
and coin selection/ranking.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class MomentumConfig:
    """Configuration for momentum calculations."""
    periods: Dict[str, int] = None
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = {
                '7d': 7,
                '30d': 30,
                '90d': 90,
                '180d': 180,
                '12w': 84,  # 12 weeks = 84 trading days
            }


class MomentumCalculator:
    """
    Calculator for momentum indicators.
    
    Momentum is calculated as the percentage change over a specified period.
    Used for both regime detection and ranking coins for allocation.
    """
    
    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()
    
    def calculate(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        period: int = 84
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate momentum (rate of change) for given period.
        
        Args:
            prices: Price series or DataFrame with price columns
            period: Lookback period in days
            
        Returns:
            Momentum values (percentage change over period)
        """
        return prices.pct_change(periods=period)
    
    def calculate_all_periods(
        self,
        prices: pd.Series,
        coin_name: str
    ) -> pd.DataFrame:
        """
        Calculate momentum for all configured periods.
        
        Args:
            prices: Price series for a single coin
            coin_name: Name of the coin for column naming
            
        Returns:
            DataFrame with momentum for each period
        """
        result = pd.DataFrame(index=prices.index)
        
        for period_name, period_days in self.config.periods.items():
            col_name = f"{coin_name}_momentum_{period_name}"
            result[col_name] = self.calculate(prices, period_days)
        
        return result
    
    def calculate_12w_momentum(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        lag: int = 1
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate 12-week (84-day) momentum with optional lag.
        
        This is the primary momentum indicator used for:
        - Regime detection composite score
        - Ranking coins for allocation
        
        Args:
            prices: Price series or DataFrame
            lag: Number of days to lag (avoid look-ahead bias)
            
        Returns:
            12-week momentum values
        """
        momentum = self.calculate(prices, period=84)
        
        if lag > 0:
            momentum = momentum.shift(lag)
        
        return momentum
    
    def rank_coins_by_momentum(
        self,
        coin_momentum: Dict[str, float]
    ) -> list:
        """
        Rank coins by their 12-week momentum (descending).
        
        Args:
            coin_momentum: Dictionary of coin -> momentum value
            
        Returns:
            List of coin names sorted by momentum (highest first)
        """
        # Filter out NaN values
        valid_momentum = {
            coin: mom for coin, mom in coin_momentum.items()
            if not pd.isna(mom)
        }
        
        return sorted(
            valid_momentum.keys(),
            key=lambda x: valid_momentum[x],
            reverse=True
        )
    
    def calculate_momentum_weights(
        self,
        coin_momentum: Dict[str, float],
        top_n: int
    ) -> Dict[str, float]:
        """
        Calculate momentum-weighted allocation for top N coins.
        
        Weights are proportional to positive momentum values.
        If all momentum values are non-positive, uses equal weights.
        
        Args:
            coin_momentum: Dictionary of coin -> momentum value
            top_n: Number of top coins to include
            
        Returns:
            Dictionary of coin -> weight (sums to 1.0)
        """
        # Get top N coins
        ranked_coins = self.rank_coins_by_momentum(coin_momentum)[:top_n]
        
        if not ranked_coins:
            return {}
        
        # Get momentum values for top coins
        weights = np.array([coin_momentum[c] for c in ranked_coins])
        
        # Ensure non-negative weights
        weights = np.maximum(weights, 0)
        
        # Normalize to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Equal weights if all momentum is non-positive
            weights = np.ones(len(ranked_coins)) / len(ranked_coins)
        
        return dict(zip(ranked_coins, weights))
    
    def get_average_momentum(
        self,
        momentum_df: pd.DataFrame,
        coins: list
    ) -> pd.Series:
        """
        Calculate average 12-week momentum across specified coins.
        
        Used as input to the regime composite score.
        
        Args:
            momentum_df: DataFrame with momentum columns
            coins: List of coin names to average
            
        Returns:
            Series of average momentum values
        """
        momentum_cols = [
            f"{coin}_momentum_12w" for coin in coins
            if f"{coin}_momentum_12w" in momentum_df.columns
        ]
        
        if not momentum_cols:
            return pd.Series(0, index=momentum_df.index)
        
        return momentum_df[momentum_cols].mean(axis=1)
