"""
Volatility Indicator Calculations

Implements realized volatility indicators used for regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class VolatilityConfig:
    """Configuration for volatility calculations."""
    periods: Dict[str, int] = None
    annualization_factor: int = 252  # Trading days per year
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = {
                '21d': 21,
                '30d': 30,
                '90d': 90,
            }


class VolatilityCalculator:
    """
    Calculator for volatility indicators.
    
    Realized volatility is calculated as the rolling standard deviation
    of log returns, annualized by default.
    """
    
    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()
    
    def calculate_log_returns(
        self,
        prices: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate log returns from prices.
        
        Args:
            prices: Price series or DataFrame
            
        Returns:
            Log returns
        """
        return np.log(prices / prices.shift(1))
    
    def calculate_realized_volatility(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        period: int = 30,
        annualize: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate realized volatility (rolling standard deviation of returns).
        
        Args:
            prices: Price series or DataFrame
            period: Lookback period in days
            annualize: Whether to annualize the volatility
            
        Returns:
            Realized volatility values
        """
        log_returns = self.calculate_log_returns(prices)
        vol = log_returns.rolling(window=period).std()
        
        if annualize:
            vol = vol * np.sqrt(self.config.annualization_factor)
        
        return vol
    
    def calculate_all_periods(
        self,
        prices: pd.Series,
        coin_name: str
    ) -> pd.DataFrame:
        """
        Calculate realized volatility for all configured periods.
        
        Args:
            prices: Price series for a single coin
            coin_name: Name of the coin for column naming
            
        Returns:
            DataFrame with volatility for each period
        """
        result = pd.DataFrame(index=prices.index)
        
        for period_name, period_days in self.config.periods.items():
            col_name = f"{coin_name}_realized_vol_{period_name}"
            result[col_name] = self.calculate_realized_volatility(
                prices, period_days
            )
        
        return result
    
    def calculate_volatility_regime_signal(
        self,
        vol_df: pd.DataFrame,
        coins: list,
        period: str = '30d'
    ) -> pd.Series:
        """
        Calculate average volatility signal for regime detection.
        
        High volatility is typically associated with bearish conditions.
        The signal is inverted in the composite score calculation.
        
        Args:
            vol_df: DataFrame with volatility columns
            coins: List of coin names to average
            period: Volatility period to use
            
        Returns:
            Series of average volatility values
        """
        vol_cols = [
            f"{coin}_realized_vol_{period}" for coin in coins
            if f"{coin}_realized_vol_{period}" in vol_df.columns
        ]
        
        if not vol_cols:
            return pd.Series(0, index=vol_df.index)
        
        return vol_df[vol_cols].mean(axis=1)
    
    def calculate_volatility_z_score(
        self,
        volatility: pd.Series,
        lookback: int = 252
    ) -> pd.Series:
        """
        Calculate z-score of volatility relative to historical values.
        
        Useful for normalizing volatility across different market conditions.
        
        Args:
            volatility: Volatility series
            lookback: Lookback period for z-score calculation
            
        Returns:
            Z-score of volatility
        """
        rolling_mean = volatility.rolling(window=lookback).mean()
        rolling_std = volatility.rolling(window=lookback).std()
        
        return (volatility - rolling_mean) / rolling_std
    
    def get_current_volatility_regime(
        self,
        volatility: float,
        low_threshold: float = 0.3,
        high_threshold: float = 0.6
    ) -> str:
        """
        Classify current volatility into regime categories.
        
        Args:
            volatility: Current annualized volatility
            low_threshold: Threshold for low volatility
            high_threshold: Threshold for high volatility
            
        Returns:
            'low', 'normal', or 'high' volatility regime
        """
        if volatility < low_threshold:
            return 'low'
        elif volatility > high_threshold:
            return 'high'
        else:
            return 'normal'
