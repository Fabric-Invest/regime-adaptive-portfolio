"""
Volume-Price Divergence Calculations

Implements divergence indicators measuring the relationship
between volume and price movements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class DivergenceConfig:
    """Configuration for divergence calculations."""
    periods: Dict[str, int] = None
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = {
                '7d': 7,
                '30d': 30,
                '90d': 90,
            }


class DivergenceCalculator:
    """
    Calculator for Volume-Price Divergence indicators.
    
    Divergence measures the difference between normalized volume
    and price changes. Positive divergence (volume > price move)
    can indicate accumulation or distribution phases.
    """
    
    def __init__(self, config: Optional[DivergenceConfig] = None):
        self.config = config or DivergenceConfig()
    
    def calculate_normalized_change(
        self,
        series: pd.Series,
        period: int
    ) -> pd.Series:
        """
        Calculate z-score normalized change over period.
        
        Args:
            series: Input series (price or volume)
            period: Lookback period
            
        Returns:
            Z-score normalized change
        """
        pct_change = series.pct_change(periods=period)
        
        # Rolling z-score normalization
        rolling_mean = pct_change.rolling(window=period * 2).mean()
        rolling_std = pct_change.rolling(window=period * 2).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        z_score = (pct_change - rolling_mean) / rolling_std
        
        return z_score
    
    def calculate_divergence(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        period: int = 30
    ) -> pd.Series:
        """
        Calculate volume-price divergence.
        
        Divergence = normalized_volume_change - normalized_price_change
        
        Positive divergence: Volume moving more than price (accumulation/distribution)
        Negative divergence: Price moving more than volume (momentum exhaustion)
        
        Args:
            prices: Price series
            volumes: Volume series
            period: Lookback period
            
        Returns:
            Divergence values
        """
        price_z = self.calculate_normalized_change(prices, period)
        volume_z = self.calculate_normalized_change(volumes, period)
        
        divergence = volume_z - price_z
        
        return divergence
    
    def calculate_all_periods(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        coin_name: str
    ) -> pd.DataFrame:
        """
        Calculate divergence for all configured periods.
        
        Args:
            prices: Price series
            volumes: Volume series
            coin_name: Name of the coin for column naming
            
        Returns:
            DataFrame with divergence for each period
        """
        result = pd.DataFrame(index=prices.index)
        
        for period_name, period_days in self.config.periods.items():
            col_name = f"{coin_name}_divergence_vol_price_div_{period_name}"
            result[col_name] = self.calculate_divergence(
                prices, volumes, period_days
            )
        
        return result
    
    def get_average_divergence(
        self,
        div_df: pd.DataFrame,
        coins: list,
        period: str = '30d'
    ) -> pd.Series:
        """
        Calculate average divergence across specified coins.
        
        Used as input to the regime composite score.
        
        Args:
            div_df: DataFrame with divergence columns
            coins: List of coin names to average
            period: Period to use for divergence
            
        Returns:
            Series of average divergence values
        """
        div_cols = [
            f"{coin}_divergence_vol_price_div_{period}" for coin in coins
            if f"{coin}_divergence_vol_price_div_{period}" in div_df.columns
        ]
        
        if not div_cols:
            return pd.Series(0, index=div_df.index)
        
        return div_df[div_cols].mean(axis=1)
    
    def calculate_obv(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        OBV adds volume on up days and subtracts on down days.
        Divergence between OBV and price can signal trend changes.
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            OBV series
        """
        price_change = prices.diff()
        
        # +1 for up day, -1 for down day
        direction = np.sign(price_change)
        
        obv = (direction * volumes).cumsum()
        
        return obv
    
    def calculate_obv_divergence(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        period: int = 30
    ) -> pd.Series:
        """
        Calculate OBV divergence from price.
        
        Compares the trend of OBV with the trend of price.
        
        Args:
            prices: Price series
            volumes: Volume series
            period: Lookback period for trend comparison
            
        Returns:
            OBV divergence indicator
        """
        obv = self.calculate_obv(prices, volumes)
        
        # Normalize both series over the period
        price_pct = prices.pct_change(periods=period)
        obv_pct = obv.pct_change(periods=period)
        
        # Z-score normalize
        price_z = (price_pct - price_pct.rolling(period).mean()) / price_pct.rolling(period).std()
        obv_z = (obv_pct - obv_pct.rolling(period).mean()) / obv_pct.rolling(period).std()
        
        divergence = obv_z - price_z
        
        return divergence
    
    def detect_divergence_signal(
        self,
        divergence: pd.Series,
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Detect significant divergence signals.
        
        Args:
            divergence: Divergence series
            threshold: Z-score threshold for significant divergence
            
        Returns:
            Series with values: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        signal = pd.Series(0, index=divergence.index)
        
        # Bullish divergence: Volume rising faster than price
        signal[divergence > threshold] = 1
        
        # Bearish divergence: Volume falling faster than price
        signal[divergence < -threshold] = -1
        
        return signal
