"""
Moving Average Spread Calculations

Implements MA spread indicators used for regime detection.
MA spreads measure the relationship between short and long-term trends.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class MASpreadConfig:
    """Configuration for MA spread calculations."""
    ma_periods: Dict[str, int] = None
    spread_pairs: list = None
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = {
                '21': 21,
                '50': 50,
                '90': 90,
                '200': 200,
            }
        if self.spread_pairs is None:
            # (short_period, long_period) pairs
            self.spread_pairs = [
                ('21', '50'),
                ('21', '90'),
                ('21', '200'),
                ('50', '90'),
                ('50', '200'),
                ('90', '200'),
            ]


class MASpreadCalculator:
    """
    Calculator for Moving Average spread indicators.
    
    MA spreads measure the percentage difference between short and long
    moving averages. Positive spreads indicate bullish conditions,
    negative spreads indicate bearish conditions.
    """
    
    def __init__(self, config: Optional[MASpreadConfig] = None):
        self.config = config or MASpreadConfig()
    
    def calculate_sma(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        period: int
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Price series or DataFrame
            period: MA period in days
            
        Returns:
            SMA values
        """
        return prices.rolling(window=period).mean()
    
    def calculate_ema(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        period: int
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series or DataFrame
            period: EMA span
            
        Returns:
            EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_ma_spread(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        short_period: int,
        long_period: int,
        use_ema: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate spread between short and long moving averages.
        
        Spread = (MA_short - MA_long) / MA_long * 100
        
        Args:
            prices: Price series or DataFrame
            short_period: Short MA period
            long_period: Long MA period
            use_ema: Use EMA instead of SMA
            
        Returns:
            MA spread as percentage
        """
        if use_ema:
            ma_short = self.calculate_ema(prices, short_period)
            ma_long = self.calculate_ema(prices, long_period)
        else:
            ma_short = self.calculate_sma(prices, short_period)
            ma_long = self.calculate_sma(prices, long_period)
        
        # Percentage spread
        spread = (ma_short - ma_long) / ma_long * 100
        
        return spread
    
    def calculate_all_mas(
        self,
        prices: pd.Series,
        coin_name: str
    ) -> pd.DataFrame:
        """
        Calculate all configured moving averages.
        
        Args:
            prices: Price series for a single coin
            coin_name: Name of the coin for column naming
            
        Returns:
            DataFrame with all MA columns
        """
        result = pd.DataFrame(index=prices.index)
        
        for period_name, period_days in self.config.ma_periods.items():
            col_name = f"{coin_name}_ma_{period_name}"
            result[col_name] = self.calculate_sma(prices, period_days)
        
        return result
    
    def calculate_all_spreads(
        self,
        prices: pd.Series,
        coin_name: str
    ) -> pd.DataFrame:
        """
        Calculate all configured MA spreads.
        
        Args:
            prices: Price series for a single coin
            coin_name: Name of the coin for column naming
            
        Returns:
            DataFrame with all MA spread columns
        """
        result = pd.DataFrame(index=prices.index)
        
        for short_name, long_name in self.config.spread_pairs:
            short_period = self.config.ma_periods[short_name]
            long_period = self.config.ma_periods[long_name]
            
            col_name = f"{coin_name}_ma_spread_{short_name}_{long_name}"
            result[col_name] = self.calculate_ma_spread(
                prices, short_period, long_period
            )
        
        return result
    
    def get_average_ma_spread(
        self,
        spread_df: pd.DataFrame,
        coins: list,
        spread_pair: Tuple[str, str] = ('50', '200')
    ) -> pd.Series:
        """
        Calculate average MA spread across specified coins.
        
        Used as input to the regime composite score.
        The 50/200 spread is the classic "golden cross / death cross" signal.
        
        Args:
            spread_df: DataFrame with spread columns
            coins: List of coin names to average
            spread_pair: Tuple of (short_period, long_period) names
            
        Returns:
            Series of average spread values
        """
        spread_cols = [
            f"{coin}_ma_spread_{spread_pair[0]}_{spread_pair[1]}"
            for coin in coins
            if f"{coin}_ma_spread_{spread_pair[0]}_{spread_pair[1]}" in spread_df.columns
        ]
        
        if not spread_cols:
            return pd.Series(0, index=spread_df.index)
        
        return spread_df[spread_cols].mean(axis=1)
    
    def get_trend_direction(
        self,
        prices: pd.Series,
        short_period: int = 50,
        long_period: int = 200
    ) -> pd.Series:
        """
        Get trend direction based on MA crossover.
        
        Args:
            prices: Price series
            short_period: Short MA period
            long_period: Long MA period
            
        Returns:
            Series with values: 1 (uptrend), -1 (downtrend), 0 (neutral)
        """
        ma_short = self.calculate_sma(prices, short_period)
        ma_long = self.calculate_sma(prices, long_period)
        
        trend = pd.Series(0, index=prices.index)
        trend[ma_short > ma_long] = 1   # Golden cross / uptrend
        trend[ma_short < ma_long] = -1  # Death cross / downtrend
        
        return trend
    
    def detect_crossover(
        self,
        prices: pd.Series,
        short_period: int = 50,
        long_period: int = 200
    ) -> pd.DataFrame:
        """
        Detect MA crossover events.
        
        Args:
            prices: Price series
            short_period: Short MA period
            long_period: Long MA period
            
        Returns:
            DataFrame with golden_cross and death_cross boolean columns
        """
        ma_short = self.calculate_sma(prices, short_period)
        ma_long = self.calculate_sma(prices, long_period)
        
        # Previous day comparison (treat missing as False so ~ is valid)
        short_above = ma_short > ma_long
        short_above_prev = short_above.shift(1).fillna(False).astype(bool)
        
        result = pd.DataFrame(index=prices.index)
        result['golden_cross'] = short_above & ~short_above_prev  # Crossed above
        result['death_cross'] = ~short_above & short_above_prev   # Crossed below
        
        return result
