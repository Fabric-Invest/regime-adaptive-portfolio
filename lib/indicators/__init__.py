"""
Technical Indicators Module

Contains implementations of various technical indicators used
for regime detection and portfolio allocation.
"""

from lib.indicators.momentum import MomentumCalculator
from lib.indicators.volatility import VolatilityCalculator
from lib.indicators.ma_spreads import MASpreadCalculator
from lib.indicators.divergence import DivergenceCalculator

__all__ = [
    'MomentumCalculator',
    'VolatilityCalculator',
    'MASpreadCalculator',
    'DivergenceCalculator',
]
