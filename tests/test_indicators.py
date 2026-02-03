"""
Unit tests for indicator calculations.
"""

import sys
from pathlib import Path

# Add project root so "lib" is importable when run as script or from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import numpy as np
import pandas as pd

from lib.indicators.momentum import MomentumCalculator, MomentumConfig
from lib.indicators.volatility import VolatilityCalculator, VolatilityConfig
from lib.indicators.ma_spreads import MASpreadCalculator, MASpreadConfig
from lib.indicators.divergence import DivergenceCalculator, DivergenceConfig


class TestMomentumCalculator:
    """Tests for MomentumCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        return MomentumCalculator()
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price series."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        # Simulate trending prices
        trend = np.linspace(100, 150, 200)
        noise = np.random.normal(0, 2, 200)
        prices = trend + noise
        return pd.Series(prices, index=dates)
    
    def test_basic_momentum(self, calculator, sample_prices):
        """Test basic momentum calculation."""
        momentum = calculator.calculate(sample_prices, period=30)
        
        assert len(momentum) == len(sample_prices)
        assert momentum.isna().sum() == 30  # First 30 values are NaN
        
        # In uptrending market, recent momentum should be positive
        assert momentum.iloc[-1] > 0
    
    def test_12w_momentum(self, calculator, sample_prices):
        """Test 12-week momentum calculation (84-day window; lag=0 to test window only)."""
        momentum = calculator.calculate_12w_momentum(sample_prices, lag=0)
        
        # 84 day period
        assert momentum.isna().sum() == 84
        assert len(momentum) == len(sample_prices)
    
    def test_rank_coins_by_momentum(self, calculator):
        """Test coin ranking by momentum."""
        coin_momentum = {
            'bitcoin': 0.10,
            'ethereum': 0.25,
            'solana': 0.15,
            'chainlink': 0.05,
        }
        
        ranked = calculator.rank_coins_by_momentum(coin_momentum)
        
        assert ranked[0] == 'ethereum'  # Highest
        assert ranked[-1] == 'chainlink'  # Lowest
    
    def test_rank_coins_handles_nan(self, calculator):
        """Test that NaN values are filtered in ranking."""
        coin_momentum = {
            'bitcoin': 0.10,
            'ethereum': np.nan,
            'solana': 0.15,
        }
        
        ranked = calculator.rank_coins_by_momentum(coin_momentum)
        
        assert 'ethereum' not in ranked
        assert len(ranked) == 2
    
    def test_momentum_weights(self, calculator):
        """Test momentum-weighted allocation."""
        coin_momentum = {
            'bitcoin': 0.10,
            'ethereum': 0.20,
            'solana': 0.30,
        }
        
        weights = calculator.calculate_momentum_weights(coin_momentum, top_n=2)
        
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.001
        
        # Solana should have higher weight
        assert weights['solana'] > weights['ethereum']
    
    def test_momentum_weights_equal_for_negative(self, calculator):
        """Test equal weights for all-negative momentum."""
        coin_momentum = {
            'bitcoin': -0.10,
            'ethereum': -0.20,
        }
        
        weights = calculator.calculate_momentum_weights(coin_momentum, top_n=2)
        
        # Should be equal weights
        assert abs(weights['bitcoin'] - weights['ethereum']) < 0.001


class TestVolatilityCalculator:
    """Tests for VolatilityCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        return VolatilityCalculator()
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price series."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        return pd.Series(prices, index=dates)
    
    def test_log_returns(self, calculator, sample_prices):
        """Test log returns calculation."""
        returns = calculator.calculate_log_returns(sample_prices)
        
        assert len(returns) == len(sample_prices)
        assert returns.isna().sum() == 1  # First value is NaN
    
    def test_realized_volatility(self, calculator, sample_prices):
        """Test realized volatility calculation."""
        vol = calculator.calculate_realized_volatility(sample_prices, period=30)
        
        assert len(vol) == len(sample_prices)
        assert vol.isna().sum() == 30  # First 30 values are NaN
        
        # Volatility should be positive
        assert (vol.dropna() > 0).all()
    
    def test_annualized_volatility(self, calculator, sample_prices):
        """Test that volatility is annualized correctly."""
        vol_ann = calculator.calculate_realized_volatility(
            sample_prices, period=30, annualize=True
        )
        vol_raw = calculator.calculate_realized_volatility(
            sample_prices, period=30, annualize=False
        )
        
        # Annualized should be ~sqrt(252) times larger
        ratio = vol_ann.dropna().iloc[-1] / vol_raw.dropna().iloc[-1]
        expected_ratio = np.sqrt(252)
        
        assert abs(ratio - expected_ratio) < 0.1
    
    def test_volatility_regime(self, calculator):
        """Test volatility regime classification."""
        regime_low = calculator.get_current_volatility_regime(0.2)
        regime_normal = calculator.get_current_volatility_regime(0.45)
        regime_high = calculator.get_current_volatility_regime(0.7)
        
        assert regime_low == 'low'
        assert regime_normal == 'normal'
        assert regime_high == 'high'


class TestMASpreadCalculator:
    """Tests for MASpreadCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        return MASpreadCalculator()
    
    @pytest.fixture
    def uptrend_prices(self):
        """Generate uptrending price series."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
        trend = np.linspace(100, 200, 250)
        noise = np.random.normal(0, 2, 250)
        return pd.Series(trend + noise, index=dates)
    
    @pytest.fixture
    def downtrend_prices(self):
        """Generate downtrending price series."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
        trend = np.linspace(200, 100, 250)
        noise = np.random.normal(0, 2, 250)
        return pd.Series(trend + noise, index=dates)
    
    def test_sma_calculation(self, calculator, uptrend_prices):
        """Test SMA calculation."""
        sma = calculator.calculate_sma(uptrend_prices, period=20)
        
        assert len(sma) == len(uptrend_prices)
        assert sma.isna().sum() == 19  # First 19 values are NaN
    
    def test_ema_calculation(self, calculator, uptrend_prices):
        """Test EMA calculation."""
        ema = calculator.calculate_ema(uptrend_prices, period=20)
        
        assert len(ema) == len(uptrend_prices)
        # EMA starts from first value (no NaN)
    
    def test_ma_spread_uptrend(self, calculator, uptrend_prices):
        """Test MA spread in uptrend."""
        spread = calculator.calculate_ma_spread(uptrend_prices, 50, 200)
        
        # In uptrend, short MA > long MA, so spread is positive
        assert spread.dropna().iloc[-1] > 0
    
    def test_ma_spread_downtrend(self, calculator, downtrend_prices):
        """Test MA spread in downtrend."""
        spread = calculator.calculate_ma_spread(downtrend_prices, 50, 200)
        
        # In downtrend, short MA < long MA, so spread is negative
        assert spread.dropna().iloc[-1] < 0
    
    def test_trend_direction(self, calculator, uptrend_prices, downtrend_prices):
        """Test trend direction detection."""
        trend_up = calculator.get_trend_direction(uptrend_prices)
        trend_down = calculator.get_trend_direction(downtrend_prices)
        
        # Uptrend should end with 1
        assert trend_up.dropna().iloc[-1] == 1
        
        # Downtrend should end with -1
        assert trend_down.dropna().iloc[-1] == -1
    
    def test_crossover_detection(self, calculator):
        """Test MA crossover detection."""
        # Create prices that cross over
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Start down, then reverse up
        prices = pd.Series(
            np.concatenate([
                np.linspace(100, 80, 50),
                np.linspace(80, 120, 50)
            ]),
            index=dates
        )
        
        crossovers = calculator.detect_crossover(prices, 10, 30)
        
        # Should have at least one golden cross
        assert crossovers['golden_cross'].sum() >= 1


class TestDivergenceCalculator:
    """Tests for DivergenceCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        return DivergenceCalculator()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample price and volume data."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100))),
            index=dates
        )
        volumes = pd.Series(
            np.abs(np.random.normal(1000, 200, 100)),
            index=dates
        )
        
        return prices, volumes
    
    def test_divergence_calculation(self, calculator, sample_data):
        """Test basic divergence calculation."""
        prices, volumes = sample_data
        
        divergence = calculator.calculate_divergence(prices, volumes, period=30)
        
        assert len(divergence) == len(prices)
        # Should have some non-NaN values
        assert not divergence.dropna().empty
    
    def test_obv_calculation(self, calculator, sample_data):
        """Test OBV calculation."""
        prices, volumes = sample_data
        
        obv = calculator.calculate_obv(prices, volumes)
        
        assert len(obv) == len(prices)
        # OBV should be cumulative sum, so no NaN after first
        assert obv.isna().sum() <= 1
    
    def test_divergence_signal(self, calculator):
        """Test divergence signal detection."""
        divergence = pd.Series([0.0, 2.0, -2.0, 0.5])
        
        signals = calculator.detect_divergence_signal(divergence, threshold=1.5)
        
        assert signals.iloc[1] == 1   # Bullish
        assert signals.iloc[2] == -1  # Bearish
        assert signals.iloc[0] == 0   # Neutral
        assert signals.iloc[3] == 0   # Neutral
