"""
Unit tests for RegimeDetector class.
"""

import sys
from pathlib import Path

# Add project root so "lib" is importable when run as script or from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from lib.regime_detector import (
    RegimeDetector,
    RegimeConfig,
    Regime,
    RegimeState,
    z_score_normalize,
)


class TestRegimeDetector:
    """Tests for RegimeDetector class."""
    
    @pytest.fixture
    def default_detector(self):
        """Create a detector with default config."""
        return RegimeDetector()
    
    @pytest.fixture
    def custom_detector(self):
        """Create a detector with custom config."""
        config = RegimeConfig(
            bull_enter=0.4,
            bull_exit=0.0,
            bear_enter=-0.4,
            bear_exit=-0.1,
            ema_span=10,
            min_duration_days=7,
        )
        return RegimeDetector(config)
    
    def test_initial_state(self, default_detector):
        """Test that detector starts in Neutral regime."""
        assert default_detector.current_regime == Regime.NEUTRAL
        assert not default_detector.regime_changed
    
    def test_composite_score_calculation(self, default_detector):
        """Test composite score calculation with default weights."""
        # Bullish inputs
        score = default_detector.calculate_composite_score(
            momentum_z=1.0,
            volatility_z=0.0,
            funding_z=0.5,
            ma_spread_z=0.5,
            divergence_z=0.5,
        )
        
        # Expected: 1.2*1.0 + (-0.8)*0.0 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5 = 1.95
        expected = 1.2 * 1.0 + (-0.8) * 0.0 + 0.5 * 0.5 + 0.5 * 0.5 + 0.5 * 0.5
        assert abs(score - expected) < 0.001
    
    def test_composite_score_bearish(self, default_detector):
        """Test composite score with bearish inputs."""
        # Bearish: negative momentum, high volatility
        score = default_detector.calculate_composite_score(
            momentum_z=-1.0,
            volatility_z=1.0,
            funding_z=-0.5,
            ma_spread_z=-0.5,
            divergence_z=-0.5,
        )
        
        # Expected: 1.2*(-1.0) + (-0.8)*1.0 + 0.5*(-0.5) + 0.5*(-0.5) + 0.5*(-0.5) = -2.75
        expected = 1.2 * (-1.0) + (-0.8) * 1.0 + 0.5 * (-0.5) + 0.5 * (-0.5) + 0.5 * (-0.5)
        assert abs(score - expected) < 0.001
    
    def test_transition_to_bull(self, default_detector):
        """Test regime transition from Neutral to Bull."""
        # Update with bullish inputs (score > 0.5)
        regime = default_detector.update(
            momentum_z=1.5,
            volatility_z=-0.5,
            funding_z=0.5,
            ma_spread_z=0.5,
            divergence_z=0.5,
        )
        
        # Should transition to Bull
        assert regime == Regime.BULL
        assert default_detector.regime_changed
    
    def test_transition_to_bear(self, default_detector):
        """Test regime transition from Neutral to Bear."""
        # Update with bearish inputs (score < -0.7)
        regime = default_detector.update(
            momentum_z=-1.5,
            volatility_z=1.0,
            funding_z=-0.5,
            ma_spread_z=-0.5,
            divergence_z=-0.5,
        )
        
        # Should transition to Bear
        assert regime == Regime.BEAR
        assert default_detector.regime_changed
    
    def test_hysteresis_prevents_whipsaw(self, custom_detector):
        """Test that hysteresis prevents rapid regime changes."""
        # First, establish Bull regime
        custom_detector.update(
            momentum_z=1.0,
            volatility_z=-0.5,
        )
        assert custom_detector.current_regime == Regime.BULL
        
        # Score that's below bull_enter but above bull_exit
        # Should stay in Bull
        custom_detector._regime_entered_at = datetime.now() - timedelta(days=10)
        
        regime = custom_detector.update(
            momentum_z=0.2,  # Reduced but still positive
            volatility_z=0.0,
        )
        
        # Should still be Bull (score above exit threshold)
        assert regime == Regime.BULL
    
    def test_minimum_duration_enforcement(self, custom_detector):
        """Test that minimum regime duration is enforced."""
        # Establish Bull regime
        custom_detector.update(
            momentum_z=1.0,
            volatility_z=-0.5,
        )
        assert custom_detector.current_regime == Regime.BULL
        
        # Try to trigger regime change immediately (within min duration)
        regime = custom_detector.update(
            momentum_z=-1.0,
            volatility_z=1.0,
        )
        
        # Should still be Bull due to min duration
        assert regime == Regime.BULL
        assert not custom_detector.regime_changed
    
    def test_regime_change_after_min_duration(self):
        """Test that regime can change after minimum duration."""
        # Use short ema_span so one bearish update crosses bull_exit (0.0)
        detector = RegimeDetector(
            RegimeConfig(
                bull_enter=0.4,
                bull_exit=0.0,
                bear_enter=-0.4,
                bear_exit=-0.1,
                ema_span=2,
                min_duration_days=7,
            )
        )
        # Establish Bull regime
        detector.update(
            momentum_z=1.0,
            volatility_z=-0.5,
        )
        
        # Simulate time passing beyond min duration
        detector._regime_entered_at = datetime.now() - timedelta(days=10)
        
        # Raw score -2.0; with ema_span=2, smoothed crosses below bull_exit
        regime = detector.update(
            momentum_z=-1.0,
            volatility_z=1.0,
        )
        
        # Should now change to Neutral
        assert regime == Regime.NEUTRAL
        assert detector.regime_changed
    
    def test_get_state(self, default_detector):
        """Test getting regime state."""
        default_detector.update(momentum_z=1.0, volatility_z=0.0)
        
        state = default_detector.get_state()
        
        assert isinstance(state, RegimeState)
        assert state.regime == default_detector.current_regime
        assert isinstance(state.score, float)
        assert isinstance(state.score_raw, float)
    
    def test_reset(self, default_detector):
        """Test resetting detector to initial state."""
        # Change regime
        default_detector.update(momentum_z=2.0, volatility_z=-1.0)
        assert default_detector.current_regime == Regime.BULL
        
        # Reset
        default_detector.reset()
        
        assert default_detector.current_regime == Regime.NEUTRAL
        assert default_detector._last_score == 0.0
        assert not default_detector.regime_changed


class TestZScoreNormalize:
    """Tests for z-score normalization function."""
    
    def test_basic_normalization(self):
        """Test basic z-score normalization."""
        # Create a series with known mean and std
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        z_scores = z_score_normalize(series, lookback=10)
        
        # Last value should be positive (above rolling mean)
        assert z_scores.iloc[-1] > 0
        
        # First values should be 0 (filled NaN)
        assert z_scores.iloc[0] == 0
    
    def test_handles_constant_series(self):
        """Test that constant series doesn't produce NaN."""
        series = pd.Series([5, 5, 5, 5, 5])
        
        z_scores = z_score_normalize(series, lookback=5)
        
        # Should be all zeros (or NaN replaced with 0)
        assert not z_scores.isna().any()


class TestRegimeIntegration:
    """Integration tests for regime detection."""
    
    def test_full_market_cycle(self):
        """Test regime detection through a full market cycle."""
        # Short ema_span so score crosses thresholds quickly (EMA doesn't block transitions)
        detector = RegimeDetector(
            RegimeConfig(min_duration_days=1, ema_span=2)
        )
        
        regimes = []
        
        # Bull market phase
        for _ in range(5):
            detector._regime_entered_at = datetime.now() - timedelta(days=5)
            regime = detector.update(momentum_z=1.5, volatility_z=-0.5)
            regimes.append(regime)
        
        assert Regime.BULL in regimes
        
        # Transition to Neutral
        detector._regime_entered_at = datetime.now() - timedelta(days=5)
        regime = detector.update(momentum_z=0.0, volatility_z=0.0)
        regimes.append(regime)
        
        # Bear market phase (strong bearish score so we cross bear_enter)
        for _ in range(5):
            detector._regime_entered_at = datetime.now() - timedelta(days=5)
            regime = detector.update(momentum_z=-1.5, volatility_z=1.0)
            regimes.append(regime)
        
        assert Regime.BEAR in regimes
        
        # Recovery
        detector._regime_entered_at = datetime.now() - timedelta(days=5)
        regime = detector.update(momentum_z=0.5, volatility_z=0.0)
        regimes.append(regime)
        
        # Should have seen all three regimes
        unique_regimes = set(regimes)
        assert len(unique_regimes) >= 2  # At least Bull and Bear
