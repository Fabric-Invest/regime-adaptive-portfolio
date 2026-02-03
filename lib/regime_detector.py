"""
Regime Detector

Implements the regime detection logic with composite scoring,
EMA smoothing, and hysteresis-based state machine.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classifications."""
    BULL = "Bull"
    NEUTRAL = "Neutral"
    BEAR = "Bear"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Composite score weights (optimized from forward-walk validation)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 1.2,
        'volatility': -0.8,  # Negative: high volatility = bearish
        'funding': 0.5,
        'ma_spread': 0.5,
        'divergence': 0.5,
    })
    
    # Hysteresis thresholds (optimized from forward-walk validation)
    bull_enter: float = 0.5
    bull_exit: float = 0.1
    bear_enter: float = -0.7
    bear_exit: float = -0.3
    
    # Smoothing parameters
    ema_span: int = 21
    
    # Minimum regime duration (prevents whipsawing)
    min_duration_days: int = 14


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    regime: Regime
    score: float
    score_raw: float
    entered_at: datetime
    days_in_regime: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'regime': self.regime.value,
            'score': round(self.score, 4),
            'score_raw': round(self.score_raw, 4),
            'entered_at': self.entered_at.isoformat() if self.entered_at else None,
            'days_in_regime': self.days_in_regime,
        }


class RegimeDetector:
    """
    Detects market regimes using a composite scoring approach.
    
    The detector combines multiple indicators into a single composite score,
    applies EMA smoothing, and uses a hysteresis state machine to determine
    the current market regime (Bull, Neutral, or Bear).
    
    Features:
    - Composite score from momentum, volatility, funding, MA spreads, divergence
    - EMA smoothing to reduce noise
    - Hysteresis thresholds to prevent whipsawing
    - Minimum regime duration enforcement
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        
        # Current state
        self._current_regime = Regime.NEUTRAL
        self._regime_entered_at: Optional[datetime] = None
        self._last_score: float = 0.0
        self._last_score_raw: float = 0.0
        self._score_history: pd.Series = pd.Series(dtype=float)
        self._regime_changed: bool = False
        
    @property
    def current_regime(self) -> Regime:
        """Get the current regime."""
        return self._current_regime
    
    @property
    def regime_changed(self) -> bool:
        """Check if regime changed on last update."""
        return self._regime_changed
    
    def get_state(self) -> RegimeState:
        """Get the current regime state with metadata."""
        days_in_regime = 0
        if self._regime_entered_at:
            days_in_regime = (datetime.now() - self._regime_entered_at).days
        
        return RegimeState(
            regime=self._current_regime,
            score=self._last_score,
            score_raw=self._last_score_raw,
            entered_at=self._regime_entered_at,
            days_in_regime=days_in_regime,
        )
    
    def calculate_composite_score(
        self,
        momentum_z: float,
        volatility_z: float,
        funding_z: float = 0.0,
        ma_spread_z: float = 0.0,
        divergence_z: float = 0.0,
    ) -> float:
        """
        Calculate the raw composite regime score.
        
        All inputs should be z-score normalized values.
        
        Args:
            momentum_z: Average 12-week momentum (z-score)
            volatility_z: Average realized volatility (z-score)
            funding_z: Average funding rate (z-score)
            ma_spread_z: Average MA spread (z-score)
            divergence_z: Average volume-price divergence (z-score)
            
        Returns:
            Raw composite score (unbounded)
        """
        w = self.config.weights
        
        score = (
            w['momentum'] * momentum_z
            + w['volatility'] * volatility_z  # Note: weight is negative
            + w['funding'] * funding_z
            + w['ma_spread'] * ma_spread_z
            + w['divergence'] * divergence_z
        )
        
        return score
    
    def apply_ema_smoothing(
        self,
        score: float,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Apply EMA smoothing to the composite score.
        
        Args:
            score: Raw composite score
            timestamp: Timestamp for the score (default: now)
            
        Returns:
            EMA-smoothed score
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        self._score_history[timestamp] = score
        
        # Apply EMA
        if len(self._score_history) < 2:
            return score
        
        smoothed = self._score_history.ewm(
            span=self.config.ema_span, 
            adjust=False
        ).mean()
        
        return smoothed.iloc[-1]
    
    def _can_change_regime(self) -> bool:
        """Check if enough time has passed to allow regime change."""
        if self._regime_entered_at is None:
            return True
        
        days_in_regime = (datetime.now() - self._regime_entered_at).days
        return days_in_regime >= self.config.min_duration_days
    
    def _apply_hysteresis(self, score: float) -> Regime:
        """
        Apply hysteresis logic to determine regime from score.
        
        Uses different thresholds for entering vs exiting each regime
        to prevent rapid switching (whipsawing).
        
        Args:
            score: EMA-smoothed composite score
            
        Returns:
            Target regime based on hysteresis rules
        """
        current = self._current_regime
        
        if current == Regime.NEUTRAL:
            # From Neutral: need to cross entry thresholds
            if score > self.config.bull_enter:
                return Regime.BULL
            elif score < self.config.bear_enter:
                return Regime.BEAR
            else:
                return Regime.NEUTRAL
                
        elif current == Regime.BULL:
            # From Bull: need to drop below exit threshold
            if score < self.config.bull_exit:
                return Regime.NEUTRAL
            else:
                return Regime.BULL
                
        elif current == Regime.BEAR:
            # From Bear: need to rise above exit threshold
            if score > self.config.bear_exit:
                return Regime.NEUTRAL
            else:
                return Regime.BEAR
        
        return current
    
    def update(
        self,
        momentum_z: float,
        volatility_z: float,
        funding_z: float = 0.0,
        ma_spread_z: float = 0.0,
        divergence_z: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Regime:
        """
        Update the regime detector with new indicator values.
        
        Args:
            momentum_z: Average 12-week momentum (z-score)
            volatility_z: Average realized volatility (z-score)
            funding_z: Average funding rate (z-score)
            ma_spread_z: Average MA spread (z-score)
            divergence_z: Average volume-price divergence (z-score)
            timestamp: Current timestamp
            
        Returns:
            Current regime after update
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate raw composite score
        score_raw = self.calculate_composite_score(
            momentum_z=momentum_z,
            volatility_z=volatility_z,
            funding_z=funding_z,
            ma_spread_z=ma_spread_z,
            divergence_z=divergence_z,
        )
        self._last_score_raw = score_raw
        
        # Apply EMA smoothing
        score = self.apply_ema_smoothing(score_raw, timestamp)
        self._last_score = score
        
        # Determine target regime using hysteresis
        target_regime = self._apply_hysteresis(score)
        
        # Check if regime change is allowed (min duration)
        self._regime_changed = False
        
        if target_regime != self._current_regime:
            if self._can_change_regime():
                logger.info(
                    f"Regime change: {self._current_regime.value} -> {target_regime.value} "
                    f"(score: {score:.3f})"
                )
                self._current_regime = target_regime
                self._regime_entered_at = timestamp
                self._regime_changed = True
            else:
                days_remaining = self.config.min_duration_days - (
                    datetime.now() - self._regime_entered_at
                ).days
                logger.debug(
                    f"Regime change blocked: {days_remaining} days remaining in "
                    f"minimum duration"
                )
        
        return self._current_regime
    
    def update_from_dataframe(
        self,
        data: pd.DataFrame,
        coins: list,
    ) -> pd.Series:
        """
        Update regime detection from a DataFrame with indicator columns.
        
        This is useful for batch processing historical data.
        
        Args:
            data: DataFrame with z-score normalized indicator columns
            coins: List of coin names
            
        Returns:
            Series of regime values for each timestamp
        """
        regimes = []
        
        # Get column names
        momentum_cols = [f"{c}_momentum_12w_z" for c in coins if f"{c}_momentum_12w_z" in data.columns]
        vol_cols = [f"{c}_realized_vol_30d_z" for c in coins if f"{c}_realized_vol_30d_z" in data.columns]
        funding_cols = [f"{c}_fundingRateDaily_z" for c in coins if f"{c}_fundingRateDaily_z" in data.columns]
        ma_cols = [f"{c}_ma_spread_50_200_z" for c in coins if f"{c}_ma_spread_50_200_z" in data.columns]
        div_cols = [f"{c}_divergence_vol_price_div_30d_z" for c in coins if f"{c}_divergence_vol_price_div_30d_z" in data.columns]
        
        for timestamp, row in data.iterrows():
            # Calculate averages
            momentum_z = row[momentum_cols].mean() if momentum_cols else 0.0
            vol_z = row[vol_cols].mean() if vol_cols else 0.0
            funding_z = row[funding_cols].mean() if funding_cols else 0.0
            ma_z = row[ma_cols].mean() if ma_cols else 0.0
            div_z = row[div_cols].mean() if div_cols else 0.0
            
            regime = self.update(
                momentum_z=momentum_z,
                volatility_z=vol_z,
                funding_z=funding_z,
                ma_spread_z=ma_z,
                divergence_z=div_z,
                timestamp=timestamp,
            )
            
            regimes.append(regime.value)
        
        return pd.Series(regimes, index=data.index, name='regime')
    
    def get_regime_string(self) -> str:
        """Get current regime as string."""
        return self._current_regime.value
    
    def reset(self):
        """Reset the detector to initial state."""
        self._current_regime = Regime.NEUTRAL
        self._regime_entered_at = None
        self._last_score = 0.0
        self._last_score_raw = 0.0
        self._score_history = pd.Series(dtype=float)
        self._regime_changed = False
        
        logger.info("Regime detector reset to initial state")


def z_score_normalize(
    series: pd.Series,
    lookback: int = 252
) -> pd.Series:
    """
    Calculate rolling z-score normalization.
    
    Args:
        series: Input series
        lookback: Lookback period for mean/std calculation
        
    Returns:
        Z-score normalized series
    """
    rolling_mean = series.rolling(window=lookback, min_periods=1).mean()
    rolling_std = series.rolling(window=lookback, min_periods=1).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_score = (series - rolling_mean) / rolling_std
    
    return z_score.fillna(0)
