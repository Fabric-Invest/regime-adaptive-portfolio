"""
Portfolio Allocator

Implements the portfolio allocation logic based on detected regimes.
Maps regimes to allocation parameters and calculates momentum-weighted
allocations across the portfolio.
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from lib.regime_detector import Regime

logger = logging.getLogger(__name__)


class BearPhase(Enum):
    """Bear market sub-phases."""
    CRASH = "Bear-Crash"      # Severe decline, minimal exposure
    BASE = "Bear-Base"        # Bottoming out, cautious exposure
    TRANSITION = "Bear-Transition"  # Potential recovery, increasing exposure


@dataclass
class AllocationConfig:
    """Configuration for portfolio allocation."""
    # Core coins in priority order
    core_coins: List[str] = field(default_factory=lambda: [
        'bitcoin', 'ethereum', 'solana', 'chainlink', 'aave'
    ])
    
    # Bull regime allocation
    bull_top_n: int = 3
    bull_exposure: float = 0.95
    
    # Neutral regime allocation
    neutral_top_n: int = 2
    neutral_exposure: float = 0.75
    
    # Bear regime sub-phases
    bear_crash_top_n: int = 1
    bear_crash_exposure: float = 0.25
    bear_base_top_n: int = 2
    bear_base_exposure: float = 0.45
    bear_transition_top_n: int = 2
    bear_transition_exposure: float = 0.65
    
    # Bear phase thresholds
    bear_crash_threshold: float = -0.05  # BTC 12-week momentum threshold
    
    # Funding rate tilt (small adjustment based on funding)
    funding_tilt: float = 0.025
    
    # Minimum allocation per coin
    min_allocation: float = 0.0
    
    # Maximum allocation per coin (for diversification)
    max_allocation: float = 0.5


@dataclass
class Allocation:
    """Portfolio allocation result."""
    weights: Dict[str, float]  # coin -> weight
    exposure: float            # Total exposure (sum of weights)
    regime: Regime
    bear_phase: Optional[BearPhase] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'weights': {k: round(v, 4) for k, v in self.weights.items()},
            'exposure': round(self.exposure, 4),
            'regime': self.regime.value,
            'bear_phase': self.bear_phase.value if self.bear_phase else None,
        }
    
    def get_non_zero_weights(self) -> Dict[str, float]:
        """Get only non-zero allocations."""
        return {k: v for k, v in self.weights.items() if v > 0}


class PortfolioAllocator:
    """
    Allocates portfolio based on regime and coin momentum.
    
    The allocator maps the current regime to allocation parameters
    (top_n coins, total exposure) and then selects and weights
    the top N coins by their 12-week momentum.
    
    Features:
    - Regime-specific allocation rules
    - Bear market sub-phases (crash, base, transition)
    - Momentum-weighted coin selection
    - Funding rate tilt adjustment
    """
    
    def __init__(self, config: Optional[AllocationConfig] = None):
        self.config = config or AllocationConfig()
    
    def _determine_bear_phase(
        self,
        btc_momentum: float
    ) -> BearPhase:
        """
        Determine the bear market sub-phase based on BTC momentum.
        
        Args:
            btc_momentum: Bitcoin's 12-week momentum
            
        Returns:
            Bear phase classification
        """
        if btc_momentum < self.config.bear_crash_threshold:
            return BearPhase.CRASH
        elif btc_momentum < 0:
            return BearPhase.BASE
        else:
            return BearPhase.TRANSITION
    
    def _get_allocation_params(
        self,
        regime: Regime,
        btc_momentum: float = 0.0
    ) -> Tuple[int, float, Optional[BearPhase]]:
        """
        Get allocation parameters for the given regime.
        
        Args:
            regime: Current market regime
            btc_momentum: Bitcoin's 12-week momentum (for bear phases)
            
        Returns:
            Tuple of (top_n, exposure, bear_phase)
        """
        if regime == Regime.BULL:
            return (
                self.config.bull_top_n,
                self.config.bull_exposure,
                None
            )
        elif regime == Regime.NEUTRAL:
            return (
                self.config.neutral_top_n,
                self.config.neutral_exposure,
                None
            )
        elif regime == Regime.BEAR:
            bear_phase = self._determine_bear_phase(btc_momentum)
            
            if bear_phase == BearPhase.CRASH:
                return (
                    self.config.bear_crash_top_n,
                    self.config.bear_crash_exposure,
                    bear_phase
                )
            elif bear_phase == BearPhase.BASE:
                return (
                    self.config.bear_base_top_n,
                    self.config.bear_base_exposure,
                    bear_phase
                )
            else:  # TRANSITION
                return (
                    self.config.bear_transition_top_n,
                    self.config.bear_transition_exposure,
                    bear_phase
                )
        
        # Default (should not reach here)
        return (2, 0.5, None)
    
    def _rank_coins_by_momentum(
        self,
        coin_momentum: Dict[str, float]
    ) -> List[str]:
        """
        Rank coins by momentum in descending order.
        
        Args:
            coin_momentum: Dictionary of coin -> 12-week momentum
            
        Returns:
            List of coins sorted by momentum (highest first)
        """
        # Filter out NaN values
        valid = {
            coin: mom for coin, mom in coin_momentum.items()
            if not pd.isna(mom) and coin in self.config.core_coins
        }
        
        if not valid:
            return self.config.core_coins[:1]  # Return BTC as fallback
        
        return sorted(valid.keys(), key=lambda x: valid[x], reverse=True)
    
    def _calculate_momentum_weights(
        self,
        coin_momentum: Dict[str, float],
        top_coins: List[str]
    ) -> Dict[str, float]:
        """
        Calculate momentum-weighted allocations for top coins.
        
        Weights are proportional to positive momentum values.
        Equal weights are used if all momentum values are non-positive.
        
        Args:
            coin_momentum: Dictionary of coin -> momentum
            top_coins: List of top N coins to allocate to
            
        Returns:
            Dictionary of coin -> weight (weights sum to 1)
        """
        if not top_coins:
            return {}
        
        # Get momentum values for top coins
        momentum_values = np.array([
            coin_momentum.get(coin, 0) for coin in top_coins
        ])
        
        # Ensure non-negative weights
        weights = np.maximum(momentum_values, 0)
        
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Equal weights if all momentum is non-positive
            weights = np.ones(len(top_coins)) / len(top_coins)
        
        return dict(zip(top_coins, weights))
    
    def _apply_funding_tilt(
        self,
        weights: Dict[str, float],
        funding_rates: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply small adjustment based on funding rates.
        
        Positive funding (longs paying) -> slight increase
        Negative funding (shorts paying) -> slight decrease
        
        Args:
            weights: Current allocation weights
            funding_rates: Dictionary of coin -> funding rate
            
        Returns:
            Adjusted weights
        """
        if not funding_rates:
            return weights
        
        adjusted = weights.copy()
        
        for coin in adjusted:
            if coin in funding_rates:
                fr = funding_rates[coin]
                # Add/subtract small tilt based on funding sign
                tilt = self.config.funding_tilt * np.sign(fr)
                adjusted[coin] = max(0, adjusted[coin] + tilt)
        
        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def _clip_allocations(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Clip allocations to min/max bounds.
        
        Does not re-normalize so that total exposure is preserved
        (caller scales by exposure; re-normalizing would undo that).
        
        Args:
            weights: Allocation weights (already exposure-scaled)
            
        Returns:
            Clipped weights (sum may be <= original if clipping reduced any)
        """
        clipped = {}
        
        for coin, weight in weights.items():
            clipped[coin] = np.clip(
                weight,
                self.config.min_allocation,
                self.config.max_allocation
            )
        
        return clipped
    
    def get_allocations(
        self,
        regime: Regime,
        coin_momentum: Dict[str, float],
        btc_momentum: float = 0.0,
        funding_rates: Optional[Dict[str, float]] = None,
    ) -> Allocation:
        """
        Calculate portfolio allocations for the given regime.
        
        Args:
            regime: Current market regime
            coin_momentum: Dictionary of coin -> 12-week momentum
            btc_momentum: Bitcoin's 12-week momentum (for bear phases)
            funding_rates: Optional funding rates for tilt adjustment
            
        Returns:
            Allocation object with weights and metadata
        """
        # Get allocation parameters for regime
        top_n, exposure, bear_phase = self._get_allocation_params(
            regime, btc_momentum
        )
        
        logger.info(
            f"Allocation params - Regime: {regime.value}, "
            f"Top N: {top_n}, Exposure: {exposure:.0%}"
            + (f", Bear Phase: {bear_phase.value}" if bear_phase else "")
        )
        
        # Rank coins by momentum
        ranked_coins = self._rank_coins_by_momentum(coin_momentum)
        top_coins = ranked_coins[:top_n]
        
        logger.debug(f"Top {top_n} coins by momentum: {top_coins}")
        
        # Calculate momentum-weighted allocations
        relative_weights = self._calculate_momentum_weights(
            coin_momentum, top_coins
        )
        
        # Apply funding tilt if provided
        if funding_rates:
            relative_weights = self._apply_funding_tilt(
                relative_weights, funding_rates
            )
        
        # Apply exposure scaling
        weights = {
            coin: weight * exposure
            for coin, weight in relative_weights.items()
        }
        
        # Ensure all core coins have an entry (even if 0)
        for coin in self.config.core_coins:
            if coin not in weights:
                weights[coin] = 0.0
        
        # Clip to min/max bounds
        weights = self._clip_allocations(weights)
        
        # Recalculate actual exposure after clipping
        actual_exposure = sum(weights.values())
        
        return Allocation(
            weights=weights,
            exposure=actual_exposure,
            regime=regime,
            bear_phase=bear_phase,
        )
    
    def calculate_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        min_trade_pct: float = 0.01,
    ) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance from current to target weights.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            min_trade_pct: Minimum trade size as percentage of portfolio
            
        Returns:
            Dictionary of coin -> trade amount (positive = buy, negative = sell)
        """
        trades = {}
        
        # Get all coins
        all_coins = set(current_weights.keys()) | set(target_weights.keys())
        
        for coin in all_coins:
            current = current_weights.get(coin, 0)
            target = target_weights.get(coin, 0)
            
            diff = target - current
            
            # Only trade if difference exceeds minimum
            if abs(diff) >= min_trade_pct:
                trade_value = diff * portfolio_value
                trades[coin] = trade_value
        
        return trades
    
    def get_cash_allocation(self, allocation: Allocation) -> float:
        """
        Calculate the cash (uninvested) portion of the portfolio.
        
        Args:
            allocation: Current allocation
            
        Returns:
            Cash allocation as decimal (e.g., 0.25 = 25% cash)
        """
        return 1.0 - allocation.exposure


class MockPortfolioAllocator(PortfolioAllocator):
    """
    Mock allocator for testing.
    
    Returns fixed allocations regardless of regime.
    """
    
    def __init__(
        self,
        fixed_weights: Optional[Dict[str, float]] = None,
        config: Optional[AllocationConfig] = None
    ):
        super().__init__(config)
        self._fixed_weights = fixed_weights or {
            'bitcoin': 0.4,
            'ethereum': 0.3,
            'solana': 0.15,
            'chainlink': 0.1,
            'aave': 0.05,
        }
    
    def get_allocations(
        self,
        regime: Regime,
        coin_momentum: Dict[str, float],
        btc_momentum: float = 0.0,
        funding_rates: Optional[Dict[str, float]] = None,
    ) -> Allocation:
        """Return fixed allocations."""
        return Allocation(
            weights=self._fixed_weights.copy(),
            exposure=sum(self._fixed_weights.values()),
            regime=regime,
            bear_phase=None,
        )
