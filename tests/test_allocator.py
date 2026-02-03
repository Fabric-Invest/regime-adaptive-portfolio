"""
Unit tests for PortfolioAllocator class.
"""

import sys
from pathlib import Path

# Add project root so "lib" is importable when run as script or from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import numpy as np

from lib.portfolio_allocator import (
    PortfolioAllocator,
    AllocationConfig,
    Allocation,
    BearPhase,
)
from lib.regime_detector import Regime


class TestPortfolioAllocator:
    """Tests for PortfolioAllocator class."""
    
    @pytest.fixture
    def default_allocator(self):
        """Create an allocator with default config."""
        return PortfolioAllocator()
    
    @pytest.fixture
    def custom_allocator(self):
        """Create an allocator with custom config."""
        config = AllocationConfig(
            core_coins=['bitcoin', 'ethereum', 'solana'],
            bull_top_n=2,
            bull_exposure=0.90,
            neutral_top_n=2,
            neutral_exposure=0.70,
            bear_crash_top_n=1,
            bear_crash_exposure=0.20,
        )
        return PortfolioAllocator(config)
    
    @pytest.fixture
    def sample_momentum(self):
        """Sample momentum values for testing."""
        return {
            'bitcoin': 0.15,
            'ethereum': 0.25,
            'solana': 0.35,
            'chainlink': 0.10,
            'aave': 0.05,
        }
    
    def test_bull_allocation(self, default_allocator, sample_momentum):
        """Test allocation in Bull regime."""
        allocation = default_allocator.get_allocations(
            regime=Regime.BULL,
            coin_momentum=sample_momentum,
        )
        
        assert allocation.regime == Regime.BULL
        assert allocation.bear_phase is None
        
        # Bull should have 3 coins with 95% exposure
        non_zero = allocation.get_non_zero_weights()
        assert len(non_zero) == 3
        assert abs(sum(non_zero.values()) - 0.95) < 0.1  # Allow for clipping adjustments
        
        # Top 3 by momentum should be: solana, ethereum, bitcoin
        assert 'solana' in non_zero
        assert 'ethereum' in non_zero
        assert 'bitcoin' in non_zero
    
    def test_neutral_allocation(self, default_allocator, sample_momentum):
        """Test allocation in Neutral regime."""
        allocation = default_allocator.get_allocations(
            regime=Regime.NEUTRAL,
            coin_momentum=sample_momentum,
        )
        
        assert allocation.regime == Regime.NEUTRAL
        
        # Neutral should have 2 coins with 75% exposure
        non_zero = allocation.get_non_zero_weights()
        assert len(non_zero) == 2
        assert abs(sum(non_zero.values()) - 0.75) < 0.1
        
        # Top 2: solana, ethereum
        assert 'solana' in non_zero
        assert 'ethereum' in non_zero
    
    def test_bear_crash_allocation(self, default_allocator, sample_momentum):
        """Test allocation in Bear-Crash phase."""
        allocation = default_allocator.get_allocations(
            regime=Regime.BEAR,
            coin_momentum=sample_momentum,
            btc_momentum=-0.10,  # Below crash threshold (-0.05)
        )
        
        assert allocation.regime == Regime.BEAR
        assert allocation.bear_phase == BearPhase.CRASH
        
        # Bear crash should have 1 coin with 25% exposure
        non_zero = allocation.get_non_zero_weights()
        assert len(non_zero) == 1
        assert abs(sum(non_zero.values()) - 0.25) < 0.1
    
    def test_bear_base_allocation(self, default_allocator, sample_momentum):
        """Test allocation in Bear-Base phase."""
        allocation = default_allocator.get_allocations(
            regime=Regime.BEAR,
            coin_momentum=sample_momentum,
            btc_momentum=-0.03,  # Above crash threshold but negative
        )
        
        assert allocation.regime == Regime.BEAR
        assert allocation.bear_phase == BearPhase.BASE
        
        # Bear base should have 2 coins with 45% exposure
        non_zero = allocation.get_non_zero_weights()
        assert len(non_zero) == 2
        assert abs(sum(non_zero.values()) - 0.45) < 0.1
    
    def test_bear_transition_allocation(self, default_allocator, sample_momentum):
        """Test allocation in Bear-Transition phase."""
        allocation = default_allocator.get_allocations(
            regime=Regime.BEAR,
            coin_momentum=sample_momentum,
            btc_momentum=0.02,  # Positive (recovering)
        )
        
        assert allocation.regime == Regime.BEAR
        assert allocation.bear_phase == BearPhase.TRANSITION
        
        # Bear transition should have 2 coins with 65% exposure
        non_zero = allocation.get_non_zero_weights()
        assert len(non_zero) == 2
        assert abs(sum(non_zero.values()) - 0.65) < 0.1
    
    def test_momentum_weighting(self, default_allocator):
        """Test that allocations are momentum-weighted."""
        momentum = {
            'bitcoin': 0.10,
            'ethereum': 0.20,
            'solana': 0.30,
        }
        
        allocation = default_allocator.get_allocations(
            regime=Regime.BULL,
            coin_momentum=momentum,
        )
        
        weights = allocation.weights
        
        # Solana should have highest weight (highest momentum)
        assert weights.get('solana', 0) >= weights.get('ethereum', 0)
        assert weights.get('ethereum', 0) >= weights.get('bitcoin', 0)
    
    def test_equal_weights_for_zero_momentum(self, default_allocator):
        """Test equal weighting when all momentum is zero or negative."""
        momentum = {
            'bitcoin': -0.10,
            'ethereum': -0.05,
            'solana': 0.0,
        }
        
        allocation = default_allocator.get_allocations(
            regime=Regime.NEUTRAL,
            coin_momentum=momentum,
        )
        
        non_zero = allocation.get_non_zero_weights()
        weights = list(non_zero.values())
        
        # Should be approximately equal (within tolerance)
        if len(weights) > 1:
            assert abs(weights[0] - weights[1]) < 0.2
    
    def test_funding_tilt_adjustment(self, default_allocator, sample_momentum):
        """Test funding rate tilt adjustment."""
        funding_rates = {
            'solana': 0.001,  # Positive funding
            'ethereum': -0.001,  # Negative funding
        }
        
        allocation = default_allocator.get_allocations(
            regime=Regime.BULL,
            coin_momentum=sample_momentum,
            funding_rates=funding_rates,
        )
        
        # Funding tilt should slightly adjust weights
        # This is a basic sanity check
        assert sum(allocation.weights.values()) > 0
    
    def test_calculate_rebalance_trades(self, default_allocator):
        """Test rebalance trade calculation."""
        current = {
            'bitcoin': 0.40,
            'ethereum': 0.30,
            'solana': 0.20,
        }
        target = {
            'bitcoin': 0.30,
            'ethereum': 0.35,
            'solana': 0.25,
        }
        
        trades = default_allocator.calculate_rebalance_trades(
            current_weights=current,
            target_weights=target,
            portfolio_value=10000,
        )
        
        # Bitcoin should be sold (negative)
        assert trades.get('bitcoin', 0) < 0
        
        # Ethereum and Solana should be bought (positive)
        assert trades.get('ethereum', 0) > 0
        assert trades.get('solana', 0) > 0
    
    def test_rebalance_trades_minimum_threshold(self, default_allocator):
        """Test that small rebalances are filtered out."""
        current = {'bitcoin': 0.40}
        target = {'bitcoin': 0.405}  # Only 0.5% difference
        
        trades = default_allocator.calculate_rebalance_trades(
            current_weights=current,
            target_weights=target,
            portfolio_value=10000,
            min_trade_pct=0.01,  # 1% minimum
        )
        
        # Should be no trades (below threshold)
        assert 'bitcoin' not in trades
    
    def test_get_cash_allocation(self, default_allocator, sample_momentum):
        """Test cash allocation calculation."""
        allocation = default_allocator.get_allocations(
            regime=Regime.NEUTRAL,
            coin_momentum=sample_momentum,
        )
        
        cash = default_allocator.get_cash_allocation(allocation)
        
        # Neutral has 75% exposure, so 25% cash
        assert abs(cash - 0.25) < 0.1
    
    def test_allocation_to_dict(self, default_allocator, sample_momentum):
        """Test allocation serialization."""
        allocation = default_allocator.get_allocations(
            regime=Regime.BULL,
            coin_momentum=sample_momentum,
        )
        
        d = allocation.to_dict()
        
        assert 'weights' in d
        assert 'exposure' in d
        assert 'regime' in d
        assert d['regime'] == 'Bull'


class TestAllocationConfig:
    """Tests for AllocationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AllocationConfig()
        
        assert config.bull_top_n == 3
        assert config.bull_exposure == 0.95
        assert config.neutral_top_n == 2
        assert config.neutral_exposure == 0.75
        assert len(config.core_coins) == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AllocationConfig(
            core_coins=['bitcoin', 'ethereum'],
            bull_top_n=2,
            bull_exposure=1.0,
        )
        
        assert len(config.core_coins) == 2
        assert config.bull_top_n == 2
        assert config.bull_exposure == 1.0
