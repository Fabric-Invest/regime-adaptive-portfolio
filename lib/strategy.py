"""
Regime-Adaptive Portfolio Strategy

Main strategy class that integrates all components with the Fabric SDK.
Implements a Type-2 portfolio rebalancing strategy based on market regime detection.
"""

import os
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

# Fabric SDK imports (graceful fallback for development)
try:
    from fabric_sdk import Strategy, cached, FabricClient, create_publisher
    FABRIC_SDK_AVAILABLE = True
except ImportError:
    FABRIC_SDK_AVAILABLE = False
    # Mock base class for development
    class Strategy:
        def __init__(self):
            self.params = {}
        def run(self):
            pass
        def log(self, msg, level="INFO"):
            print(f"[{level}] {msg}")
    def cached(func):
        return func

from lib.regime_detector import RegimeDetector, RegimeConfig, Regime, z_score_normalize
from lib.portfolio_allocator import PortfolioAllocator, AllocationConfig, Allocation
from lib.data.funding_fetcher import FundingFetcher, MockFundingFetcher
from lib.data.price_fetcher import PriceFetcher, MockPriceFetcher
from lib.indicators.momentum import MomentumCalculator
from lib.indicators.volatility import VolatilityCalculator
from lib.indicators.ma_spreads import MASpreadCalculator
from lib.indicators.divergence import DivergenceCalculator

logger = logging.getLogger(__name__)


class RegimeAdaptiveStrategy(Strategy):
    """
    Regime-Adaptive Portfolio Strategy.
    
    This strategy dynamically allocates across 5 core crypto assets
    (BTC, ETH, SOL, LINK, AAVE) based on detected market regimes.
    
    Key Features:
    - Regime detection using composite scoring with hysteresis
    - Momentum-weighted allocation within each regime
    - Bear market sub-phases (crash, base, transition)
    - Weekly rebalancing + regime change triggers
    - DEX execution on EVM (0x aggregator)
    
    Fabric SDK Integration:
    - Reports equity, trades, and metrics to platform
    - Sends notifications on regime changes and rebalances
    - Supports paper trading and live DEX execution
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the strategy.
        
        Args:
            config_path: Path to config.yaml (default: ./config.yaml)
        """
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize strategy parameters
        self.params = {
            'name': self.config.get('name', 'Regime-Adaptive Portfolio'),
            'version': self.config.get('version', '1.0.0'),
            'timeframe': self.config.get('timeframe', '1d'),
        }
        
        # Core coins from config
        self.core_coins = list(self.config.get('tokens', {}).keys())
        if not self.core_coins:
            self.core_coins = ['bitcoin', 'ethereum', 'solana', 'chainlink', 'aave']
        
        # Token address and symbol mapping (for Fabric/DEX: use these symbols in report_trade)
        self.token_addresses = {
            coin: info.get('address')
            for coin, info in self.config.get('tokens', {}).items()
        }
        self.token_symbols = {
            coin: info.get('symbol', coin.upper())
            for coin, info in self.config.get('tokens', {}).items()
        }
        
        # Initialize components
        self._init_components()
        
        # State
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_allocations: Dict[str, float] = {coin: 0.0 for coin in self.core_coins}
        self.last_rebalance: Optional[datetime] = None
        self.run_id: Optional[str] = None
        
        # Fabric client
        self.fabric_client: Optional[Any] = None
        self.publisher: Optional[Any] = None
        
        logger.info(f"Initialized {self.params['name']} v{self.params['version']}")
        logger.info(f"Core coins: {self.core_coins}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    def _init_components(self):
        """Initialize strategy components."""
        # Regime detector with config
        regime_config = self.config.get('regime', {})
        self.regime_detector = RegimeDetector(
            RegimeConfig(
                weights=regime_config.get('weights', {}),
                bull_enter=regime_config.get('thresholds', {}).get('bull_enter', 0.5),
                bull_exit=regime_config.get('thresholds', {}).get('bull_exit', 0.1),
                bear_enter=regime_config.get('thresholds', {}).get('bear_enter', -0.7),
                bear_exit=regime_config.get('thresholds', {}).get('bear_exit', -0.3),
                ema_span=regime_config.get('ema_span', 21),
                min_duration_days=regime_config.get('min_duration_days', 14),
            )
        )
        
        # Portfolio allocator with config
        alloc_config = self.config.get('allocation', {})
        self.allocator = PortfolioAllocator(
            AllocationConfig(
                core_coins=self.core_coins,
                bull_top_n=alloc_config.get('bull', {}).get('top_n', 3),
                bull_exposure=alloc_config.get('bull', {}).get('exposure', 0.95),
                neutral_top_n=alloc_config.get('neutral', {}).get('top_n', 2),
                neutral_exposure=alloc_config.get('neutral', {}).get('exposure', 0.75),
                bear_crash_top_n=alloc_config.get('bear', {}).get('crash', {}).get('top_n', 1),
                bear_crash_exposure=alloc_config.get('bear', {}).get('crash', {}).get('exposure', 0.25),
                bear_base_top_n=alloc_config.get('bear', {}).get('base', {}).get('top_n', 2),
                bear_base_exposure=alloc_config.get('bear', {}).get('base', {}).get('exposure', 0.45),
                bear_transition_top_n=alloc_config.get('bear', {}).get('transition', {}).get('top_n', 2),
                bear_transition_exposure=alloc_config.get('bear', {}).get('transition', {}).get('exposure', 0.65),
                bear_crash_threshold=alloc_config.get('bear', {}).get('crash', {}).get('threshold', -0.05),
                funding_tilt=alloc_config.get('funding_tilt', 0.025),
            )
        )
        
        # Indicator calculators
        self.momentum_calc = MomentumCalculator()
        self.volatility_calc = VolatilityCalculator()
        self.ma_spread_calc = MASpreadCalculator()
        self.divergence_calc = DivergenceCalculator()
        
        # Funding fetcher (use mock in dev mode)
        dev_mode = os.getenv('FABRIC_DEV_MODE', 'false').lower() == 'true'
        if dev_mode:
            self.funding_fetcher = MockFundingFetcher()
            logger.info("Using mock funding fetcher (dev mode)")
        else:
            self.funding_fetcher = FundingFetcher()
        # Price fetcher for 12-week momentum (84+ days of daily prices)
        coin_to_cg = {
            c: info.get("coingecko_id", c)
            for c, info in self.config.get("tokens", {}).items()
        }
        if dev_mode:
            self.price_fetcher = MockPriceFetcher()
            logger.info("Using mock price fetcher (dev mode); 12w momentum will be empty until Fabric supplies data")
        else:
            self.price_fetcher = PriceFetcher(coin_to_coingecko_id=coin_to_cg)
    
    def before_loop(self):
        """
        Called once before the main loop starts.
        
        Initializes platform connection, loads historical data,
        and sets up initial state.
        """
        logger.info("=" * 60)
        logger.info("Strategy startup: before_loop")
        logger.info("=" * 60)
        
        # Initialize Fabric client
        # Note: FABRIC_API_KEY is auto-injected by fabric-cli dev via docker-compose.yaml
        if FABRIC_SDK_AVAILABLE:
            api_url = os.getenv('FABRIC_API_URL', 'https://api.fabricinvest.com')
            api_key = os.getenv('FABRIC_API_KEY')  # Auto-populated in dev mode
            
            if api_key:
                try:
                    self.fabric_client = FabricClient(
                        api_url=api_url,
                        api_key=api_key
                    )
                    
                    # Start a run
                    self.run_id = self.fabric_client.start_run(
                        strategy_name=self.params['name'],
                        symbol="PORTFOLIO",  # Multi-asset
                        timeframe=self.params['timeframe'],
                        initial_capital=Decimal(str(self.config.get('initial_capital', 10000)))
                    )
                    
                    logger.info(f"Started Fabric run: {self.run_id}")
                    
                    # Initialize publisher for notifications
                    strategy_id = os.getenv('FABRIC_STRATEGY_ID')
                    if strategy_id:
                        self.publisher = create_publisher(
                            strategy_id=strategy_id,
                            api_key=api_key
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to initialize Fabric client: {e}")
        
        # Initialize historical data buffer
        self._load_historical_data()
        
        logger.info("Strategy initialization complete")
    
    def _load_historical_data(self):
        """
        Load historical price data for all core coins.
        
        Needed for 12-week momentum (84+ days). Tries Fabric SDK if available,
        then falls back to price fetcher (e.g. CoinGecko).
        """
        historical_days = self.config.get('data', {}).get('historical_days', 200)
        momentum_period = self.config.get('data', {}).get('momentum_period', 84)
        logger.info(f"Loading {historical_days} days of historical data (12w momentum needs {momentum_period}+ days)...")
        
        # Initialize empty DataFrames for each data type
        self.historical_data = {
            'prices': pd.DataFrame(),
            'volumes': pd.DataFrame(),
            'momentum': pd.DataFrame(),
            'volatility': pd.DataFrame(),
            'ma_spreads': pd.DataFrame(),
            'divergence': pd.DataFrame(),
        }
        
        # Try cached file first (from scripts/fetch_historical_prices.py) for local testing
        cache_path = Path(__file__).parent.parent / "data" / "historical_prices.csv"
        if cache_path.exists():
            try:
                prices_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not prices_df.empty and len(prices_df) >= momentum_period:
                    # Align columns to core_coins (config may have subset)
                    prices_df = prices_df.reindex(columns=[c for c in self.core_coins if c in prices_df.columns]).dropna(how="all")
                    if not prices_df.empty:
                        self.historical_data["prices"] = prices_df
                        logger.info(f"Loaded {len(prices_df)} days of prices from cache {cache_path}")
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Try Fabric SDK if it provides historical candles (Type-2 multi-asset)
        if self.historical_data["prices"].empty and self.fabric_client and hasattr(self.fabric_client, 'get_candles'):
            try:
                prices_df = self._fetch_prices_from_fabric(historical_days)
                if not prices_df.empty and len(prices_df) >= momentum_period:
                    self.historical_data['prices'] = prices_df
                    logger.info(f"Loaded {len(prices_df)} days of prices from Fabric SDK")
            except Exception as e:
                logger.warning(f"Fabric price fetch failed, using fallback: {e}")
        
        # Fallback: price fetcher (e.g. CoinGecko) so 12-week momentum has data
        if self.historical_data["prices"].empty and hasattr(self, "price_fetcher"):
            try:
                prices_df = self.price_fetcher.fetch_historical_prices(days=historical_days)
                if not prices_df.empty:
                    self.historical_data['prices'] = prices_df
                    logger.info(f"Loaded {len(prices_df)} days of prices from price fetcher ({list(prices_df.columns)})")
            except Exception as e:
                logger.warning(f"Price fetcher failed: {e}")
        
        if self.historical_data['prices'].empty:
            logger.warning("No historical prices loaded; 12-week momentum and regime inputs will be empty until data is available")
        
        logger.info("Historical data initialized")
    
    def _fetch_prices_from_fabric(self, days: int) -> pd.DataFrame:
        """Build prices DataFrame from Fabric SDK get_candles (if API exists)."""
        out = {}
        for coin in self.core_coins:
            symbol = self.token_symbols.get(coin, coin.upper())
            candles = self.fabric_client.get_candles(symbol=symbol, timeframe="1d", limit=days)
            if candles is not None and len(candles) > 0:
                # Assume candles is array of [ts, o, h, l, c, v] or similar
                if hasattr(candles, '__iter__') and not isinstance(candles, (str, dict)):
                    arr = list(candles)
                    if arr and len(arr[0]) >= 5:
                        closes = [float(c[4]) for c in arr]
                        # Use last column as close; timestamps from first column if needed
                        ts = [float(c[0]) for c in arr] if len(arr[0]) > 0 else range(len(closes))
                        out[coin] = pd.Series(closes, index=pd.to_datetime(ts, unit='ms'))
                    elif arr and len(arr[0]) == 2:
                        out[coin] = pd.Series([float(c[1]) for c in arr], index=pd.to_datetime([c[0] for c in arr], unit='ms'))
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out)
    
    def should_rebalance(self) -> bool:
        """
        Check if rebalancing should occur.
        
        Rebalance on:
        1. Weekly schedule (Friday)
        2. Regime change
        
        Returns:
            True if rebalancing should occur
        """
        current_time = datetime.now()
        
        # Check for regime change
        if self.regime_detector.regime_changed:
            logger.info("Rebalance triggered: regime change")
            return True
        
        # Check weekly schedule
        rebal_config = self.config.get('rebalancing', {})
        if rebal_config.get('schedule') == 'weekly':
            day_of_week = rebal_config.get('day_of_week', 4)  # Friday
            
            if current_time.weekday() == day_of_week:
                # Check if we haven't already rebalanced today
                if self.last_rebalance is None or \
                   self.last_rebalance.date() < current_time.date():
                    logger.info("Rebalance triggered: weekly schedule")
                    return True
        
        return False
    
    @cached
    def get_coin_prices(self) -> Dict[str, pd.Series]:
        """Get current price series for all coins."""
        # In production, this would use Fabric SDK's self.candles or similar
        # For now, return from historical data
        prices = {}
        if 'prices' in self.historical_data:
            df = self.historical_data['prices']
            for coin in self.core_coins:
                if coin in df.columns:
                    prices[coin] = df[coin]
        return prices
    
    @cached
    def get_coin_momentum(self) -> Dict[str, float]:
        """
        Calculate 12-week momentum for all coins.
        
        Returns:
            Dictionary of coin -> 12-week momentum value
        """
        momentum = {}
        prices = self.get_coin_prices()
        
        for coin, price_series in prices.items():
            if len(price_series) >= 84:  # Need at least 12 weeks of data
                mom_series = self.momentum_calc.calculate_12w_momentum(price_series)
                if len(mom_series) > 0:
                    momentum[coin] = mom_series.iloc[-1]
        
        return momentum
    
    def calculate_regime_inputs(self) -> Dict[str, float]:
        """
        Calculate z-score normalized inputs for regime detection.
        
        Returns:
            Dictionary with momentum_z, volatility_z, funding_z, etc.
        """
        inputs = {
            'momentum_z': 0.0,
            'volatility_z': 0.0,
            'funding_z': 0.0,
            'ma_spread_z': 0.0,
            'divergence_z': 0.0,
        }
        
        prices = self.get_coin_prices()
        
        if not prices:
            return inputs
        
        # Calculate momentum z-scores
        momentum_values = []
        for coin, price_series in prices.items():
            if len(price_series) >= 84:
                mom = self.momentum_calc.calculate_12w_momentum(price_series)
                if len(mom) > 0:
                    mom_z = z_score_normalize(mom)
                    momentum_values.append(mom_z.iloc[-1])
        
        if momentum_values:
            inputs['momentum_z'] = np.mean(momentum_values)
        
        # Calculate volatility z-scores
        vol_values = []
        for coin, price_series in prices.items():
            if len(price_series) >= 30:
                vol = self.volatility_calc.calculate_realized_volatility(price_series, 30)
                if len(vol) > 0:
                    vol_z = z_score_normalize(vol)
                    vol_values.append(vol_z.iloc[-1])
        
        if vol_values:
            inputs['volatility_z'] = np.mean(vol_values)
        
        # Funding rates
        funding_rates = self.funding_fetcher.get_rates(self.core_coins)
        if funding_rates:
            avg_funding = np.mean(list(funding_rates.values()))
            inputs['funding_z'] = avg_funding * 10000  # Scale to reasonable range
        
        # MA spread z-scores
        ma_values = []
        for coin, price_series in prices.items():
            if len(price_series) >= 200:
                spread = self.ma_spread_calc.calculate_ma_spread(price_series, 50, 200)
                if len(spread) > 0:
                    spread_z = z_score_normalize(spread)
                    ma_values.append(spread_z.iloc[-1])
        
        if ma_values:
            inputs['ma_spread_z'] = np.mean(ma_values)
        
        return inputs
    
    def update_regime(self) -> Regime:
        """
        Update the regime detector with current market data.
        
        Returns:
            Current regime after update
        """
        inputs = self.calculate_regime_inputs()
        
        regime = self.regime_detector.update(
            momentum_z=inputs['momentum_z'],
            volatility_z=inputs['volatility_z'],
            funding_z=inputs['funding_z'],
            ma_spread_z=inputs['ma_spread_z'],
            divergence_z=inputs['divergence_z'],
        )
        
        state = self.regime_detector.get_state()
        logger.info(f"Regime: {state.regime.value}, Score: {state.score:.3f}")
        
        return regime
    
    def calculate_allocations(self) -> Allocation:
        """
        Calculate target portfolio allocations.
        
        Returns:
            Allocation object with weights and metadata
        """
        # Get current regime
        regime = self.regime_detector.current_regime
        
        # Get coin momentum
        coin_momentum = self.get_coin_momentum()
        btc_momentum = coin_momentum.get('bitcoin', 0.0)
        
        # Get funding rates
        funding_rates = self.funding_fetcher.get_rates(self.core_coins)
        
        # Calculate allocations
        allocation = self.allocator.get_allocations(
            regime=regime,
            coin_momentum=coin_momentum,
            btc_momentum=btc_momentum,
            funding_rates=funding_rates,
        )
        
        logger.info(f"Target allocation: {allocation.to_dict()}")
        
        return allocation
    
    def _trade_quantity_and_price(self, coin: str, usd_amount: float) -> tuple:
        """
        Convert trade USD amount to (quantity, price) for Fabric report_trade.
        Fabric/DEX typically expects quantity in base asset and price per unit.
        """
        price = 0.0
        quantity = usd_amount
        if usd_amount <= 0:
            return (0.0, 0.0)
        if "prices" in self.historical_data and not self.historical_data["prices"].empty:
            df = self.historical_data["prices"]
            if coin in df.columns:
                series = df[coin].dropna()
                if len(series) > 0:
                    price = float(series.iloc[-1])
                    if price > 0:
                        quantity = usd_amount / price
        return (quantity, price)

    def execute_rebalance(self, allocation: Allocation):
        """
        Execute rebalancing trades to achieve target allocation.
        
        Args:
            allocation: Target allocation
        """
        logger.info("=" * 40)
        logger.info("Executing rebalance")
        logger.info("=" * 40)
        
        # Calculate required trades
        portfolio_value = float(self.config.get('initial_capital', 10000))  # Would be actual portfolio value
        
        trades = self.allocator.calculate_rebalance_trades(
            current_weights=self.current_allocations,
            target_weights=allocation.weights,
            portfolio_value=portfolio_value,
        )
        
        if not trades:
            logger.info("No trades required")
            return
        
        # Log trades
        for coin, amount in trades.items():
            direction = "BUY" if amount > 0 else "SELL"
            logger.info(f"  {direction} {coin}: ${abs(amount):.2f}")
        
        # Execute trades via DEX
        # In production, this would use Fabric SDK's DEX integration
        # For now, just update current allocations
        self.current_allocations = allocation.weights.copy()
        self.last_rebalance = datetime.now()
        
        # Send trades to Fabric so the platform can execute them (DEX/paper).
        # report_trade: records the trade for the run.
        # send_trading_signal: if available, signals execution (Fabric may use this to trigger DEX).
        if self.fabric_client and self.run_id:
            for coin, amount in trades.items():
                try:
                    symbol = self.token_symbols.get(coin, coin.upper())
                    usd_amount = abs(amount)
                    quantity, price = self._trade_quantity_and_price(coin, usd_amount)
                    side = "buy" if amount > 0 else "sell"
                    self.fabric_client.report_trade(
                        run_id=self.run_id,
                        symbol=symbol,
                        side=side,
                        price=price,
                        quantity=quantity,
                        timestamp=datetime.now(),
                    )
                    # If Fabric uses send_trading_signal for execution, send it so the bot can execute
                    if hasattr(self.fabric_client, "send_trading_signal"):
                        self.fabric_client.send_trading_signal(
                            signal_type=side,
                            symbol=symbol,
                            price=price,
                            quantity=quantity,
                            confidence=1.0,
                            metadata={"run_id": self.run_id, "coin": coin},
                        )
                except Exception as e:
                    logger.error(f"Failed to report/send trade: {e}")
        
        # Send notification
        self._send_rebalance_notification(allocation, trades)
    
    def _send_rebalance_notification(
        self,
        allocation: Allocation,
        trades: Dict[str, float]
    ):
        """Send notification about rebalance."""
        if not self.publisher:
            return
        
        try:
            self.publisher.publish_strategy_notification(
                template_name="rebalance",
                variables={
                    "regime": allocation.regime.value,
                    "bear_phase": allocation.bear_phase.value if allocation.bear_phase else None,
                    "exposure": f"{allocation.exposure:.0%}",
                    "trades": [
                        {
                            "coin": coin,
                            "direction": "BUY" if amt > 0 else "SELL",
                            "amount": f"${abs(amt):.2f}"
                        }
                        for coin, amt in trades.items()
                    ],
                    "allocations": {
                        coin: f"{weight:.1%}"
                        for coin, weight in allocation.get_non_zero_weights().items()
                    }
                },
                priority="normal",
            )
        except Exception as e:
            logger.error(f"Failed to send rebalance notification: {e}")
    
    def _send_regime_change_notification(self, old_regime: Regime, new_regime: Regime):
        """Send notification about regime change."""
        if not self.publisher:
            return
        
        try:
            state = self.regime_detector.get_state()
            
            self.publisher.publish_strategy_notification(
                template_name="regime_change",
                variables={
                    "old_regime": old_regime.value,
                    "new_regime": new_regime.value,
                    "score": f"{state.score:.3f}",
                    "timestamp": datetime.now().isoformat(),
                },
                priority="high",
            )
        except Exception as e:
            logger.error(f"Failed to send regime change notification: {e}")
    
    def report_metrics(self):
        """Report current metrics to Fabric platform."""
        if not self.fabric_client or not self.run_id:
            return
        
        try:
            # Calculate portfolio value
            portfolio_value = float(self.config.get('initial_capital', 10000))
            cash = portfolio_value * (1 - sum(self.current_allocations.values()))
            positions_value = portfolio_value - cash
            
            # Report equity
            self.fabric_client.report_equity(
                run_id=self.run_id,
                equity=portfolio_value,
                cash=cash,
                positions_value=positions_value,
            )
            
            # Report custom metrics
            state = self.regime_detector.get_state()
            
            self.fabric_client.report_metric(
                name="regime_score",
                value=state.score,
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            logger.error(f"Failed to report metrics: {e}")
    
    def run(self):
        """
        Main strategy loop.
        
        For live execution, this would run continuously.
        For backtesting, this would be called once per candle.
        """
        logger.info("Starting strategy run loop")
        
        # Initialize
        self.before_loop()
        
        try:
            import time
            dev_mode = os.getenv('FABRIC_DEV_MODE', 'false').lower() == 'true'
            # In dev, run at most N iterations (default 1); 0 = no limit
            dev_iterations = int(os.getenv('FABRIC_DEV_ITERATIONS', '1'))
            iteration = 0

            # Main loop (for live execution)
            while True:
                iteration += 1
                # Update regime
                old_regime = self.regime_detector.current_regime
                new_regime = self.update_regime()
                
                # Check for regime change notification
                if self.regime_detector.regime_changed:
                    self._send_regime_change_notification(old_regime, new_regime)
                
                # Check if rebalancing needed
                if self.should_rebalance():
                    allocation = self.calculate_allocations()
                    self.execute_rebalance(allocation)
                
                # Report metrics
                self.report_metrics()
                
                # In dev mode: exit after N iterations (default 1) so local runs don't run forever
                if dev_mode and dev_iterations > 0 and iteration >= dev_iterations:
                    logger.info("Dev mode: exiting after %s iteration(s)", iteration)
                    break
                
                # Sleep until next check (e.g., daily); in dev use short sleep for multi-iteration testing
                sleep_seconds = 86400  # 24 hours
                if dev_mode:
                    sleep_seconds = int(os.getenv('FABRIC_DEV_SLEEP_SECONDS', '2'))
                time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
        except Exception as e:
            logger.exception(f"Strategy error: {e}")
            raise
        finally:
            self.after_loop()
    
    def after_loop(self):
        """
        Called once after the main loop ends.
        
        Cleanup and final reporting.
        """
        logger.info("Strategy shutdown: after_loop")
        
        # Report final metrics
        self.report_metrics()
        
        logger.info("Strategy shutdown complete")
