# Fabric Invest - Strategist Usage Guide

**Complete guide for building, testing, and deploying trading strategies on the Fabric platform.**

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Creating Strategies](#creating-strategies)
4. [Local Development](#local-development)
5. [Testing & Backtesting](#testing--backtesting)
6. [Deployment](#deployment)
7. [Monitoring & Performance](#monitoring--performance)
8. [Notifications](#notifications)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Fabric Invest is a comprehensive platform for building, testing, and deploying algorithmic trading strategies. The platform consists of:

- **Fabric CLI** - Command-line tool for strategy development and deployment
- **Fabric SDK** - Python library for building strategies with built-in indicators and platform integration
- **Fabric Platform** - Backend API and infrastructure for running strategies in production
- **Web Dashboard** - Admin portal for managing strategies, deployments, and monitoring performance

### Key Capabilities

✅ **Strategy Development**
- Multiple strategy types (momentum, portfolio, ML/event-driven, etc.)
- Built-in technical indicators (RSI, MACD, EMA, etc.)
- Hyperparameter optimization
- Type-safe Python API

✅ **Local Development**
- Hot reload development server
- Local testing with mock platform services
- Web-based development dashboard
- Real-time debugging

✅ **Deployment**
- GitHub integration for version control
- Automated Docker builds
- One-click deployment to production
- ECS-based execution

✅ **Monitoring**
- Real-time performance metrics
- Equity curve tracking
- Trade history and analytics
- Custom notifications

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Docker Desktop** (for local development)
- **Git** (for version control)
- **GitHub account** (for deployment)
- **Fabric account** (for platform access)

### Installation

#### 1. Install Fabric CLI

**macOS (Homebrew):**
```bash
brew tap Fabric-Invest/homebrew-tap
brew install fabric-cli
```

**Linux/Windows:**
Download from [GitHub Releases](https://github.com/Fabric-Invest/fabric-cli/releases) and extract to your PATH.

**From Source:**
```bash
git clone https://github.com/Fabric-Invest/fabric-cli.git
cd fabric-cli
go build -o fabric-cli ./main.go
sudo mv fabric-cli /usr/local/bin/
```

#### 2. Authenticate with Platform

```bash
# Start authentication flow
fabric-cli auth
```

This opens your browser for OAuth2 + PKCE authentication. Tokens are securely stored and automatically refreshed.

**Verify authentication:**
```bash
fabric-cli auth --check
```

#### 3. Verify Installation

```bash
# Check CLI version
fabric-cli --version

# Run health check
fabric-cli doctor
```

---

## Creating Strategies

### Strategy Types

Fabric supports five strategy types:

| Type | Description | Use Case |
|------|-------------|----------|
| **Type-1** | Technical Analysis | Momentum trading, indicator-based strategies |
| **Type-2** | Portfolio Rebalancing | Asset allocation, rebalancing systems |
| **Type-3** | Backtest Analysis | Historical data analysis and research |
| **Type-4** | Optimization | Parameter optimization and walk-forward analysis |
| **Type-5** | Event-Driven ML | ML-based strategies reacting to events (Twitter, listings, etc.) |

### Creating a New Strategy

#### Option 1: Using CLI Wizard (Recommended)

```bash
# Interactive wizard
fabric-cli wizard
```

The wizard guides you through:
1. Strategy name
2. Strategy type selection
3. Asset types (crypto, stocks, both)
4. Execution mode (paper trading, DEX, both)

#### Option 2: Using CLI Init

```bash
# Create Type-1 strategy (momentum/technical analysis)
fabric-cli init my-momentum-strategy --type type-1

# Create Type-5 strategy (event-driven ML)
fabric-cli init my-ml-strategy --type type-5

# Create with custom options
fabric-cli init my-strategy \
  --type type-1 \
  --language python \
  --path ./strategies
```

#### Option 3: Manual Setup

1. **Fork the strategy template:**
   ```bash
   git clone https://github.com/Fabric-Invest/strategy-template.git my-strategy
   cd my-strategy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or install Fabric SDK separately
   pip install fabric-sdk
   ```

### Strategy Structure

A typical strategy directory:

```
my-strategy/
├── main.py              # Entry point
├── config.yaml          # Strategy configuration
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── .env.example         # Environment variables template
├── .env                 # Environment variables (not in git)
├── README.md           # Strategy documentation
├── lib/                # Strategy libraries
│   ├── strategy.py     # Main strategy logic
│   ├── indicators/     # Custom indicators
│   └── utils/          # Utility functions
├── tests/              # Test files
├── logs/               # Log files
└── data/               # Data files
```

### Writing Your First Strategy

#### Basic Strategy Template (Type-1)

```python
from fabric_sdk import Strategy, cached, indicators

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'stop_loss_pct': 0.10,
            'take_profit_pct': 0.10,
        }

    @cached
    def rsi(self):
        return indicators.rsi(
            self.candles,
            period=self.get_param('rsi_period')
        )

    def should_long(self) -> bool:
        rsi_value = self.rsi()
        return rsi_value < self.get_param('rsi_oversold')

    def on_position_opened(self, position):
        self.log(f"Position opened at ${position.entry_price}")

    def should_exit_long(self) -> bool:
        rsi_value = self.rsi()
        return rsi_value > 70

    def on_position_closed(self, trade):
        self.log(f"Trade closed: P&L ${trade.pnl:+.2f}")
```

#### Strategy Lifecycle Methods

```python
class MyStrategy(Strategy):
    def __init__(self):
        # Initialize strategy parameters
        super().__init__()
        self.params = {...}

    def before_loop(self):
        # Called once before backtesting starts
        # Setup initial state, fetch data, etc.
        pass

    def should_long(self) -> bool:
        # Return True to enter long position
        return False

    def should_short(self) -> bool:
        # Return True to enter short position
        return False

    def on_position_opened(self, position):
        # Called when position opens
        # Set stop loss, take profit, log entry
        pass

    def update_position(self):
        # Called every candle while position is open
        # Check stop loss, take profit, trailing stops
        pass

    def should_exit_long(self) -> bool:
        # Return True to exit long position
        return False

    def should_exit_short(self) -> bool:
        # Return True to exit short position
        return False

    def on_position_closed(self, trade):
        # Called when position closes
        # Log results, update statistics
        pass

    def after_loop(self):
        # Called once after backtesting ends
        # Generate reports, cleanup
        pass
```

#### Available Properties

Access market data and strategy state:

```python
# Current candle data
self.close          # Current close price
self.open           # Current open price
self.high           # Current high price
self.low            # Current low price
self.volume         # Current volume
self.current_time   # Current candle timestamp

# Historical data
self.candles        # Numpy array of OHLCV data

# Account state
self.balance        # Available cash
self.equity         # Total account value
self.position       # Current position (if any)

# Position checks
self.is_long        # True if in long position
self.is_short       # True if in short position
self.is_flat        # True if no position

# Parameters
self.get_param('rsi_period')  # Get parameter value
```

#### Built-in Indicators

**Momentum Indicators:**
```python
indicators.rsi(candles, period=14)
indicators.macd(candles, fast=12, slow=26, signal=9)
indicators.stoch(candles, k=14, d=3)
indicators.roc(candles, period=9)
```

**Trend Indicators:**
```python
indicators.sma(candles, period=20)
indicators.ema(candles, period=20)
indicators.wma(candles, period=20)
indicators.vwma(candles, period=20)
```

**Volume Indicators:**
```python
indicators.volume_sma(candles, period=20)
indicators.obv(candles)
```

### Configuration (config.yaml)

```yaml
# Strategy metadata
name: "My Momentum Strategy"
type: "type-1"
description: "RSI-based momentum trading"
version: "1.0.0"

# Trading parameters
symbol: "BTC-USD"
timeframe: "4h"
initial_capital: 10000

# Strategy-specific config
strategy:
  rsi_period: 14
  rsi_oversold: 30
  stop_loss_pct: 0.10
  take_profit_pct: 0.10
  position_size_pct: 0.45

# Platform integration
platform:
  api_url: "https://api.fabricinvest.com"
  enable_metrics: true
  enable_notifications: true
```

---

## Local Development

### Starting Development Server

```bash
# Start with hot reload
fabric-cli dev

# With specific environment
fabric-cli dev --env staging

# Offline mode (all services mocked)
fabric-cli dev --offline

# Custom port
fabric-cli dev --port 3001

# Disable dashboard
fabric-cli dev --no-dashboard
```

**Features:**
- 🔥 **Hot Reload**: Automatic restart on file changes
- 🌐 **Web Dashboard**: Visual monitoring at `http://localhost:3001`
- 🔐 **Authentication**: Real platform authentication
- 📊 **Live Logs**: Color-coded, structured logging
- 🔄 **Graceful Shutdown**: Clean shutdown on Ctrl+C

### Development Dashboard

Access the dashboard at `http://localhost:3001` (or your custom port):

**Features:**
- Real-time equity chart
- Live activity feed
- Notification preview
- Platform integration metrics
- One-click testing controls
- Log viewer
- State inspector

### Environment Variables

Create `.env` file in your strategy directory:

```bash
# Platform API
FABRIC_API_URL=http://localhost:8000
FABRIC_API_KEY=your_api_key_here

# Development mode
FABRIC_DEV_MODE=true

# Secrets (for local development)
CMC_API_KEY=your_key
TWITTER_BEARER_TOKEN=your_token
```

**Note:** Secrets are managed by the Fabric SDK's `CredentialsManager`:
- In production: Automatically loads from AWS Secrets Manager
- In local dev: Falls back to environment variables

### Hot Reload

The development server watches for file changes and automatically restarts your strategy:

```
[INFO] File changed: lib/strategy.py
[INFO] Restarting strategy...
[INFO] Strategy restarted successfully
```

### Debugging

#### Debug Mode

```bash
# Start in debug mode
fabric-cli debug

# With breakpoint
fabric-cli debug --break main.py:45

# Record execution trace
fabric-cli debug --trace
```

#### State Inspection

```bash
# Inspect all state
fabric-cli inspect

# Inspect specific component
fabric-cli inspect positions
fabric-cli inspect indicators
fabric-cli inspect equity

# Watch in real-time
fabric-cli inspect --watch
```

#### Logs

```bash
# Tail logs in real-time
fabric-cli logs --follow

# Filter by level
fabric-cli logs --level error

# Search logs
fabric-cli logs --search "BTC-USD"

# Export logs
fabric-cli logs --export logs.json
```

### Preview Notifications

```bash
# Preview all notifications
fabric-cli preview notifications

# Preview specific notification
fabric-cli preview notification trade_executed

# With mock data
fabric-cli preview --with-mock-data
```

---

## Testing & Backtesting

### Running Tests

```bash
# Run all tests
fabric-cli test

# Unit tests only
fabric-cli test --unit

# Integration tests
fabric-cli test --integration

# With coverage
fabric-cli test --coverage

# Watch mode
fabric-cli test --watch
```

### Backtesting

#### Using CLI Backtest Command

```bash
fabric-cli backtest \
  --strategy-path strategies/momentum_strategy.py \
  --data ./data/btc_4h.csv \
  --symbol BTC-USD \
  --initial-capital 10000 \
  --fee-rate 0.001 \
  --output reports/backtest.json
```

**CSV Data Format:**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000,42500,41900,42300,1000.5
2024-01-01 04:00:00,42300,42800,42200,42600,1200.3
...
```

#### Backtest Results

The backtest generates a JSON report:

```json
{
  "total_trades": 42,
  "winning_trades": 28,
  "losing_trades": 14,
  "win_rate": 0.667,
  "total_pnl": 1250.50,
  "total_return": 0.125,
  "max_drawdown": -0.08,
  "sharpe_ratio": 1.85,
  "equity_curve": [...],
  "trades": [...]
}
```

### Hyperparameter Optimization

Define optimization ranges in your strategy:

```python
class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.params = {
            'rsi_period': 14,
            'stop_loss_pct': 0.10,
        }
        
        # Optimization ranges: (min, max, step)
        self.hyperparameters = {
            'rsi_period': (10, 20, 2),        # Test 10, 12, 14, 16, 18, 20
            'stop_loss_pct': (0.05, 0.15, 0.02),  # Test 5%-15% in 2% steps
        }
```

Run optimization:

```bash
fabric optimize --strategy my_strategy.py --symbol BTC-USD
```

### Validation

Before deploying, validate your strategy:

```bash
# Run all validation checks
fabric-cli validate

# Quick check (critical issues only)
fabric-cli validate --quick

# Auto-fix issues
fabric-cli validate --fix

# Generate compliance report
fabric-cli validate --report
```

**Checks:**
- Configuration compliance
- Required files present
- Platform integration
- Error handling patterns
- Logging standards
- Security best practices

### Pre-Deployment Verification

```bash
# Run all verification checks
fabric-cli verify

# Check specific aspects
fabric-cli verify --notifications
fabric-cli verify --database
fabric-cli verify --performance

# Dry run deployment
fabric-cli verify --dry-run
```

---

## Deployment

### Overview

Deployment workflow:

1. **Push to GitHub** - Your strategy code
2. **Connect GitHub** - Link repository in admin portal
3. **Create Version** - Tag a version for deployment
4. **Build** - Automated Docker build via CodeBuild
5. **Deploy** - One-click deployment to ECS
6. **Monitor** - Track performance in real-time

### Step 1: Push to GitHub

```bash
cd my-strategy

# Initialize git (if not already)
git init
git add .
git commit -m "Initial strategy implementation"

# Add remote
git remote add origin https://github.com/your-org/my-strategy.git
git push -u origin main
```

### Step 2: Connect GitHub in Admin Portal

1. Navigate to **Admin Portal** → **Deployments**
2. Click **"Connect GitHub"**
3. Authorize Fabric to access your repositories
4. GitHub OAuth completes automatically

**Via CLI:**
```bash
# Check GitHub connection status
fabric-cli status github
```

### Step 3: Create Strategy in Portal

1. Navigate to **Strategies** → **New Strategy**
2. Fill in strategy details:
   - Name: `My Momentum Strategy`
   - Description: Strategy description
   - Type: Select strategy type
3. Click **Create**

**Via API:**
```bash
curl -X POST https://api.fabricinvest.com/api/v1/admin/strategies \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Momentum Strategy",
    "description": "RSI-based momentum trading",
    "slug": "my-momentum-strategy"
  }'
```

### Step 4: Link Repository

1. In the strategy details page, click **"Link Repository"**
2. Search and select your GitHub repository
3. Select branch (typically `main` or `master`)
4. System fetches commit SHA automatically
5. Click **Link**

**Repository Requirements:**
- Must contain `Dockerfile`
- Must contain `requirements.txt`
- Entry point should be `main.py` (configurable)

### Step 5: Create Version

1. In strategy details, go to **Versions** tab
2. Click **"New Version"**
3. Enter version label (e.g., `v1.0.0`)
4. Optionally add description
5. Click **Create**

### Step 6: Build

After linking repository, system automatically triggers build:

1. **Build Status**: Monitor in **Builds** tab
2. **Build Logs**: Click build to view CodeBuild logs
3. **Build Time**: Typically 3-5 minutes

**Manual Build Trigger:**
- Click **"Build"** button in version details

**Build Process:**
1. CodeBuild clones GitHub repository
2. Builds Docker image from `Dockerfile`
3. Runs tests (if configured)
4. Pushes image to ECR
5. Updates build status

### Step 7: Deploy

Once build completes successfully:

1. Go to **Deployments** tab
2. Click **"Deploy"** button
3. Select deployment configuration:
   - **CPU**: 256 (0.25 vCPU) or 512 (0.5 vCPU)
   - **Memory**: 512 MB or 1024 MB
   - **Environment**: TESTING or PRODUCTION
4. Click **"Deploy"**

**Deployment Status:**
- **PENDING**: Deployment queued
- **RUNNING**: Strategy is executing
- **STOPPED**: Manually stopped
- **FAILED**: Deployment error

### Step 8: Monitor Deployment

**View Real-Time Status:**
- Navigate to deployment details page
- View logs, metrics, and performance

**Stop Deployment:**
```bash
# Via portal
# Click "Stop" button in deployment details

# Via API
curl -X POST https://api.fabricinvest.com/api/v1/admin/deployments/{run_id}/stop \
  -H "Authorization: Bearer $API_KEY"
```

### Deployment via CLI

```bash
# Push strategy to platform
fabric-cli push

# Deploy to development
fabric-cli deploy my-strategy

# Deploy to production
fabric-cli deploy my-strategy --env production

# Check deployment status
fabric-cli status deployment-123
```

### Publishing Strategy

Once deployed and tested:

1. Navigate to strategy details
2. Click **"Publish"** button
3. Strategy becomes available to subscribers
4. Set publication status to **PUBLISHED**

**Publication States:**
- **DRAFT**: Admin-only, not visible to users
- **TESTING**: Internal testing
- **PUBLISHED**: Available for subscriptions
- **ARCHIVED**: Hidden from new subscriptions

---

## Monitoring & Performance

### Performance Metrics

Access performance data via:

**Web Portal:**
- Navigate to strategy → **Performance** tab
- View equity curve, metrics, and trades

**API:**
```bash
# Get strategy performance
GET /api/v1/strategies/{strategy_id}/performance

# Get deployment metrics
GET /api/v1/admin/deployments/{run_id}/performance/metrics

# Get equity curve
GET /api/v1/strategies/{strategy_id}/runs/{run_id}/equity

# Get trades
GET /api/v1/strategies/{strategy_id}/runs/{run_id}/trades
```

### Metrics Reported

Your strategy automatically reports:

**Equity Curve:**
- Total portfolio value
- Cash balance
- Positions value
- Timestamp

**Trades:**
- Entry/exit prices
- Position size
- P&L (profit/loss)
- Fees
- Timestamp

**Custom Metrics:**
```python
from fabric_sdk import FabricClient

fabric = FabricClient(api_url="https://api.fabricinvest.com", api_key=...)

# Report custom metric
fabric.report_metric(
    name="sharpe_ratio",
    value=1.85,
    timestamp=datetime.now()
)
```

### Real-Time Monitoring

**Dashboard Features:**
- Live equity curve chart
- Trade feed
- Performance metrics table
- Log streaming

**Alerts:**
- Performance degradation
- Unusual trading activity
- Error thresholds
- Health check failures

### Logs

**View Logs:**
```bash
# Via portal
# Navigate to deployment → Logs tab

# Via CLI
fabric-cli logs deployment-123 --follow

# Via API
GET /api/v1/admin/deployments/{run_id}/logs
```

**Log Levels:**
- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages

---

## Notifications

### Strategy Notifications

Send notifications to strategy subscribers:

#### Custom Template Notifications (Recommended)

**1. Upload Template to S3:**

```python
from fabric_sdk import create_publisher

publisher = create_publisher(
    strategy_id="your-strategy-uuid",
    api_key=os.getenv("FABRIC_API_KEY")
)

template_html = """
<html>
<body>
    <h1>Buy Signal: {{ symbol }}</h1>
    <p>Price: ${{ price }}</p>
    <p>Confidence: {{ confidence }}%</p>
    <p>Reason: {{ reason }}</p>
</body>
</html>
"""

publisher.upload_template(
    template_name="buy_signal",
    template_content=template_html,
    version="v1"
)
```

**2. Send Notification:**

```python
from fabric_sdk import create_publisher

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.publisher = create_publisher(
            strategy_id="your-strategy-uuid",
            api_key=os.getenv("FABRIC_API_KEY")
        )

    def on_position_opened(self, position):
        self.publisher.publish_strategy_notification(
            template_name="buy_signal",
            variables={
                "symbol": self.symbol,
                "price": float(position.entry_price),
                "quantity": float(position.quantity),
                "confidence": 85,
                "reason": "RSI oversold with volume spike"
            },
            priority="high",
            broadcast_filters={"tiers": ["premium", "vip"]}  # Optional
        )
```

#### Trading Signals

Notify all subscribers of buy/sell signals:

```python
from fabric_sdk import FabricClient

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.fabric = FabricClient(
            api_url="https://api.fabricinvest.com",
            api_key=os.getenv("FABRIC_API_KEY")
        )

    def before_loop(self):
        # Start run to get run_id for authentication
        self.run_id = self.fabric.start_run(
            strategy_name="MyStrategy",
            symbol=self.symbol,
            timeframe="4h",
            initial_capital=Decimal("10000")
        )

    def should_long(self) -> bool:
        # ... detect condition ...
        self.fabric.send_trading_signal(
            signal_type="buy",
            symbol=self.symbol,
            price=float(self.close),
            quantity=float(self.position.quantity) if self.position else None,
            confidence=0.9,
            metadata={"reason": "RSI oversold"}
        )
        return True
```

### Notification Channels

Notifications are delivered via:

- **Email**: HTML email with template rendering
- **SMS**: Text messages (SMS provider integration)
- **Discord**: Discord webhook integration

### Template Management

**Template Storage:**
- Templates stored in S3: `s3://fabric-strategies/{strategy_id}/templates/`
- Automatic caching (1-hour TTL)
- Version-controlled templates

**Template Variables:**
- Jinja2 syntax: `{{ variable_name }}`
- Full Jinja2 features (loops, conditionals, filters)

---

## Advanced Features

### Secrets Management

**In Production:**
- Secrets automatically loaded from AWS Secrets Manager
- SDK's `CredentialsManager` handles credential retrieval
- No API keys in code

**In Local Development:**
- Set environment variables:
  ```bash
  export CMC_API_KEY=your_key
  export TWITTER_BEARER_TOKEN=your_token
  ```
- Or use `.env` file (not committed to git)

**Available Secrets:**
- `CMC_API_KEY`: CoinMarketCap API key
- `ZERO_X_API_KEY`: 0x API key for DEX
- `INFURA_PROJECT_ID`: Infura project ID
- `SOLANA_RPC_URL`: Solana RPC endpoint
- `TWITTER_BEARER_TOKEN`: Twitter API token
- `JUPITER_API_KEY`: Jupiter API key
- `WALLET_PRIVATE_KEY`: EVM wallet private key
- `SOLANA_PRIVATE_KEY`: Solana wallet private key

### Caching Indicators

Use `@cached` decorator to avoid recalculating indicators:

```python
from fabric_sdk import cached

class MyStrategy(Strategy):
    @cached
    def rsi(self):
        # This is only calculated once per candle
        return indicators.rsi(self.candles, period=14)

    def should_long(self):
        # Call rsi() multiple times - only calculated once
        if self.rsi() < 30:
            if self.rsi() > 20:  # Still cached!
                return True
        return False
```

### Event-Driven Strategies (Type-5)

For event-driven ML strategies:

```python
from lib.data_sources.event_listener import EventListener
from lib.execution.position_manager import PositionManager

class EventDrivenStrategy:
    def __init__(self):
        self.event_listener = EventListener(config, event_queue, logger)
        self.position_manager = PositionManager(config, logger)

    def _handle_event(self, event: Event):
        # Process event
        market_data = self._fetch_market_data(event.symbol)
        prediction = self.ml_predictor.predict(event, market_data)
        
        if self._should_enter(prediction):
            self.position_manager.open_position(...)
```

**Event Sources:**
- Twitter monitoring
- Webhooks
- Blockchain events
- Custom sources

### Multi-Chain Support

**EVM Chains:**
```python
# Configure in config.yaml
execution:
  chain: "ethereum"  # or "polygon", "arbitrum", etc.
  rpc_url: "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
```

**Solana:**
```python
execution:
  chain: "solana"
  rpc_url: "https://api.mainnet-beta.solana.com"
```

### API Integration

**Platform API Client:**
```python
from fabric_sdk import FabricClient

fabric = FabricClient(
    api_url="https://api.fabricinvest.com",
    api_key=os.getenv("FABRIC_API_KEY")
)

# Register run
run_id = fabric.start_run(
    strategy_name="MyStrategy",
    symbol="BTC-USD",
    timeframe="4h",
    initial_capital=Decimal("10000")
)

# Report equity
fabric.report_equity(
    run_id=run_id,
    equity=10500.0,
    cash=5000.0,
    positions_value=5500.0
)

# Report trade
fabric.report_trade(
    run_id=run_id,
    symbol="BTC-USD",
    side="buy",
    price=45000.0,
    quantity=0.1,
    timestamp=datetime.now()
)
```

### GitHub Integration

**Automatic Deployment:**
1. Push to GitHub
2. Create version in portal
3. Link repository
4. Build and deploy automatically

**Branch Selection:**
- Select specific branch for deployment
- Use commit SHA for precise versioning

---

## Troubleshooting

### Common Issues

#### Strategy Won't Start

```bash
# Check environment
fabric-cli doctor

# Check logs
fabric-cli logs

# Run in debug mode
fabric-cli debug
```

#### Authentication Issues

```bash
# Re-authenticate
fabric-cli auth

# Check auth status
fabric-cli auth --check
```

#### Import Errors

```bash
# Ensure virtual environment is active
fabric-cli dev  # Auto-creates venv

# Reinstall dependencies
cd my-strategy
source venv/bin/activate
pip install -r requirements.txt
```

#### Build Failures

**Common Causes:**
- Missing `Dockerfile`
- Dockerfile syntax errors
- Missing dependencies in `requirements.txt`
- Build context issues

**Debug:**
1. Check build logs in portal
2. Test Docker build locally:
   ```bash
   docker build -t my-strategy .
   docker run my-strategy
   ```

#### Deployment Failures

**Common Causes:**
- Missing environment variables
- Insufficient resources (CPU/memory)
- Network connectivity issues
- Strategy crashes on startup

**Debug:**
1. Check deployment logs
2. Verify environment variables
3. Test strategy locally first
4. Check resource allocation

#### Performance Issues

**Optimization Tips:**
- Use `@cached` for indicators
- Minimize API calls
- Optimize data fetching
- Use efficient data structures

#### Notification Delivery Issues

**Check:**
- Template syntax (Jinja2)
- Template uploaded to S3
- Subscriber filters
- Notification channel configuration

---

## Best Practices

### Code Organization

1. **Separate Concerns:**
   - Strategy logic in `lib/strategy.py`
   - Indicators in `lib/indicators/`
   - Utilities in `lib/utils/`

2. **Error Handling:**
   ```python
   try:
       result = self.calculate_indicator()
   except Exception as e:
       self.log(f"Error calculating indicator: {e}", level="ERROR")
       return None
   ```

3. **Logging:**
   ```python
   self.log("Trade executed", level="INFO")
   self.log(f"Error: {error}", level="ERROR")
   ```

### Testing

1. **Unit Tests:**
   - Test individual functions
   - Mock external dependencies
   - Use test fixtures

2. **Integration Tests:**
   - Test full strategy flow
   - Use mock market data
   - Verify platform integration

3. **Backtesting:**
   - Test on historical data
   - Validate performance metrics
   - Check edge cases

### Security

1. **Never commit secrets:**
   - Use `.env` file (in `.gitignore`)
   - Use AWS Secrets Manager in production
   - Don't hardcode API keys

2. **Validate Input:**
   - Sanitize user inputs
   - Validate configuration
   - Check data types

3. **Error Handling:**
   - Catch and log errors
   - Don't expose sensitive information
   - Graceful degradation

### Performance

1. **Optimize Indicators:**
   - Use `@cached` decorator
   - Minimize calculations
   - Reuse computed values

2. **Efficient Data Structures:**
   - Use numpy arrays for OHLCV data
   - Avoid unnecessary copies
   - Use generators for large datasets

3. **API Calls:**
   - Batch requests when possible
   - Use async operations
   - Cache responses

---

## Resources

### Documentation

- **Platform API Docs**: https://api.fabricinvest.com/docs
- **SDK Documentation**: https://docs.fabricinvest.com/sdk
- **CLI Documentation**: See `fabric-cli --help`

### Examples

- **Momentum Strategy**: `fabric-sdk/examples/momentum_strategy.py`
- **Event-Driven Strategy**: `coinbase-listing-strategy/`
- **Notification Examples**: `fabric-sdk/examples/notifications_example.py`

### Support

- **GitHub Issues**: https://github.com/Fabric-Invest/fabric-cli/issues
- **Email**: support@fabricinvest.com
- **Discord**: https://discord.gg/fabric

---

## Quick Reference

### CLI Commands

```bash
# Setup
fabric-cli init my-strategy --type type-1
fabric-cli auth

# Development
fabric-cli dev
fabric-cli test
fabric-cli backtest --strategy-path strategy.py --data data.csv

# Deployment
fabric-cli push
fabric-cli deploy my-strategy
fabric-cli status deployment-123

# Debugging
fabric-cli debug
fabric-cli inspect
fabric-cli logs --follow
```

### Strategy Template

```python
from fabric_sdk import Strategy, cached, indicators

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.params = {'rsi_period': 14}

    @cached
    def rsi(self):
        return indicators.rsi(self.candles, period=14)

    def should_long(self) -> bool:
        return self.rsi() < 30

    def on_position_opened(self, position):
        self.log(f"Opened at ${position.entry_price}")
```

### Deployment Workflow

```
1. Push to GitHub
2. Connect GitHub in portal
3. Create strategy
4. Link repository
5. Create version
6. Build (automatic)
7. Deploy
8. Monitor
```

---

**Last Updated:** 2025-01-27  
**Version:** 1.0.0

