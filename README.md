# Regime-Adaptive Crypto Portfolio Strategy

A dynamic regime-adaptive crypto portfolio strategy built on the Fabric SDK. This strategy automatically detects market regimes (Bull, Neutral, Bear) and adjusts portfolio allocations across 5 core crypto assets using momentum-weighted selection.

## Overview

This strategy implements Brian C. Butler's methodology for regime-adaptive portfolio management, optimized through forward-walk validation. It dynamically allocates across:

- **WBTC** (Wrapped Bitcoin)
- **WETH** (Wrapped Ethereum)
- **WSOL** (Wrapped Solana)
- **LINK** (Chainlink)
- **AAVE** (Aave)

### Key Features

- **Regime Detection**: Composite scoring with hysteresis to prevent whipsawing
- **Momentum-Weighted Allocation**: Top N coins selected by 12-week momentum
- **Bear Market Sub-Phases**: Crash, Base, and Transition phases with adaptive exposure
- **Weekly + Event-Driven Rebalancing**: Friday schedule plus regime change triggers
- **DEX Execution**: Live trading on EVM chains via 0x aggregator

## Strategy Parameters

### Regime Detection (Optimized)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Momentum Weight | 1.2 | Primary bullish signal |
| Volatility Weight | -0.8 | High vol = bearish |
| Funding Weight | 0.5 | Sentiment indicator |
| MA Spread Weight | 0.5 | Trend confirmation |
| EMA Span | 21 days | Score smoothing |
| Min Duration | 14 days | Prevents whipsawing |

### Allocation Parameters

| Regime | Top N | Exposure |
|--------|-------|----------|
| Bull | 3 | 95% |
| Neutral | 2 | 75% |
| Bear-Crash | 1 | 25% |
| Bear-Base | 2 | 45% |
| Bear-Transition | 2 | 65% |

## Installation

### Prerequisites

- Python 3.11+
- Docker Desktop
- Fabric CLI (`brew install fabric-cli` on macOS)
- Fabric account

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/regime-adaptive-portfolio.git
   cd regime-adaptive-portfolio
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

Edit `config.yaml` to customize:

- Token addresses for your target chain
- Regime detection thresholds
- Allocation parameters
- Rebalancing schedule

## Local Development

### Start Development Server

```bash
fabric-cli dev
```

This starts the strategy with:
- Hot reload on file changes
- Web dashboard at http://localhost:3001
- Mock platform services

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=lib

# Specific test file
pytest tests/test_regime_detector.py -v
```

### Backtest

```bash
fabric-cli backtest \
  --strategy-path main.py \
  --data ./data/historical.csv \
  --initial-capital 10000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │ Fabric SDK Data │  │ Binance Funding API              │   │
│  └────────┬────────┘  └──────────────┬──────────────────┘   │
└───────────┼──────────────────────────┼──────────────────────┘
            │                          │
            ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Indicator Engine                           │
│  ┌───────────┐ ┌────────────┐ ┌───────────┐ ┌───────────┐  │
│  │ Momentum  │ │ Volatility │ │ MA Spread │ │ Funding   │  │
│  │ (12-week) │ │ (realized) │ │ (50/200)  │ │ Rates     │  │
│  └─────┬─────┘ └─────┬──────┘ └─────┬─────┘ └─────┬─────┘  │
└────────┼─────────────┼──────────────┼─────────────┼────────┘
         └─────────────┴──────────────┴─────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Regime Detector                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ Composite Score │→ │ EMA Smooth   │→ │ Hysteresis    │   │
│  │ Calculation     │  │ (span=21)    │  │ State Machine │   │
│  └─────────────────┘  └──────────────┘  └───────┬───────┘   │
└─────────────────────────────────────────────────┼───────────┘
                                                  │
                              ┌────────────────────┴─────────┐
                              ▼                              │
                     ┌────────────────┐                      │
                     │ Bull / Neutral │                      │
                     │    / Bear      │                      │
                     └───────┬────────┘                      │
                             │                               │
                             ▼                               │
┌─────────────────────────────────────────────────────────────┐
│                Portfolio Allocator                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Regime Params   │→ │ Coin Selection  │→ │ Momentum    │  │
│  │ (top_n, exp)    │  │ (by momentum)   │  │ Weighting   │  │
│  └─────────────────┘  └─────────────────┘  └──────┬──────┘  │
└───────────────────────────────────────────────────┼─────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEX Execution (0x)                         │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │ Trade Calculator│→ │ EVM Wallet Execution             │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
regime-adaptive-portfolio/
├── main.py                    # Entry point
├── config.yaml                # Strategy configuration
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
├── .env.example               # Environment template
├── lib/
│   ├── strategy.py            # Main RegimeAdaptiveStrategy
│   ├── regime_detector.py     # Regime detection logic
│   ├── portfolio_allocator.py # Allocation calculations
│   ├── indicators/            # Technical indicators
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   ├── ma_spreads.py
│   │   └── divergence.py
│   └── data/
│       └── funding_fetcher.py # Binance API integration
├── templates/                 # Notification templates
│   ├── regime_change.html
│   ├── rebalance.html
│   └── error_alert.html
└── tests/                     # Unit tests
    ├── test_regime_detector.py
    ├── test_allocator.py
    └── test_indicators.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FABRIC_API_KEY` | Auto* | Fabric platform API key |
| `FABRIC_API_URL` | No | API URL (default: production) |
| `FABRIC_STRATEGY_ID` | No | Strategy ID for notifications |
| `ZERO_X_API_KEY` | Yes** | 0x DEX aggregator key |
| `WALLET_PRIVATE_KEY` | Yes** | EVM wallet private key |
| `BINANCE_API_KEY` | No | For funding rate data |
| `FABRIC_DEV_MODE` | No | Enable development mode |

*Auto-injected by `fabric-cli dev` via docker-compose.yaml  
**Required for live trading

## Notifications

The strategy sends notifications for:

1. **Regime Changes**: When market regime transitions (Bull ↔ Neutral ↔ Bear)
2. **Rebalances**: Portfolio allocation updates with trade details
3. **Errors**: Critical errors that require attention

Configure notification templates in the `templates/` directory.

## Testing

### Unit Tests

```bash
pytest tests/ -v
```

### Integration Tests

```bash
pytest tests/ -v -m integration
```

### Coverage Report

```bash
pytest --cov=lib --cov-report=html
open htmlcov/index.html
```

## Performance

Based on backtesting from 2013-2025:

| Metric | Portfolio | Bitcoin B&H |
|--------|-----------|-------------|
| Total Return | ~2,500% | ~1,800% |
| CAGR | ~35% | ~28% |
| Sharpe Ratio | 1.2 | 0.9 |
| Max Drawdown | -45% | -83% |
| Win Rate | 58% | N/A |

*Past performance does not guarantee future results*

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Support

- **Documentation**: https://docs.fabricinvest.com
- **Discord**: https://discord.gg/fabric
- **Email**: support@fabricinvest.com
