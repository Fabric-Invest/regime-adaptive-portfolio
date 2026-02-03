# Deployment: Portfolio and Swaps (Fabric SDK)

This doc explains how to ensure the regime-adaptive strategy creates the **correct portfolio** and can **conduct swaps** when deployed with the Fabric SDK.

## How the bot executes trades after you upload

When you deploy this strategy to Fabric:

1. **Your code runs in Fabric’s environment** (ECS/container). The strategy loop runs there: it updates regime, decides when to rebalance, and computes target allocations and trades.

2. **The strategy sends each trade to Fabric**, not directly to a DEX:
   - **`report_trade(...)`** – records the trade for the run (symbol, side, price, quantity, timestamp).
   - **`send_trading_signal(...)`** – if the Fabric SDK provides it, the strategy also sends a trading signal so the **platform** can treat it as an order to execute.

3. **Fabric’s platform executes the trades.** Execution is done by Fabric’s infrastructure, not inside your container. Fabric uses:
   - The deployment’s **execution mode** (e.g. paper vs live DEX),
   - The **chain** and **DEX** from your `config.yaml` (e.g. Ethereum + 0x),
   - **Secrets** you configure for the strategy (e.g. `ZERO_X_API_KEY`, `WALLET_PRIVATE_KEY`).

So: **your bot doesn’t call the DEX itself.** It sends trade intent (report_trade + optional send_trading_signal) to Fabric; Fabric’s systems perform the actual swap/order based on your deployment config and keys.

**What you need to do in Fabric:**
- In the Fabric portal, set the deployment to **live execution** (or paper, if you only want simulation).
- Configure the **secrets** required for execution (0x API key, wallet key, RPC, etc.) for this strategy/deployment.
- Ensure **chain** and **dex** in `config.yaml` match the network and aggregator Fabric uses for that deployment.

If you’re unsure whether execution is driven by `report_trade`, `send_trading_signal`, or another API, check Fabric’s docs or ask support; this strategy uses both when the SDK supports `send_trading_signal`.

---

## Correct portfolio

The portfolio is defined entirely by `config.yaml` → **`tokens`**.

- Each entry (e.g. `bitcoin`, `ethereum`, `solana`, `chainlink`, `aave`) is a constituent with `symbol`, `address`, and `decimals`.
- The strategy loads these at startup and uses them for allocation and for reporting trades to Fabric.
- **To change the portfolio:** edit `tokens` in `config.yaml` (add/remove/update symbols and addresses), then rebuild and redeploy.

Current constituents (see `config.yaml`):

| Config key | Symbol | Chain (config) |
|------------|--------|----------------|
| bitcoin    | WBTC   | Ethereum       |
| ethereum   | WETH   | Ethereum       |
| solana     | WSOL   | Ethereum (Wormhole) |
| chainlink  | LINK   | Ethereum       |
| aave       | AAVE   | Ethereum       |

The strategy starts a Fabric run with `symbol="PORTFOLIO"` (multi-asset). The **constituents** Fabric uses for the portfolio are the tokens you list in `config.yaml`; the strategy reports trades using each token’s **`symbol`** (WBTC, WETH, etc.) so the platform/DEX can match them.

## Conducting swaps

1. **Trade reporting**
   - Rebalances produce per-coin trade amounts (USD). The strategy reports each trade to Fabric via `report_trade(run_id, symbol=..., side=..., price=..., quantity=..., timestamp=...)`.
   - **Symbol:** Uses the token **symbol** from config (e.g. `WBTC`, `WETH`, `LINK`) so Fabric/DEX can route swaps to the right market.
   - **Quantity and price:** When latest price is available (e.g. from historical data), the strategy sends **quantity in base asset** and **price per unit** so the platform can execute swaps. If price is missing, it falls back to notional (USD) and `price=0`; confirm with Fabric docs whether execution supports that.

2. **Execution config**
   - `config.yaml` → **`execution`** must match the environment Fabric uses for this strategy:
     - `chain`: e.g. `ethereum`
     - `dex`: e.g. `0x`
     - `slippage_tolerance`, `gas_buffer`: set for live execution
   - Ensure the deployment in the Fabric portal is set to the same chain and execution mode (paper vs live DEX).

3. **DEX execution**
   - The strategy **reports** trades to Fabric; Fabric (or your deployment setup) is responsible for turning those into actual DEX swaps. If your Fabric setup uses a DEX integration, ensure:
     - Reported `symbol` values match the identifiers the DEX/connector expects (our config symbols are standard: WBTC, WETH, etc.).
     - Quantity and price format match what the Fabric execution layer expects (base units + price per unit when we have price).

## Checklist before/after deploy

- [ ] **Portfolio:** `config.yaml` → `tokens` lists exactly the assets you want in the portfolio (symbols and addresses for your chain).
- [ ] **Symbols:** Strategy uses `token_symbols` from config in `report_trade` (see `lib/strategy.py`: `token_symbols`, `execute_rebalance`).
- [ ] **Execution:** `config.yaml` → `execution` (chain, dex, slippage) matches the Fabric deployment environment.
- [ ] **After deploy:** In Fabric portal, confirm the run shows the expected portfolio constituents and that trades appear with correct symbols and sizes; check logs for any `report_trade` or execution errors.

## 12-week momentum: making sure the bot has the data

The 12-week momentum indicator needs **at least 84 days of daily close prices** per coin. The strategy loads this in `before_loop()` so momentum and regime inputs are available.

**How the bot gets the data:**

1. **Fabric SDK (if available)** – If the Fabric client exposes something like `get_candles(symbol, timeframe, limit)`, the strategy tries that first and builds a daily price series for each token.
2. **Price fetcher (fallback)** – If Fabric doesn’t provide historical prices, the strategy uses a price fetcher (e.g. CoinGecko) keyed by `config.yaml` → `tokens` → `coingecko_id` for each coin. It requests `data.historical_days` (default 200) days so 12-week momentum (84 days) and MA spreads (200 days) have enough data.

**What you need to do:**

- **Config:** In `config.yaml` → `data` keep `historical_days: 200` and `momentum_period: 84`.
- **Tokens:** Each entry under `tokens` should have a `coingecko_id` (e.g. `bitcoin`, `ethereum`, `solana`, `chainlink`, `aave`). The price fetcher uses this to request history from CoinGecko.
- **If using CoinGecko:** Free tier works with rate limits; set `COINGECKO_API_KEY` in the strategy’s environment (Fabric secrets or `.env`) for higher limits.
- **If Fabric supplies data:** Ensure the deployment has access to historical candles (e.g. 1d) for the portfolio symbols.

**Check that it works:** After startup, logs should show either "Loaded … days of prices from Fabric SDK" or "Loaded … days of prices from price fetcher". If you see "No historical prices loaded; 12-week momentum … will be empty", fix the data source and config above.

---

## Local vs production

- **Local (`fabric-cli dev`):** Uses the same `config.yaml`; trades are reported to the local Fabric platform. With `FABRIC_DEV_MODE=true`, the price fetcher is mocked so 12w momentum is empty unless you point at real data.
- **Production:** Use the same checklist; ensure either Fabric provides historical prices or `COINGECKO_API_KEY` is set so 12-week momentum has 84+ days of data.

For more on Fabric deployment and Type-2 verification, see **STRATEGIST_USAGE_GUIDE.md** → Deployment → “Type-2 portfolio and swap verification”.
