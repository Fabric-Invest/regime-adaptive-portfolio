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

## Local vs production

- **Local (`fabric-cli dev`):** Uses the same `config.yaml`; trades are reported to the local Fabric platform. Execution may be simulated or use a testnet; confirm in your Fabric dev setup.
- **Production:** Use the same checklist; ensure production `config.yaml` (or overrides) has the correct `tokens` and `execution` for the chain and DEX you use in production.

For more on Fabric deployment and Type-2 verification, see **STRATEGIST_USAGE_GUIDE.md** → Deployment → “Type-2 portfolio and swap verification”.
