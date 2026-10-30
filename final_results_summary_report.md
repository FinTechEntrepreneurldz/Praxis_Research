# Praxis Crypto Portfolio Final Results Summary

Generated UTC: 2026-10-22 12:37:13.902522+00:00
Output folder: /content/drive/MyDrive/QSentiaCryptoPMRevised/backtests/final_results_package_20261022_123302

## Experiment Setup

The evaluation used 14 crypto assets: AAVE/USDT, ADA/USDT, ATOM/USDT, BCH/USDT, BNB/USDT, BTC/USDT, DOGE/USDT, ETC/USDT, ETH/USDT, LTC/USDT, NEAR/USDT, SOL/USDT, UNI/USDT, XRP/USDT.
The base benchmark asset was BTC/USDT.
The number of walk-forward validation windows was 16.
The data frequency was treated as 96 bars per day and 35040 bars per year.
The transaction cost rate applied to turnover was 0.1500%.

## Strategy Ranking by Sharpe Ratio

1. Benchmark: BTC Buy & Hold: Sharpe 0.687, equity multiple 2.4159x, total return 141.59%, maximum drawdown 67.91%, average exposure 1.000.
2. Model: Agentic Alpha: Sharpe 0.562, equity multiple 1.8579x, total return 85.79%, maximum drawdown 66.40%, average exposure 1.000.
3. Benchmark: Top-3 7D Momentum: Sharpe 0.535, equity multiple 1.4777x, total return 47.77%, maximum drawdown 77.15%, average exposure 1.000.
4. Benchmark: Large Cap Equal Weight: Sharpe 0.518, equity multiple 1.6919x, total return 69.19%, maximum drawdown 68.41%, average exposure 1.000.
5. Benchmark: Equal Weight All Crypto: Sharpe 0.431, equity multiple 1.2067x, total return 20.67%, maximum drawdown 73.36%, average exposure 1.000.

## Comparison Against BTC Buy-and-Hold

Model: Agentic Alpha produced an equity ratio of 0.769x versus BTC, with a Sharpe delta of -0.125, a total return delta of -55.79%, and a maximum drawdown delta of -1.51%. It beat BTC by Sharpe in 8/16 windows.
Benchmark: Top-3 7D Momentum produced an equity ratio of 0.612x versus BTC, with a Sharpe delta of -0.152, a total return delta of -93.82%, and a maximum drawdown delta of 9.23%. It beat BTC by Sharpe in 6/16 windows.
Benchmark: Large Cap Equal Weight produced an equity ratio of 0.700x versus BTC, with a Sharpe delta of -0.169, a total return delta of -72.40%, and a maximum drawdown delta of 0.50%. It beat BTC by Sharpe in 7/16 windows.
Benchmark: Equal Weight All Crypto produced an equity ratio of 0.500x versus BTC, with a Sharpe delta of -0.255, a total return delta of -120.91%, and a maximum drawdown delta of 5.44%. It beat BTC by Sharpe in 6/16 windows.

## Agentic Alpha vs SAC RL Protective Overlay

The RL comparison was not available. This usually means cached SAC risk-manager models were not found. Run the full pipeline first so that models like SAC_riskrl_v4_w0.zip are created.

## Trade Logging Interpretation

The trade blotter records weight changes rather than exchange-executed fills. Each validation window starts with normalized portfolio value equal to 1.0. The estimated trade notional is therefore expressed in normalized portfolio units. To convert a trade row into dollars, multiply the normalized trade notional by the actual capital allocated to that window.

## Saved Files

overall_summary.csv
window_summary_by_strategy_and_window.csv
benchmark_comparison_vs_btc.csv
agentic_vs_rl_direct_comparison.csv
equity_curves_all_strategies.csv
global_equity_curves_chained_oos.csv
trade_blotter_all.csv
selection_decisions_all.csv
weights_all_snapshots.csv