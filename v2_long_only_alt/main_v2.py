"""
Pipeline V2 — long-only, basé sur 4 stratégies issues directement des bouquins.

1. TSMOM (Moskowitz, Ooi, Pedersen 2012)
2. 52-Week High (George & Hwang 2004)
3. Idiosyncratic Momentum (Blitz, Huij & Martens 2011)
4. Bollinger Mean Reversion long-only (Chan ch. 4)

Composite vol-managed + filtre régime + walk-forward eval.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

from universe import get_universe
from data import download_prices, compute_returns, filter_universe
from signals_v2 import (
    signal_tsmom, signal_52w_high, signal_idio_momentum, signal_bollinger_mr,
    combine_signals_v2, signal_to_long_weights, tsmom_to_weights,
)
from backtest_v2 import (
    get_monthly_rebalance_dates, get_weekly_rebalance_dates,
    compute_strategy_returns, apply_vol_targeting, apply_market_filter,
    compute_metrics, print_metrics, split_train_test, walk_forward_metrics,
)


# ============================================================================
# CONFIG
# ============================================================================
START_DATE = "2014-01-01"
END_DATE = None
TRAIN_FRAC = 0.6
COST_BPS = 10
TARGET_VOL = 0.10
TOP_QUANTILE = 0.20
USE_REGIME_FILTER = True
MIN_HISTORY_DAYS = 750
N_WALK_FOLDS = 5
OUTPUT_DIR = Path(__file__).parent / "outputs_v2"
OUTPUT_DIR.mkdir(exist_ok=True)


def backtest_one(weights, daily_returns, prices, name):
    """Backtest standard : coûts -> filtre régime -> vol target."""
    ret, to = compute_strategy_returns(weights, daily_returns,
                                       cost_bps=COST_BPS, execution_lag=1)
    if USE_REGIME_FILTER:
        ret = apply_market_filter(ret, prices, sma_window=200, defensive_exposure=0.3)
    ret, lev = apply_vol_targeting(ret, target_vol=TARGET_VOL)
    return ret, to


def main():
    print("=" * 78)
    print("PIPELINE QUANT V2 — STOXX EUROPE 600")
    print("Long-only · TSMOM · 52W High · Idio Momentum · Bollinger MR")
    print("=" * 78)

    # ------------------------------------------------------------------------
    print("\n[1/5] Chargement des données...")
    tickers = get_universe()
    prices = download_prices(tickers, start=START_DATE, end=END_DATE)
    prices = filter_universe(prices, min_history_days=MIN_HISTORY_DAYS, min_price=1.0)
    print(f"  {prices.shape[1]} actions · {prices.shape[0]} jours")
    print(f"  {prices.index.min().date()} -> {prices.index.max().date()}")

    daily_returns = prices.pct_change().fillna(0)

    # ------------------------------------------------------------------------
    print("\n[2/5] Calcul des 4 signaux...")

    print("  • TSMOM (signe trend 12m)...")
    sig_tsmom = signal_tsmom(prices, lookback=252)

    print("  • 52-Week High...")
    sig_52w = signal_52w_high(prices, lookback=252)

    print("  • Idiosyncratic Momentum (CAPM résiduel glissant)... [le plus lent]")
    market_proxy = daily_returns.mean(axis=1)
    sig_idio = signal_idio_momentum(prices, market_proxy=market_proxy,
                                    lookback_capm=252, mom_lookback=252, mom_skip=21)

    print("  • Bollinger Mean Reversion (20d, 2σ)...")
    sig_boll = signal_bollinger_mr(prices, window=20, n_std=2.0)

    # Composite : poids basés sur la robustesse académique
    # Idiosyncratic momentum = factor le plus robuste → poids le plus élevé
    composite = combine_signals_v2({
        "tsmom": sig_tsmom.fillna(0),  # binaire 0/1, on z-score implicitement
        "52w_high": sig_52w,
        "idio_mom": sig_idio,
        "bollinger": sig_boll,
    }, weights={
        "tsmom": 0.20,
        "52w_high": 0.25,
        "idio_mom": 0.35,
        "bollinger": 0.20,
    })

    # ------------------------------------------------------------------------
    print("\n[3/5] Construction des portefeuilles...")
    rebal = get_monthly_rebalance_dates(prices.index)
    print(f"  Rebalancements mensuels : {len(rebal)}")

    strategies = {}

    # TSMOM : weights spécifiques (toutes les actions à signal=1)
    print("  TSMOM weights...")
    w_tsmom = tsmom_to_weights(sig_tsmom.reindex(rebal).dropna(how="all"))
    ret_tsmom, _ = backtest_one(w_tsmom, daily_returns, prices, "TSMOM")
    strategies["TSMOM (Moskowitz 2012)"] = ret_tsmom

    # 52W High : top quintile
    print("  52W High weights...")
    w_52w = signal_to_long_weights(sig_52w.reindex(rebal).dropna(how="all"),
                                   top_quantile=TOP_QUANTILE)
    ret_52w, _ = backtest_one(w_52w, daily_returns, prices, "52W High")
    strategies["52-Week High (George 2004)"] = ret_52w

    # Idio Momentum : top quintile
    print("  Idio Momentum weights...")
    w_idio = signal_to_long_weights(sig_idio.reindex(rebal).dropna(how="all"),
                                    top_quantile=TOP_QUANTILE)
    ret_idio, _ = backtest_one(w_idio, daily_returns, prices, "Idio Mom")
    strategies["Idio Momentum (Blitz 2011)"] = ret_idio

    # Bollinger : top quintile (les plus oversold)
    print("  Bollinger weights...")
    w_boll = signal_to_long_weights(sig_boll.reindex(rebal).dropna(how="all"),
                                    top_quantile=TOP_QUANTILE)
    ret_boll, _ = backtest_one(w_boll, daily_returns, prices, "Bollinger")
    strategies["Bollinger MR (Chan ch.4)"] = ret_boll

    # COMPOSITE
    print("  Composite weights...")
    w_comp = signal_to_long_weights(composite.reindex(rebal).dropna(how="all"),
                                    top_quantile=TOP_QUANTILE)
    ret_comp, _ = backtest_one(w_comp, daily_returns, prices, "ENSEMBLE V2")
    strategies["ENSEMBLE V2"] = ret_comp

    # Benchmark
    strategies["Benchmark (EW)"] = daily_returns.mean(axis=1)

    # ------------------------------------------------------------------------
    print("\n[4/5] Évaluation IS / OOS + Walk-Forward")
    print("-" * 78)
    sample = strategies["ENSEMBLE V2"].dropna()
    _, _, cut_date = split_train_test(sample, train_frac=TRAIN_FRAC)
    print(f"Coupure IS/OOS : {cut_date.date()}")

    summary_rows = []
    for name, ret in strategies.items():
        train = ret.loc[:cut_date].dropna()
        test = ret.loc[cut_date:].iloc[1:].dropna()
        m_is = compute_metrics(train) if len(train) > 30 else None
        m_oos = compute_metrics(test) if len(test) > 30 else None

        print(f"\n>>> {name}")
        if m_is:
            print(f"  --- In-Sample ---");  print_metrics(m_is)
        if m_oos:
            print(f"  --- Out-of-Sample ---");  print_metrics(m_oos)

        if m_is and m_oos:
            summary_rows.append({
                "Strategy": name,
                "Sharpe_IS": m_is["Sharpe"], "Sharpe_OOS": m_oos["Sharpe"],
                "CAGR_OOS": m_oos["CAGR"], "MaxDD_OOS": m_oos["MaxDD"],
                "PSR_OOS": m_oos["PSR_above_1"],
            })

    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 78)
    print("RÉSUMÉ COMPARATIF")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Walk-forward sur l'ensemble
    print("\n--- Walk-forward 5 folds sur ENSEMBLE V2 ---")
    wf = walk_forward_metrics(strategies["ENSEMBLE V2"], n_folds=N_WALK_FOLDS)
    if not wf.empty:
        print(wf[["fold", "start", "end", "Sharpe", "CAGR", "MaxDD"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\n  Sharpe moyen sur folds : {wf['Sharpe'].mean():.2f}")
        print(f"  Sharpe stabilité (1/CV): {wf['Sharpe'].mean()/wf['Sharpe'].std():.2f}")

    # ------------------------------------------------------------------------
    print("\n[5/5] Génération graphiques + export...")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    ax = axes[0]
    for name, ret in strategies.items():
        eq = (1 + ret.dropna()).cumprod()
        ax.plot(eq.index, eq.values, label=name, linewidth=1.5,
                alpha=0.95 if "ENSEMBLE" in name else 0.7,
                color="black" if "ENSEMBLE" in name else None)
    ax.axvline(cut_date, color="red", linestyle="--", alpha=0.6, label="IS/OOS")
    ax.set_yscale("log")
    ax.set_title("V2 — Stratégies long-only STOXX 600 (TSMOM · 52W · Idio Mom · Bollinger)",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ens = strategies["ENSEMBLE V2"].dropna()
    cum = (1 + ens).cumprod()
    dd = cum / cum.cummax() - 1
    ax.fill_between(dd.index, dd.values * 100, 0, color="red", alpha=0.4)
    ax.axvline(cut_date, color="red", linestyle="--", alpha=0.6)
    ax.set_title("Drawdown — ENSEMBLE V2", fontsize=12)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "equity_curves_v2.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  {fig_path}")

    pd.DataFrame(strategies).to_csv(OUTPUT_DIR / "strategy_returns_v2.csv")
    summary.to_csv(OUTPUT_DIR / "summary_v2.csv", index=False)
    if not wf.empty:
        wf.to_csv(OUTPUT_DIR / "walkforward_v2.csv", index=False)

    print("\n" + "=" * 78)
    print("TERMINÉ V2")
    print("=" * 78)

    return strategies, summary


if __name__ == "__main__":
    strategies, summary = main()
