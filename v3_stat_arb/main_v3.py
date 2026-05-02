"""
Pipeline V3 — Statistical Arbitrage par paires cointégrées sur STOXX 600.

Méthodologie :
1. Univers STOXX 600 (~250 actions liquides)
2. Walk-forward avec re-sélection des paires :
   - Formation : 2 ans rolling
   - Trading : 6 mois rolling (puis on roule)
3. Sélection : test d'Engle-Granger + half-life valide
4. Trading : z-score 20j, entry=2σ, exit=0.5σ, stop=4σ
5. Portefeuille : équipondéré sur top 20 paires par période
6. Risk management : vol targeting 10% au niveau global

Sources : Chan ch.5-6, Cartea ch.7, López de Prado ch.7 (walk-forward).

C'est une stratégie MARKET-NEUTRAL et DOLLAR-NEUTRAL par construction —
décorrélée des stratégies V1/V2 (qui sont long-only directionnelles).
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

from universe import get_universe
from data import download_prices, filter_universe
from cointegration import find_cointegrated_pairs
from stat_arb import backtest_portfolio_of_pairs
from backtest_v3 import (
    walkforward_pairs_trading, compute_metrics, print_metrics,
    apply_vol_targeting,
)


# ============================================================================
# CONFIG
# ============================================================================
START_DATE = "2014-01-01"
END_DATE = None
MIN_HISTORY_DAYS = 1000          # plus restrictif pour stat arb (besoin de longs historiques)

# Paramètres de cointégration
P_VALUE_THRESHOLD = 0.05         # seuil ADF strict
CORR_PREFILTER = 0.50            # paires candidates : corr log-prix normalisés > 0.50
TOP_N_PAIRS = 20                 # 20 paires actives par période

# Walk-forward
FORMATION_YEARS = 2.0            # 2 ans pour estimer cointégration
TRADING_MONTHS = 6               # 6 mois de trading puis re-sélection

# Trading rules
ZSCORE_WINDOW = 20               # fenêtre z-score (Chan recommande 20-60)
ENTRY_THRESHOLD = 2.0            # 2σ pour entrer
EXIT_THRESHOLD = 0.5             # 0.5σ pour sortir
STOP_THRESHOLD = 4.0             # 4σ stop-loss (cointégration cassée)

COST_BPS = 10                    # 10 bps par côté
TARGET_VOL = 0.10                # 10% annuel

OUTPUT_DIR = Path(__file__).parent / "outputs_v3"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 78)
    print("PIPELINE V3 — STATISTICAL ARBITRAGE / PAIRS TRADING")
    print("Market-neutral · Dollar-neutral · Walk-forward")
    print("=" * 78)

    # ------------------------------------------------------------------------
    print("\n[1/4] Chargement des données...")
    tickers = get_universe()
    prices = download_prices(tickers, start=START_DATE, end=END_DATE)
    prices = filter_universe(prices, min_history_days=MIN_HISTORY_DAYS, min_price=1.0)
    print(f"  {prices.shape[1]} actions | {prices.shape[0]} jours")
    print(f"  {prices.index.min().date()} -> {prices.index.max().date()}")

    daily_returns = prices.pct_change().fillna(0)
    benchmark = daily_returns.mean(axis=1)

    # ------------------------------------------------------------------------
    print("\n[2/4] Walk-forward pairs trading...")
    print(f"  Formation = {FORMATION_YEARS} ans | Trading = {TRADING_MONTHS} mois")
    print(f"  Top N paires par période : {TOP_N_PAIRS}")
    print(f"  Trading rules : entry=±{ENTRY_THRESHOLD}σ, exit=±{EXIT_THRESHOLD}σ, stop=±{STOP_THRESHOLD}σ")

    portfolio_ret, pairs_log, raw_returns = walkforward_pairs_trading(
        prices,
        formation_years=FORMATION_YEARS,
        trading_months=TRADING_MONTHS,
        top_n_pairs=TOP_N_PAIRS,
        p_threshold=P_VALUE_THRESHOLD,
        corr_prefilter=CORR_PREFILTER,
        zscore_window=ZSCORE_WINDOW,
        entry=ENTRY_THRESHOLD,
        exit=EXIT_THRESHOLD,
        stop=STOP_THRESHOLD,
        cost_bps=COST_BPS,
        target_vol=TARGET_VOL,
        verbose=True,
    )

    if portfolio_ret.empty:
        print("\n[ERREUR] Aucune paire cointégrée trouvée — ajuster les seuils.")
        return None, None

    # ------------------------------------------------------------------------
    print("\n[3/4] Évaluation globale (entièrement OOS par construction)...")
    print("-" * 78)
    print("Note : avec un walk-forward réel, TOUS les rendements sont OOS")
    print("       (les paires sont sélectionnées sur le passé uniquement).")
    print("-" * 78)

    metrics_strat = compute_metrics(portfolio_ret, name="Stat Arb V3")
    metrics_raw = compute_metrics(raw_returns, name="Stat Arb V3 (avant vol target)")
    metrics_bench = compute_metrics(benchmark.loc[portfolio_ret.index], name="Benchmark EW")

    print(f"\n>>> STAT ARB V3 (vol-targeted {TARGET_VOL*100:.0f}%)")
    print_metrics(metrics_strat)

    print(f"\n>>> STAT ARB V3 (avant vol targeting)")
    print_metrics(metrics_raw)

    print(f"\n>>> BENCHMARK EW (long-only, comparatif)")
    print_metrics(metrics_bench)

    # Corrélation avec le marché (doit être proche de 0 pour du market-neutral)
    common_idx = portfolio_ret.index.intersection(benchmark.index)
    corr_market = portfolio_ret.loc[common_idx].corr(benchmark.loc[common_idx])
    print(f"\n  Corrélation avec benchmark : {corr_market:.3f}")
    print(f"  (proche de 0 = market-neutral effectif)")

    # ------------------------------------------------------------------------
    print("\n[4/4] Analyse des paires + graphiques...")

    if not pairs_log.empty:
        # Stats sur les paires sélectionnées
        print(f"\n  Paires uniques sélectionnées au moins une fois : "
              f"{pairs_log[['stock1', 'stock2']].drop_duplicates().shape[0]}")
        print(f"  Half-life médiane : {pairs_log['half_life'].median():.1f} jours")
        print(f"  P-value médiane   : {pairs_log['pvalue'].median():.4f}")

        # Top 10 paires les plus fréquemment sélectionnées
        pair_keys = pairs_log["stock1"] + " / " + pairs_log["stock2"]
        top_pairs = pair_keys.value_counts().head(10)
        print("\n  Top 10 paires les plus fréquentes (sur toutes les périodes) :")
        for p, n in top_pairs.items():
            print(f"    {n:>2}× {p}")

        pairs_log.to_csv(OUTPUT_DIR / "pairs_log_v3.csv", index=False)

    # Graphiques
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    ax = axes[0]
    eq_strat = (1 + portfolio_ret.dropna()).cumprod()
    eq_raw = (1 + raw_returns.dropna()).cumprod()
    eq_bench = (1 + benchmark.loc[portfolio_ret.index].dropna()).cumprod()
    ax.plot(eq_strat.index, eq_strat.values,
            label=f"Stat Arb V3 vol-targeted (Sharpe={metrics_strat['Sharpe']:.2f})",
            linewidth=2, color="black")
    ax.plot(eq_raw.index, eq_raw.values,
            label=f"Stat Arb V3 raw (Sharpe={metrics_raw['Sharpe']:.2f})",
            linewidth=1.2, color="steelblue", alpha=0.8)
    ax.plot(eq_bench.index, eq_bench.values,
            label=f"Benchmark EW (Sharpe={metrics_bench['Sharpe']:.2f})",
            linewidth=1.2, color="orange", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("V3 — Statistical Arbitrage (Pairs Trading) sur STOXX 600",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cum = eq_strat
    dd = cum / cum.cummax() - 1
    ax.fill_between(dd.index, dd.values * 100, 0, color="red", alpha=0.4)
    ax.set_title(f"Drawdown — Stat Arb V3 (MaxDD = {metrics_strat['MaxDD']*100:.1f}%)",
                 fontsize=12)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "equity_curves_v3.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\n  Graphique : {fig_path}")

    # Export
    pd.DataFrame({
        "stat_arb_v3": portfolio_ret,
        "stat_arb_v3_raw": raw_returns,
        "benchmark_ew": benchmark.loc[portfolio_ret.index],
    }).to_csv(OUTPUT_DIR / "strategy_returns_v3.csv")

    summary = pd.DataFrame([
        {"Strategy": "Stat Arb V3 (vol-targeted)", **metrics_strat},
        {"Strategy": "Stat Arb V3 (raw)", **metrics_raw},
        {"Strategy": "Benchmark EW", **metrics_bench},
    ])
    summary.to_csv(OUTPUT_DIR / "summary_v3.csv", index=False)

    print("\n" + "=" * 78)
    print("TERMINÉ V3")
    print("=" * 78)

    return portfolio_ret, pairs_log


if __name__ == "__main__":
    portfolio_ret, pairs_log = main()
