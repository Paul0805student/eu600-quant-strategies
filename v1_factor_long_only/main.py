"""
Pipeline principal :
1. Télécharge les données STOXX 600 (~10 ans)
2. Calcule les 4 signaux factoriels
3. Construit chaque stratégie + ensemble
4. Backtest avec coûts, lag, vol targeting et filtre de régime
5. Évalue rigoureusement IS / OOS
6. Génère graphiques + rapport
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
from strategies import (
    signal_momentum, signal_low_vol, signal_reversal, signal_quality_trend,
    combine_signals, signal_to_weights, market_regime_filter,
)
from backtest import (
    get_monthly_rebalance_dates, compute_strategy_returns,
    apply_vol_targeting, apply_regime_filter,
    compute_metrics, print_metrics, split_train_test,
)


# ============================================================================
# CONFIGURATION
# ============================================================================
START_DATE = "2014-01-01"      # 10+ ans d'historique
END_DATE = None                # jusqu'à aujourd'hui
TRAIN_FRAC = 0.6               # 60% IS, 40% OOS
COST_BPS = 10                  # 10 bps par côté (réaliste Europe)
TOP_QUANTILE = 0.20            # top 20% en long
LONG_SHORT = False             # long-only (plus réaliste, shorting cher en Europe)
TARGET_VOL = 0.10              # 10% annuel
USE_REGIME_FILTER = True       # Filtre marché 200 SMA
MIN_HISTORY_DAYS = 750         # ~3 ans minimum
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_strategy(name, signal, daily_returns, rebalance_dates,
                 regime=None, top_q=TOP_QUANTILE, long_short=LONG_SHORT,
                 cost_bps=COST_BPS, target_vol=TARGET_VOL):
    """
    Construit et backteste une stratégie à partir d'un signal.
    Renvoie la série de rendements nets après tous les ajustements.
    """
    # Convertir signal -> poids aux dates de rebalancement
    signal_at_rebal = signal.reindex(rebalance_dates).dropna(how="all")
    weights = signal_to_weights(signal_at_rebal, top_quantile=top_q,
                                bottom_quantile=top_q, long_short=long_short)

    # Calcul des rendements avec coûts
    net_ret, turnover = compute_strategy_returns(
        weights, daily_returns, cost_bps=cost_bps, execution_lag=1
    )

    # Filtre de régime (lagué)
    if regime is not None:
        net_ret = apply_regime_filter(net_ret, regime)

    # Vol targeting
    if target_vol is not None:
        net_ret, lev = apply_vol_targeting(net_ret, target_vol=target_vol)

    return net_ret, weights, turnover


def main():
    print("=" * 78)
    print("PIPELINE QUANT — STOXX EUROPE 600")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # 1. DONNÉES
    # ------------------------------------------------------------------------
    print("\n[1/5] Chargement des données...")
    tickers = get_universe()
    print(f"  Univers initial : {len(tickers)} tickers")

    prices = download_prices(tickers, start=START_DATE, end=END_DATE)
    prices = filter_universe(prices, min_history_days=MIN_HISTORY_DAYS, min_price=1.0)
    print(f"  Univers après filtrage : {prices.shape[1]} actions, {prices.shape[0]} jours")
    print(f"  Période : {prices.index.min().date()} -> {prices.index.max().date()}")

    daily_returns = compute_returns(prices, method="simple")
    daily_returns = daily_returns.fillna(0)  # NaN = pas tradé ce jour, return = 0

    # ------------------------------------------------------------------------
    # 2. SIGNAUX
    # ------------------------------------------------------------------------
    print("\n[2/5] Calcul des signaux factoriels...")
    sig_mom = signal_momentum(prices, lookback=252, skip=21)
    sig_lv = signal_low_vol(daily_returns, lookback=63)
    sig_rev = signal_reversal(prices, lookback=5)
    sig_qt = signal_quality_trend(prices, lookback=252)

    # Filtre de régime (basé sur l'index équipondéré des actions)
    regime = market_regime_filter(prices, sma_window=200) if USE_REGIME_FILTER else None
    if regime is not None:
        n_bull = (regime > 0.5).sum()
        n_bear = (regime <= 0.5).sum()
        print(f"  Régime : {n_bull} jours bull / {n_bear} jours bear")

    # Composite : moyenne pondérée des 4 signaux
    # Pondération basée sur la robustesse historique des facteurs en Europe :
    # low-vol et momentum sont plus robustes que reversal/quality
    composite = combine_signals({
        "momentum": sig_mom,
        "low_vol": sig_lv,
        "reversal": sig_rev,
        "quality_trend": sig_qt,
    }, weights={
        "momentum": 0.30,
        "low_vol": 0.35,
        "reversal": 0.15,
        "quality_trend": 0.20,
    })

    # Dates de rebalancement mensuel
    rebalance_dates = get_monthly_rebalance_dates(prices.index)
    print(f"  Rebalancements mensuels : {len(rebalance_dates)}")

    # ------------------------------------------------------------------------
    # 3. BACKTEST DES STRATÉGIES
    # ------------------------------------------------------------------------
    print("\n[3/5] Backtest des stratégies individuelles + ensemble...")

    strategies = {}
    for name, sig in [
        ("Momentum 12-1", sig_mom),
        ("Low Volatility", sig_lv),
        ("Reversal 5d", sig_rev),
        ("Quality/Trend", sig_qt),
        ("ENSEMBLE (composite)", composite),
    ]:
        ret, w, to = run_strategy(name, sig, daily_returns, rebalance_dates,
                                  regime=regime)
        strategies[name] = ret
        print(f"  - {name:30s} : {len(ret.dropna())} obs, "
              f"turnover moyen = {to.mean()*252:.1f}/an")

    # Benchmark = équipondéré de l'univers (proxy STOXX 600)
    bench = daily_returns.mean(axis=1)
    strategies["Benchmark (EW)"] = bench

    # ------------------------------------------------------------------------
    # 4. ÉVALUATION IS / OOS
    # ------------------------------------------------------------------------
    print("\n[4/5] Évaluation In-Sample vs Out-of-Sample")
    print("-" * 78)

    # Date de coupure : cohérente entre stratégies
    sample_returns = strategies["ENSEMBLE (composite)"].dropna()
    _, _, cut_date = split_train_test(sample_returns, train_frac=TRAIN_FRAC)
    print(f"Coupure IS/OOS : {cut_date.date()}")
    print(f"  IS  : {sample_returns.index.min().date()} -> {cut_date.date()}")
    print(f"  OOS : {cut_date.date()} -> {sample_returns.index.max().date()}")

    summary = []
    for name, ret in strategies.items():
        train = ret.loc[:cut_date].dropna()
        test = ret.loc[cut_date:].iloc[1:].dropna()

        m_is = compute_metrics(train, name=f"{name} [IS]")
        m_oos = compute_metrics(test, name=f"{name} [OOS]")

        print(f"\n>>> {name}")
        print(f"  --- In-Sample ---")
        print_metrics(m_is)
        print(f"  --- Out-of-Sample ---")
        print_metrics(m_oos)

        if m_is and m_oos:
            summary.append({
                "Strategy": name,
                "Sharpe_IS": m_is["Sharpe"],
                "Sharpe_OOS": m_oos["Sharpe"],
                "CAGR_OOS": m_oos["CAGR"],
                "MaxDD_OOS": m_oos["MaxDD"],
                "Vol_OOS": m_oos["Vol"],
            })

    summary_df = pd.DataFrame(summary)
    print("\n" + "=" * 78)
    print("RÉSUMÉ COMPARATIF")
    print("=" * 78)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ------------------------------------------------------------------------
    # 5. GRAPHIQUES & EXPORT
    # ------------------------------------------------------------------------
    print("\n[5/5] Génération des graphiques...")

    # Equity curves
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # Plot 1 : equity curves complètes (log)
    ax = axes[0]
    for name, ret in strategies.items():
        equity = (1 + ret.dropna()).cumprod()
        ax.plot(equity.index, equity.values, label=name, linewidth=1.5,
                alpha=0.9 if "ENSEMBLE" in name else 0.7,
                color="black" if "ENSEMBLE" in name else None)
    ax.axvline(cut_date, color="red", linestyle="--", alpha=0.6, label="Coupure IS/OOS")
    ax.set_yscale("log")
    ax.set_title("Equity Curves (log scale) — STOXX 600 Quant Strategies", fontsize=13)
    ax.set_ylabel("Capital (base 1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2 : drawdown de l'ensemble
    ax = axes[1]
    ens = strategies["ENSEMBLE (composite)"].dropna()
    cum = (1 + ens).cumprod()
    dd = cum / cum.cummax() - 1
    ax.fill_between(dd.index, dd.values * 100, 0, color="red", alpha=0.4)
    ax.axvline(cut_date, color="red", linestyle="--", alpha=0.6)
    ax.set_title("Drawdown — Stratégie ENSEMBLE", fontsize=13)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "equity_curves.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  Graphique sauvegardé : {fig_path}")

    # Sauvegarde des rendements
    all_ret = pd.DataFrame(strategies)
    csv_path = OUTPUT_DIR / "strategy_returns.csv"
    all_ret.to_csv(csv_path)
    print(f"  Rendements sauvegardés : {csv_path}")

    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Résumé sauvegardé : {summary_path}")

    print("\n" + "=" * 78)
    print("TERMINÉ")
    print("=" * 78)

    return strategies, summary_df


if __name__ == "__main__":
    strategies, summary = main()
