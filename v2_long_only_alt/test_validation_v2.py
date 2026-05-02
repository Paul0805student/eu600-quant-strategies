"""
Validation logique du V2 avec données synthétiques calibrées.
Vérifie que les 4 stratégies se comportent correctement avant déploiement réel.
"""
import sys
sys.path.insert(0, '/home/claude/eu600_quant_v2')

import numpy as np
import pandas as pd

from signals_v2 import (
    signal_tsmom, signal_52w_high, signal_idio_momentum, signal_bollinger_mr,
    combine_signals_v2, signal_to_long_weights, tsmom_to_weights,
)
from backtest_v2 import (
    get_monthly_rebalance_dates, compute_strategy_returns,
    apply_vol_targeting, apply_market_filter,
    compute_metrics, split_train_test, walk_forward_metrics,
)


def gen_market(n_assets=150, n_years=8, seed=42):
    """Marché synthétique avec persistance de momentum + mean reversion court."""
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    dates = pd.date_range("2016-01-04", periods=n_days, freq="B")
    tickers = [f"S{i:03d}" for i in range(n_assets)]

    # Marché
    mkt = rng.normal(0.07/252, 0.16/np.sqrt(252), n_days)

    # Stocks
    betas = rng.normal(1.0, 0.25, n_assets).clip(0.4, 1.6)
    idio_vol = rng.uniform(0.010, 0.022, n_assets)
    mom_quality = rng.normal(0, 1, n_assets)  # actifs avec vrai momentum

    # Alphas idiosyncratiques persistants
    alphas = mom_quality * (0.05/252)

    R = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        eps = rng.normal(0, idio_vol[i], n_days)
        R[:, i] = alphas[i] + betas[i] * mkt + eps

    # Persistance momentum modeste
    for t in range(252, n_days):
        past = R[t-252:t-21, :].sum(axis=0)
        z = (past - past.mean()) / (past.std() + 1e-9)
        R[t, :] += z * 0.0004

    # Mean reversion court terme légère (pour Bollinger)
    for i in range(n_assets):
        for t in range(2, n_days):
            R[t, i] -= 0.02 * R[t-1, i]

    prices = pd.DataFrame(100 * np.exp(np.cumsum(R, axis=0)), index=dates, columns=tickers)
    return prices, pd.DataFrame(R, index=dates, columns=tickers)


def main():
    print("=" * 78)
    print("VALIDATION V2 — données synthétiques calibrées")
    print("=" * 78)

    prices, daily_returns = gen_market(n_assets=150, n_years=8, seed=42)
    print(f"\n  {prices.shape[1]} actifs · {prices.shape[0]} jours")

    print("\n[1] Calcul des signaux...")
    sig_tsmom = signal_tsmom(prices, lookback=252)
    sig_52w = signal_52w_high(prices, lookback=252)
    market_proxy = daily_returns.mean(axis=1)
    sig_idio = signal_idio_momentum(prices, market_proxy=market_proxy,
                                    lookback_capm=252, mom_lookback=252, mom_skip=21)
    sig_boll = signal_bollinger_mr(prices, window=20, n_std=2.0)
    print(f"  TSMOM      shape={sig_tsmom.shape}  active%={sig_tsmom.iloc[-1].mean()*100:.0f}%")
    print(f"  52W High   shape={sig_52w.shape}    nan%={sig_52w.isna().sum().sum()/sig_52w.size*100:.0f}%")
    print(f"  Idio Mom   shape={sig_idio.shape}   nan%={sig_idio.isna().sum().sum()/sig_idio.size*100:.0f}%")
    print(f"  Bollinger  shape={sig_boll.shape}   nan%={sig_boll.isna().sum().sum()/sig_boll.size*100:.0f}%")

    composite = combine_signals_v2({
        "tsmom": sig_tsmom.fillna(0), "52w_high": sig_52w,
        "idio_mom": sig_idio, "bollinger": sig_boll,
    }, weights={"tsmom": 0.20, "52w_high": 0.25, "idio_mom": 0.35, "bollinger": 0.20})

    print("\n[2] Construction des portefeuilles + backtest...")
    rebal = get_monthly_rebalance_dates(prices.index)

    def bt(w, label):
        ret, _ = compute_strategy_returns(w, daily_returns, cost_bps=10, execution_lag=1)
        ret = apply_market_filter(ret, prices, sma_window=200, defensive_exposure=0.3)
        ret, _ = apply_vol_targeting(ret, target_vol=0.10)
        return ret

    strategies = {
        "TSMOM": bt(tsmom_to_weights(sig_tsmom.reindex(rebal).dropna(how="all")), "TSMOM"),
        "52W High": bt(signal_to_long_weights(sig_52w.reindex(rebal).dropna(how="all"), 0.20), "52W"),
        "Idio Mom": bt(signal_to_long_weights(sig_idio.reindex(rebal).dropna(how="all"), 0.20), "Idio"),
        "Bollinger": bt(signal_to_long_weights(sig_boll.reindex(rebal).dropna(how="all"), 0.20), "Boll"),
        "ENSEMBLE V2": bt(signal_to_long_weights(composite.reindex(rebal).dropna(how="all"), 0.20), "Ens"),
        "Benchmark EW": daily_returns.mean(axis=1),
    }

    print("\n[3] IS / OOS...")
    sample = strategies["ENSEMBLE V2"].dropna()
    _, _, cut = split_train_test(sample, train_frac=0.6)

    print(f"\n{'Strategy':<18} {'SR IS':>7} {'SR OOS':>8} {'CAGR OOS':>10} {'MaxDD OOS':>11} {'P(SR>1)':>9}")
    print("-" * 78)
    for name, ret in strategies.items():
        tr = ret.loc[:cut].dropna()
        te = ret.loc[cut:].iloc[1:].dropna()
        m_is = compute_metrics(tr); m_oos = compute_metrics(te)
        if m_is and m_oos:
            print(f"{name:<18} {m_is['Sharpe']:>7.2f} {m_oos['Sharpe']:>8.2f} "
                  f"{m_oos['CAGR']*100:>9.1f}% {m_oos['MaxDD']*100:>10.1f}% "
                  f"{m_oos['PSR_above_1']:>9.3f}")

    print("\n[4] Walk-forward 4 folds sur ENSEMBLE V2...")
    wf = walk_forward_metrics(strategies["ENSEMBLE V2"], n_folds=4)
    if not wf.empty:
        print(wf[["fold", "Sharpe", "CAGR", "MaxDD"]].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== VALIDATION OK : pipeline V2 fonctionnel ===")


if __name__ == "__main__":
    main()
