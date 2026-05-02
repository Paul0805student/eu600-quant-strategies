"""
Backtest V3 : walk-forward avec re-sélection périodique des paires.

Procédure rigoureuse :
- Période de FORMATION (ex: 2 ans) → identifier les paires cointégrées
- Période de TRADING (ex: 6 mois) → trader ces paires fixes
- Roller en avançant dans le temps (= walk-forward réel, pas juste IS/OOS)

Cela évite le data snooping classique en stat arb : si on identifie les paires
sur toute la série puis on backteste sur la même période, on a un biais énorme.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from cointegration import find_cointegrated_pairs
from stat_arb import backtest_portfolio_of_pairs


def apply_vol_targeting(returns, target_vol=0.10, lookback=60, max_leverage=3.0):
    realized = returns.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    leverage = (target_vol / realized).clip(upper=max_leverage).fillna(1.0).shift(1).fillna(1.0)
    return returns * leverage, leverage


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0):
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std() <= 0:
        return np.nan
    sr = r.mean() / r.std() * np.sqrt(252)
    skew = r.skew()
    kurt = r.kurt()
    sr_std = np.sqrt((1 - skew * sr + (kurt / 4) * sr ** 2) / (n - 1))
    if sr_std <= 0:
        return np.nan
    return float(norm.cdf((sr - sr_benchmark) / sr_std))


def compute_metrics(returns, name="Strategy"):
    r = returns.dropna()
    if len(r) < 30:
        return {}
    annual = 252
    total = (1 + r).prod() - 1
    n_years = len(r) / annual
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = r.std() * np.sqrt(annual)
    sharpe = (r.mean() * annual) / (r.std() * np.sqrt(annual)) if r.std() > 0 else np.nan

    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()

    neg = r[r < 0]
    sortino = (r.mean() * annual) / (neg.std() * np.sqrt(annual)) if len(neg) > 5 else np.nan
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    try:
        monthly = (1 + r).resample("ME").prod() - 1
    except ValueError:
        monthly = (1 + r).resample("M").prod() - 1
    hit_rate = (monthly > 0).mean()

    psr = probabilistic_sharpe_ratio(r, sr_benchmark=1.0)

    return {
        "name": name, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
        "Sortino": sortino, "MaxDD": max_dd, "Calmar": calmar,
        "HitRate_M": hit_rate, "PSR_above_1": psr,
        "N_obs": len(r), "N_years": n_years,
    }


def print_metrics(m):
    if not m:
        print("  (pas assez de données)")
        return
    print(f"  CAGR        : {m['CAGR']*100:>7.2f} %")
    print(f"  Volatilité  : {m['Vol']*100:>7.2f} %")
    print(f"  Sharpe      : {m['Sharpe']:>7.2f}")
    print(f"  Sortino     : {m['Sortino']:>7.2f}")
    print(f"  MaxDD       : {m['MaxDD']*100:>7.2f} %")
    print(f"  Calmar      : {m['Calmar']:>7.2f}")
    print(f"  Hit Rate M  : {m['HitRate_M']*100:>7.2f} %")
    print(f"  P(SR>1)     : {m['PSR_above_1']:>7.3f}")
    print(f"  Période     : {m['N_years']:.1f} ans")


def walkforward_pairs_trading(prices, formation_years=2, trading_months=6,
                              top_n_pairs=20, p_threshold=0.05,
                              corr_prefilter=0.7,
                              zscore_window=20, entry=2.0, exit=0.5, stop=4.0,
                              cost_bps=10, target_vol=0.10, verbose=True):
    """
    Walk-forward réel pour stat arb :
    À chaque période trading_months, on re-sélectionne les paires sur les
    `formation_years` années précédentes, puis on trade ces paires fixes.

    Renvoie : (portfolio_returns, list_of_pair_dataframes, weights_log)
    """
    formation_days = int(formation_years * 252)
    trading_days = int(trading_months * 21)

    all_returns = []
    pairs_log = []

    start_idx = formation_days
    period_id = 0
    while start_idx + trading_days < len(prices):
        period_id += 1
        # Fenêtre de formation
        formation = prices.iloc[start_idx - formation_days: start_idx]
        # Fenêtre de trading (qui suit immédiatement)
        end_idx = min(start_idx + trading_days, len(prices))
        trading = prices.iloc[start_idx: end_idx]

        if verbose:
            print(f"\n  [Période {period_id}] "
                  f"Formation : {formation.index[0].date()} -> {formation.index[-1].date()} "
                  f"| Trading : {trading.index[0].date()} -> {trading.index[-1].date()}")

        # Garder uniquement les actions avec données suffisantes sur la formation
        valid_cols = formation.dropna(axis=1, thresh=int(formation_days * 0.95)).columns
        formation_clean = formation[valid_cols]

        # Sélection des paires sur la formation
        pairs_df = find_cointegrated_pairs(
            formation_clean,
            p_threshold=p_threshold,
            corr_prefilter=corr_prefilter,
            top_n=top_n_pairs,
            verbose=verbose,
        )
        if pairs_df.empty:
            if verbose:
                print(f"    (aucune paire trouvée — période skipée)")
            start_idx += trading_days
            continue

        pairs_df["period"] = period_id
        pairs_log.append(pairs_df)

        # Backtest sur la période de trading avec les paires sélectionnées
        # On fournit l'historique complet (formation + trading) pour pouvoir
        # calculer les z-scores glissants dès le début de la trading window.
        full_window = prices.iloc[start_idx - zscore_window * 3: end_idx]
        port_ret, _ = backtest_portfolio_of_pairs(
            full_window, pairs_df,
            zscore_window=zscore_window, entry=entry, exit=exit, stop=stop,
            cost_bps=cost_bps,
        )

        # Ne garder que les rendements de la période de trading effective
        port_ret = port_ret.loc[trading.index[0]: trading.index[-1]]
        all_returns.append(port_ret)

        if verbose:
            sharpe_period = port_ret.mean() / port_ret.std() * np.sqrt(252) \
                if port_ret.std() > 0 else 0
            print(f"    Sharpe période : {sharpe_period:.2f} | "
                  f"jours actifs : {(port_ret != 0).sum()}/{len(port_ret)}")

        start_idx += trading_days

    if not all_returns:
        return pd.Series(dtype=float), [], None

    full_returns = pd.concat(all_returns).sort_index()
    # Vol targeting au niveau du portefeuille global
    full_returns_targeted, _ = apply_vol_targeting(full_returns, target_vol=target_vol)

    pairs_df_all = pd.concat(pairs_log, ignore_index=True) if pairs_log else pd.DataFrame()
    return full_returns_targeted, pairs_df_all, full_returns
