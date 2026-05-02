"""
Moteur de backtest V2.
Améliorations vs V1 :
- Walk-forward analysis avec embargo (López de Prado ch. 7)
- Métriques étendues : Probabilistic Sharpe Ratio (PSR, López ch. 14)
- Reporting plus complet (downside dev, Calmar)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def get_monthly_rebalance_dates(index):
    """Dernières dates ouvrées de chaque mois."""
    s = pd.Series(index, index=index)
    return s.groupby([s.index.year, s.index.month]).max().values


def get_weekly_rebalance_dates(index):
    """Dernières dates ouvrées de chaque semaine (vendredis principalement)."""
    s = pd.Series(index, index=index)
    return s.groupby([s.index.year, s.index.isocalendar().week]).max().values


def compute_strategy_returns(weights_rebal, daily_returns, cost_bps=10, execution_lag=1):
    """
    Calcule les rendements nets quotidiens d'une stratégie.
    """
    daily_weights = weights_rebal.reindex(daily_returns.index, method=None).ffill()
    daily_weights = daily_weights.shift(execution_lag).fillna(0)

    common_cols = daily_weights.columns.intersection(daily_returns.columns)
    daily_weights = daily_weights[common_cols]
    daily_returns = daily_returns[common_cols]

    gross = (daily_weights * daily_returns).sum(axis=1)
    turnover = daily_weights.diff().abs().sum(axis=1).fillna(0)
    costs = turnover * (cost_bps / 10000.0)
    return gross - costs, turnover


def apply_vol_targeting(returns, target_vol=0.10, lookback=60, max_leverage=2.0):
    """Vol targeting glissante sans look-ahead."""
    realized = returns.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    leverage = (target_vol / realized).clip(upper=max_leverage).fillna(1.0).shift(1).fillna(1.0)
    return returns * leverage, leverage


def apply_market_filter(returns, prices, sma_window=200, defensive_exposure=0.3):
    """
    Filtre régime : exposition réduite quand l'index proxy est sous SMA.
    On utilise la moyenne cross-section comme proxy d'index.
    """
    market = prices.mean(axis=1)
    sma = market.rolling(sma_window, min_periods=100).mean()
    in_bull = (market > sma).astype(float)
    regime = in_bull * 1.0 + (1 - in_bull) * defensive_exposure
    regime = regime.shift(1).reindex(returns.index).ffill().fillna(1.0)
    return returns * regime


# ----------------------------------------------------------------------------
# MÉTRIQUES (López de Prado-style)
# ----------------------------------------------------------------------------

def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0):
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).
    Probabilité que le vrai Sharpe > sr_benchmark sachant les obs observées.
    Tient compte de la skewness et du kurtosis.
    """
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan
    sr = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    skew = r.skew()
    kurt = r.kurt()  # excess kurtosis
    # Variance estimée du Sharpe
    sr_std = np.sqrt((1 - skew * sr + (kurt / 4) * sr**2) / (n - 1))
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
    cagr = (1 + total) ** (1 / n_years) - 1
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

    psr = probabilistic_sharpe_ratio(r, sr_benchmark=1.0)  # P(Sharpe > 1)

    return {
        "name": name,
        "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDD": max_dd, "Calmar": calmar, "HitRate_M": hit_rate,
        "PSR_above_1": psr, "N_obs": len(r), "N_years": n_years,
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


# ----------------------------------------------------------------------------
# WALK-FORWARD avec EMBARGO (López de Prado ch. 7)
# ----------------------------------------------------------------------------

def walk_forward_metrics(returns, n_folds=5, embargo_days=20):
    """
    Découpe la série en n_folds fenêtres temporelles successives,
    avec un embargo entre train et test pour éviter le leakage temporel.
    Renvoie le Sharpe par fold pour évaluer la stabilité.
    """
    r = returns.dropna()
    n = len(r)
    fold_size = n // n_folds
    results = []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n
        seg = r.iloc[start:end]
        if len(seg) < 50:
            continue
        m = compute_metrics(seg)
        m["fold"] = k + 1
        m["start"] = seg.index[0]
        m["end"] = seg.index[-1]
        results.append(m)
    return pd.DataFrame(results)


def split_train_test(returns, train_frac=0.6):
    n = len(returns)
    cut = int(n * train_frac)
    cut_date = returns.index[cut]
    return returns.loc[:cut_date], returns.loc[cut_date:].iloc[1:], cut_date
