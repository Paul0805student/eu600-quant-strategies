"""
Moteur de backtest rigoureux :
- Rebalancement mensuel (dernier jour ouvré)
- Lag d'exécution de 1 jour (pas de look-ahead)
- Coûts de transaction (10 bps par côté par défaut, réaliste pour l'Europe)
- Volatility targeting (10% annuel par défaut)
- Split train (in-sample) / test (out-of-sample) pour validation honnête
"""
import numpy as np
import pandas as pd


def get_monthly_rebalance_dates(index):
    """Renvoie les dernières dates ouvrées de chaque mois présentes dans l'index."""
    s = pd.Series(index, index=index)
    return s.groupby([s.index.year, s.index.month]).max().values


def compute_strategy_returns(
    weights_monthly,
    daily_returns,
    cost_bps=10,
    execution_lag=1,
):
    """
    Calcule les rendements quotidiens d'une stratégie.

    weights_monthly : DataFrame des poids cibles aux dates de rebalancement
    daily_returns : DataFrame des rendements quotidiens des actifs
    cost_bps : coûts en bps par côté du turnover
    execution_lag : délai d'exécution en jours (1 = on rebalance à t+1 le matin)
    """
    # On reindexe les poids au calendrier quotidien (forward fill = on tient les positions)
    daily_weights = weights_monthly.reindex(daily_returns.index, method=None)
    daily_weights = daily_weights.ffill()

    # Appliquer le lag d'exécution
    daily_weights = daily_weights.shift(execution_lag).fillna(0)

    # Aligner colonnes
    common_cols = daily_weights.columns.intersection(daily_returns.columns)
    daily_weights = daily_weights[common_cols]
    daily_returns = daily_returns[common_cols]

    # Rendement brut = somme(w_i * r_i)
    gross_returns = (daily_weights * daily_returns).sum(axis=1)

    # Calcul des coûts : turnover * cost_bps * 2 (achat + vente)
    turnover = daily_weights.diff().abs().sum(axis=1).fillna(0)
    costs = turnover * (cost_bps / 10000.0)

    net_returns = gross_returns - costs

    return net_returns, turnover


def apply_vol_targeting(returns, target_vol=0.10, lookback=60, max_leverage=2.0):
    """
    Applique un volatility targeting à la série de rendements.
    Calcule la vol réalisée glissante et ajuste l'exposition.
    """
    realized_vol = returns.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    # Levier = vol_cible / vol_réalisée, capé à max_leverage
    leverage = (target_vol / realized_vol).clip(upper=max_leverage).fillna(1.0)
    leverage = leverage.shift(1).fillna(1.0)  # Pas de look-ahead
    return returns * leverage, leverage


def apply_regime_filter(returns, regime_series):
    """Multiplie les rendements par le facteur de régime (lagué)."""
    regime_lagged = regime_series.shift(1).reindex(returns.index).ffill().fillna(1.0)
    return returns * regime_lagged


def compute_metrics(returns, name="Strategy", rf=0.0):
    """Calcule les métriques de performance standard."""
    r = returns.dropna()
    if len(r) < 30:
        return {}

    annual_factor = 252
    total_return = (1 + r).prod() - 1
    n_years = len(r) / annual_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(annual_factor)
    sharpe = (r.mean() * annual_factor - rf) / (r.std() * np.sqrt(annual_factor)) if r.std() > 0 else np.nan

    # Drawdown
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()

    # Sortino (vol des rendements négatifs uniquement)
    neg = r[r < 0]
    sortino = (r.mean() * annual_factor) / (neg.std() * np.sqrt(annual_factor)) if len(neg) > 5 else np.nan

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # Hit rate mensuel — 'ME' pour pandas >= 2.2, fallback 'M' sinon
    try:
        monthly = (1 + r).resample("ME").prod() - 1
    except ValueError:
        monthly = (1 + r).resample("M").prod() - 1
    hit_rate = (monthly > 0).mean()

    return {
        "name": name,
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "HitRate_M": hit_rate,
        "N_obs": len(r),
        "N_years": n_years,
    }


def print_metrics(metrics):
    """Affiche les métriques de manière lisible."""
    if not metrics:
        print("  (pas assez de données)")
        return
    print(f"  CAGR        : {metrics['CAGR']*100:>7.2f} %")
    print(f"  Volatilité  : {metrics['Vol']*100:>7.2f} %")
    print(f"  Sharpe      : {metrics['Sharpe']:>7.2f}")
    print(f"  Sortino     : {metrics['Sortino']:>7.2f}")
    print(f"  MaxDD       : {metrics['MaxDD']*100:>7.2f} %")
    print(f"  Calmar      : {metrics['Calmar']:>7.2f}")
    print(f"  Hit Rate M  : {metrics['HitRate_M']*100:>7.2f} %")
    print(f"  Période     : {metrics['N_years']:.1f} ans ({metrics['N_obs']} jours)")


def split_train_test(returns, train_frac=0.6):
    """Sépare la série en in-sample (train) et out-of-sample (test)."""
    n = len(returns)
    cut = int(n * train_frac)
    cut_date = returns.index[cut]
    train = returns.loc[:cut_date]
    test = returns.loc[cut_date:].iloc[1:]  # Pas d'overlap
    return train, test, cut_date
