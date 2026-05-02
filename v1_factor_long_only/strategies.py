"""
Stratégies quantitatives basées sur la littérature académique :

1. Cross-sectional Momentum (Jegadeesh & Titman 1993, Asness et al. 2013)
   - Rendement 12-1 mois : skip le dernier mois (effet reversal court terme)

2. Low Volatility (Frazzini & Pedersen 2014, Blitz & van Vliet 2007)
   - Volatilité 60 jours, signal négatif (low vol = bonne perf)

3. Short-Term Reversal (Jegadeesh 1990, Lehmann 1990)
   - Rendement 5 jours, signal négatif (les perdants rebondissent)

4. Quality / Trend (combinaison maison) :
   - Distance au plus haut sur 252j (proxy de stabilité)

Toutes les signaux sont calculés en cross-section (z-score à chaque date).
"""
import numpy as np
import pandas as pd


def zscore_cross_section(df, min_obs=20):
    """Z-score en cross-section à chaque date, robuste aux NaN."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    z = df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
    # Clipper les outliers à ±3 sigma
    z = z.clip(-3, 3)
    return z


def signal_momentum(prices, lookback=252, skip=21):
    """
    Momentum 12-1 : log-return entre t-252 et t-21 jours.
    On skip le dernier mois pour éviter l'effet reversal court terme.
    """
    log_p = np.log(prices)
    raw = log_p.shift(skip) - log_p.shift(lookback)
    return zscore_cross_section(raw)


def signal_low_vol(returns, lookback=63):
    """
    Low Volatility : -volatilité 63 jours.
    Plus la vol est faible, plus le signal est élevé.
    """
    vol = returns.rolling(lookback, min_periods=40).std()
    return zscore_cross_section(-vol)


def signal_reversal(prices, lookback=5):
    """
    Reversal court terme : -return sur 5 jours.
    Les actions qui ont chuté récemment ont tendance à rebondir.
    """
    raw = -(prices.pct_change(lookback))
    return zscore_cross_section(raw)


def signal_quality_trend(prices, lookback=252):
    """
    Proxy de qualité/tendance : distance au plus haut sur 12 mois.
    Une action proche de son plus haut est dans une tendance saine.
    """
    rolling_max = prices.rolling(lookback, min_periods=120).max()
    raw = prices / rolling_max - 1.0  # Entre -1 et 0, plus proche de 0 = mieux
    return zscore_cross_section(raw)


def market_regime_filter(prices, sma_window=200):
    """
    Filtre de régime : market cap-weighted return cumulé vs sa SMA.
    Renvoie 1.0 si bull (au-dessus SMA), 0.3 si bear (réduction d'exposition).
    """
    # Proxy "marché" = équipondéré de l'univers
    market = prices.mean(axis=1)
    sma = market.rolling(sma_window, min_periods=100).mean()
    in_bull = (market > sma).astype(float)
    # Bull = 100% exposition, Bear = 30% (defensive)
    regime = in_bull * 1.0 + (1 - in_bull) * 0.3
    return regime


def combine_signals(signals_dict, weights=None):
    """
    Combine plusieurs signaux z-scorés en un signal composite.
    weights : dict {nom: poids}. Si None, équipondéré.
    """
    if weights is None:
        weights = {k: 1.0 / len(signals_dict) for k in signals_dict}

    # Aligner toutes les dataframes
    common_index = None
    common_cols = None
    for sig in signals_dict.values():
        if common_index is None:
            common_index = sig.index
            common_cols = sig.columns
        else:
            common_index = common_index.intersection(sig.index)
            common_cols = common_cols.intersection(sig.columns)

    composite = pd.DataFrame(0.0, index=common_index, columns=common_cols)
    for name, sig in signals_dict.items():
        composite = composite + weights[name] * sig.reindex(
            index=common_index, columns=common_cols
        ).fillna(0)

    return composite


def signal_to_weights(signal, top_quantile=0.2, bottom_quantile=0.2, long_short=False):
    """
    Convertit un signal en poids de portefeuille.
    - long_short=False (défaut) : long-only top quintile, équipondéré
    - long_short=True : long top + short bottom, neutre dollar
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date in signal.index:
        row = signal.loc[date].dropna()
        if len(row) < 20:
            continue

        n_long = max(1, int(len(row) * top_quantile))
        long_names = row.nlargest(n_long).index
        weights.loc[date, long_names] = 1.0 / n_long

        if long_short:
            n_short = max(1, int(len(row) * bottom_quantile))
            short_names = row.nsmallest(n_short).index
            weights.loc[date, short_names] = -1.0 / n_short

    return weights
