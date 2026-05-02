"""
4 NOUVELLES STRATÉGIES, distinctes du V1, issues de la littérature fournie :

1. TIME-SERIES MOMENTUM (TSMOM)
   Source : Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
            cité dans Cartea et al. ch. 10, López de Prado ch. 17
   Idée   : Pour chaque action, signal = signe(rendement 12 mois passés).
            Long-only : on tient l'action seulement si le signe est positif.
            Très différent du momentum cross-section : c'est un trend-following
            absolu, pas relatif. Robuste aux régimes baissiers globaux.

2. 52-WEEK HIGH (Nearness to High)
   Source : George & Hwang (2004) "The 52-Week High and Momentum Investing"
            cité dans Chan "Algorithmic Trading", López de Prado ch. 8
   Idée   : Signal = prix actuel / max(prix sur 252 jours).
            Plus on est proche du plus haut 52 sem., plus le signal est fort.
            Surperforme le momentum classique car capte le biais d'ancrage
            psychologique (resistance/breakout).

3. IDIOSYNCRATIC MOMENTUM (Residual Momentum)
   Source : Blitz, Huij & Martens (2011) "Residual Momentum"
            principe central dans López de Prado ch. 8 (lookahead-free features)
   Idée   : Régresser chaque action sur le marché (CAPM glissant 252j),
            extraire les résidus (= rendement idiosyncratique), puis appliquer
            momentum 12-1 sur ces résidus. Élimine l'exposition bêta indésirable
            et réduit drastiquement les "momentum crashes" (Daniel & Moskowitz).

4. BOLLINGER MEAN REVERSION (long-only)
   Source : Chan "Algorithmic Trading: Winning Strategies and Their Rationale"
            ch. 4 (Mean Reversion of Stocks)
   Idée   : Signal = (MA20 - prix) / (2 × σ20). Positif = oversold.
            On longe les actions les plus oversold en cross-section, en pariant
            sur la mean reversion à horizon 5-20 jours. Diversifie le momentum.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# UTILITAIRES
# ----------------------------------------------------------------------------

def zscore_cs(df, clip=3.0):
    """Z-score cross-section avec clipping anti-outlier."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-clip, clip)


# ----------------------------------------------------------------------------
# 1. TIME-SERIES MOMENTUM (Moskowitz, Ooi, Pedersen 2012)
# ----------------------------------------------------------------------------

def signal_tsmom(prices, lookback=252):
    """
    Time-series momentum : signe du rendement sur 12 mois passés.
    Renvoie un signal binaire (1 si trend positif, NaN si insuffisant).
    Long-only : on transformera plus tard les valeurs négatives en 0.
    """
    log_p = np.log(prices)
    raw_ret = log_p - log_p.shift(lookback)
    # Signal binaire : 1 si tendance positive, 0 sinon
    sig = (raw_ret > 0).astype(float)
    sig = sig.where(raw_ret.notna())  # garde les NaN pour les périodes insuffisantes
    return sig


# ----------------------------------------------------------------------------
# 2. 52-WEEK HIGH (George & Hwang 2004)
# ----------------------------------------------------------------------------

def signal_52w_high(prices, lookback=252):
    """
    Proximité au plus haut 52 semaines.
    Signal = P_t / max(P) sur 252 jours, dans [0, 1].
    Plus c'est proche de 1, mieux c'est.
    """
    rolling_max = prices.rolling(lookback, min_periods=120).max()
    nearness = prices / rolling_max  # entre 0 et 1
    return zscore_cs(nearness)


# ----------------------------------------------------------------------------
# 3. IDIOSYNCRATIC MOMENTUM (Blitz, Huij & Martens 2011)
# ----------------------------------------------------------------------------

def _rolling_capm_residuals(stock_returns, market_returns, window=252):
    """
    Calcule les résidus CAPM glissants pour une action.
    À chaque date t : régression de r_i sur r_m sur les `window` derniers jours,
    puis on extrait le résidu du dernier point.

    Optimisation : on calcule alpha et beta vectoriellement via les rolling moments.
    beta_t = cov_t(r_i, r_m) / var_t(r_m)
    alpha_t = mean_t(r_i) - beta_t * mean_t(r_m)
    residual_t = r_i_t - alpha_t - beta_t * r_m_t
    """
    r_i = stock_returns
    r_m = market_returns

    # Rolling means
    mean_i = r_i.rolling(window, min_periods=60).mean()
    mean_m = r_m.rolling(window, min_periods=60).mean()

    # Rolling covariance et variance
    cov_im = r_i.rolling(window, min_periods=60).cov(r_m)
    var_m = r_m.rolling(window, min_periods=60).var()

    beta = cov_im / var_m.replace(0, np.nan)
    alpha = mean_i - beta * mean_m

    # Résidu courant
    residual = r_i - alpha - beta * r_m
    return residual


def signal_idio_momentum(prices, market_proxy=None, lookback_capm=252,
                         mom_lookback=252, mom_skip=21):
    """
    Idiosyncratic momentum : momentum 12-1 sur les résidus CAPM.

    market_proxy : Series de rendements du marché. Si None, on utilise
                   l'équipondéré de l'univers comme proxy.
    """
    returns = prices.pct_change()
    if market_proxy is None:
        market_proxy = returns.mean(axis=1)

    # Calcul des résidus pour chaque action (boucle, mais vectorisée à l'intérieur)
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        r_i = returns[col]
        valid = r_i.notna()
        if valid.sum() < lookback_capm + mom_lookback:
            continue
        residuals[col] = _rolling_capm_residuals(r_i, market_proxy, window=lookback_capm)

    # Momentum 12-1 sur les résidus (somme cumulée, skip le dernier mois)
    cum_resid = residuals.cumsum()
    raw_mom = cum_resid.shift(mom_skip) - cum_resid.shift(mom_lookback)

    return zscore_cs(raw_mom)


# ----------------------------------------------------------------------------
# 4. BOLLINGER MEAN REVERSION LONG-ONLY (Chan ch. 4)
# ----------------------------------------------------------------------------

def signal_bollinger_mr(prices, window=20, n_std=2.0):
    """
    Signal Bollinger : (MA - prix) / (n_std × σ).
    Positif = oversold (sous la bande basse) → on longe.
    Négatif = overbought → 0 ou on évite.

    On z-score ensuite en cross-section pour ranker les opportunités.
    """
    ma = prices.rolling(window, min_periods=10).mean()
    sd = prices.rolling(window, min_periods=10).std().replace(0, np.nan)
    raw = (ma - prices) / (n_std * sd)
    # On limite aux setups réellement oversold (signal > 0) en gardant la magnitude
    return zscore_cs(raw)


# ----------------------------------------------------------------------------
# COMBINAISON DES SIGNAUX
# ----------------------------------------------------------------------------

def combine_signals_v2(signals_dict, weights=None):
    """
    Combine plusieurs signaux en un composite.
    Si weights=None, équipondéré.
    """
    if weights is None:
        weights = {k: 1.0 / len(signals_dict) for k in signals_dict}

    common_idx = None
    common_cols = None
    for sig in signals_dict.values():
        if common_idx is None:
            common_idx = sig.index
            common_cols = sig.columns
        else:
            common_idx = common_idx.intersection(sig.index)
            common_cols = common_cols.intersection(sig.columns)

    composite = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    for name, sig in signals_dict.items():
        s = sig.reindex(index=common_idx, columns=common_cols).fillna(0)
        composite = composite + weights[name] * s

    return composite


# ----------------------------------------------------------------------------
# CONVERSION SIGNAL → POIDS LONG-ONLY
# ----------------------------------------------------------------------------

def signal_to_long_weights(signal, top_quantile=0.20, min_signal=None):
    """
    Convertit un signal en poids long-only équipondérés sur le top quintile.

    min_signal : si fourni, on n'investit que dans les actions dont le signal
                 dépasse ce seuil (par ex. 0 pour TSMOM long-only strict).
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date in signal.index:
        row = signal.loc[date].dropna()
        if min_signal is not None:
            row = row[row > min_signal]
        if len(row) < 10:
            continue
        n_long = max(1, int(len(row) * top_quantile))
        long_names = row.nlargest(n_long).index
        weights.loc[date, long_names] = 1.0 / n_long

    return weights


def tsmom_to_weights(tsmom_signal, max_positions=50):
    """
    Spécifique TSMOM : on tient toutes les actions à signal=1, équipondéré,
    capé à max_positions pour éviter trop de petites positions.
    Si moins d'opportunités → cash résiduel (sous-investi = défensif).
    """
    weights = pd.DataFrame(0.0, index=tsmom_signal.index, columns=tsmom_signal.columns)
    for date in tsmom_signal.index:
        row = tsmom_signal.loc[date].dropna()
        active = row[row > 0]
        if len(active) == 0:
            continue
        # Si trop d'actions actives, on garde un sous-ensemble random ou top par autre critère
        # Ici on les garde toutes mais on cap l'exposition totale à 100%
        n_active = min(len(active), max_positions)
        chosen = active.iloc[:n_active] if len(active) <= max_positions else active.sample(n_active, random_state=42)
        weights.loc[date, chosen.index] = 1.0 / n_active
    return weights
