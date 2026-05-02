"""
Sélection de paires cointégrées via Engle-Granger (Chan ch.5, Cartea ch.7).

Workflow rigoureux pour éviter le data snooping :
1. Pré-filtrer les paires par secteur/corrélation pour réduire le test multiple
2. Tester la cointégration uniquement sur la période FORMATION (in-sample)
3. Conserver les N meilleures paires (p-value les plus faibles + half-life raisonnable)
4. Trader ces paires sur la période TEST (out-of-sample) sans re-sélection
5. Réestimer les paires périodiquement (rolling formation)

Théorie (Engle-Granger 1987) :
- Deux séries I(1) sont cointégrées si une combinaison linéaire est I(0) (stationnaire)
- Test : régresser y_t = β·x_t + ε_t, puis ADF sur les résidus ε_t
- Si ADF rejette racine unitaire → cointégration → spread mean-reverting

Half-life du spread (Ornstein-Uhlenbeck) :
- Estimer θ via régression Δs_t = -θ·s_{t-1} + ε
- Half-life = ln(2) / θ
- Half-life trop courte (<2j) = bruit ; trop longue (>60j) = pas exploitable
"""
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm


def estimate_hedge_ratio(y, x):
    """
    Régression OLS y = α + β·x + ε.
    Renvoie (alpha, beta, residuals).
    """
    X = sm.add_constant(x)
    model = OLS(y, X, missing="drop").fit()
    return model.params.iloc[0], model.params.iloc[1], model.resid


def test_cointegration_eg(y, x, p_threshold=0.05):
    """
    Test d'Engle-Granger via statsmodels.coint (plus robuste que ADF manuel
    car corrige les valeurs critiques pour le fait qu'on teste sur des résidus).

    Renvoie : (is_coint, p_value, beta).
    """
    common = pd.concat([y, x], axis=1).dropna()
    if len(common) < 100:
        return False, 1.0, np.nan
    y_clean, x_clean = common.iloc[:, 0], common.iloc[:, 1]
    try:
        score, pvalue, _ = coint(y_clean, x_clean, trend="c", autolag="aic")
    except Exception:
        return False, 1.0, np.nan
    _, beta, _ = estimate_hedge_ratio(y_clean, x_clean)
    return pvalue < p_threshold, pvalue, beta


def half_life_ou(spread):
    """
    Half-life d'un processus Ornstein-Uhlenbeck via régression :
    Δs_t = α + θ·s_{t-1} + ε
    Half-life = -ln(2) / θ (θ négatif pour mean reversion).
    """
    s = spread.dropna()
    if len(s) < 30:
        return np.nan
    s_lag = s.shift(1).dropna()
    delta_s = s.diff().dropna()
    common_idx = s_lag.index.intersection(delta_s.index)
    s_lag = s_lag.loc[common_idx]
    delta_s = delta_s.loc[common_idx]
    if len(s_lag) < 30:
        return np.nan
    X = sm.add_constant(s_lag)
    try:
        model = OLS(delta_s, X).fit()
        theta = model.params.iloc[1]
        if theta >= 0:
            return np.nan  # pas de mean reversion
        return float(-np.log(2) / theta)
    except Exception:
        return np.nan


def find_cointegrated_pairs(prices, p_threshold=0.05,
                            corr_prefilter=0.5,
                            min_half_life=2.0, max_half_life=60.0,
                            top_n=30, verbose=True):
    """
    Cherche les paires cointégrées dans un panel de prix.

    Étapes :
    1. Pré-filtrer par corrélation des log-prix normalisés (plus pertinent que
       la corrélation des log-returns pour la cointégration)
    2. Pour chaque paire candidate : test EG + half-life
    3. Conserver top_n paires triées par p-value croissante

    Renvoie : DataFrame avec colonnes [stock1, stock2, pvalue, beta, half_life]
    """
    log_prices = np.log(prices)
    # Normaliser chaque série pour rendre les corrélations comparables
    # corr sur les log-prix capture mieux les comouvements de niveau
    norm_prices = (log_prices - log_prices.mean()) / log_prices.std()
    corr = norm_prices.corr()

    tickers = list(prices.columns)
    candidates = []
    for i, j in combinations(range(len(tickers)), 2):
        c = corr.iloc[i, j]
        if pd.notna(c) and c >= corr_prefilter:
            candidates.append((tickers[i], tickers[j], c))

    if verbose:
        print(f"  Candidats après pré-filtre corr>={corr_prefilter} : {len(candidates)}")

    results = []
    for s1, s2, c in candidates:
        y, x = log_prices[s1], log_prices[s2]
        is_coint, pval, beta = test_cointegration_eg(y, x, p_threshold=p_threshold)
        if not is_coint:
            continue
        # Spread = y - β·x
        common = pd.concat([y, x], axis=1).dropna()
        spread = common.iloc[:, 0] - beta * common.iloc[:, 1]
        hl = half_life_ou(spread)
        if pd.isna(hl) or hl < min_half_life or hl > max_half_life:
            continue
        results.append({
            "stock1": s1, "stock2": s2,
            "pvalue": pval, "beta": beta,
            "half_life": hl, "corr": c,
        })

    if not results:
        if verbose:
            print(f"  Paires cointégrées + half-life valide : 0")
        return pd.DataFrame(columns=["stock1", "stock2", "pvalue", "beta", "half_life", "corr"])

    df = pd.DataFrame(results).sort_values("pvalue").head(top_n).reset_index(drop=True)
    if verbose:
        print(f"  Paires cointégrées + half-life valide : {len(df)}")
    return df
