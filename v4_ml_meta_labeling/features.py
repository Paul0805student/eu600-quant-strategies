"""
Construction de features pour le ML financier.

Inspiré de López de Prado ch.18 ("Feature Importance") et de la pratique des
fonds quant. On veut des features :
- Stationnaires (sinon problème de cointégration des séries)
- Pas redondantes (sinon collinearity)
- Avec un horizon de prédiction cohérent avec le label

Catégories de features ici :
1. Momentum (returns multi-horizons)
2. Volatilité (réalisée à plusieurs échelles)
3. Skewness / Kurtosis (asymétries de distribution)
4. RSI (oscillateur)
5. Distance aux moyennes mobiles
6. Drawdown actuel
7. Volume-related (à activer si on a les volumes)
"""
import numpy as np
import pandas as pd


def compute_features_one_asset(prices_one_asset, vol_proxy=None):
    """
    Calcule un set de features pour UN actif. Renvoie un DataFrame indexé par date.
    """
    p = prices_one_asset.dropna()
    if len(p) < 252:
        return pd.DataFrame()

    log_p = np.log(p)
    ret = p.pct_change()

    df = pd.DataFrame(index=p.index)

    # === Momentum à plusieurs horizons ===
    for h in [5, 10, 21, 63, 126, 252]:
        df[f"ret_{h}d"] = log_p.diff(h)

    # === Volatilité réalisée ===
    for w in [10, 21, 63]:
        df[f"vol_{w}d"] = ret.rolling(w, min_periods=5).std() * np.sqrt(252)

    # === Skewness / Kurtosis sur 60j ===
    df["skew_60d"] = ret.rolling(60, min_periods=20).skew()
    df["kurt_60d"] = ret.rolling(60, min_periods=20).kurt()

    # === RSI 14 jours ===
    delta = p.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # === Distance aux SMA (en σ) ===
    for w in [20, 50, 200]:
        ma = p.rolling(w, min_periods=w // 2).mean()
        sd = p.rolling(w, min_periods=w // 2).std().replace(0, np.nan)
        df[f"dist_sma{w}"] = (p - ma) / sd

    # === Drawdown actuel depuis plus haut 252j ===
    rolling_max = p.rolling(252, min_periods=60).max()
    df["dd_from_max"] = p / rolling_max - 1

    # === Vol-of-vol (vol des changements de vol) ===
    df["vol_of_vol"] = df["vol_21d"].rolling(21, min_periods=10).std()

    # === Autocorrélation 1-lag des returns (sur 60j) ===
    df["autocorr_60"] = ret.rolling(60, min_periods=20).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) > 5 else np.nan,
        raw=False
    )

    return df.dropna()


def build_features_panel(prices, max_assets=None):
    """
    Construit le panel de features pour tous les actifs.
    Renvoie un DataFrame "long" : chaque ligne = (date, ticker, feature1, feature2, ...).
    """
    cols = prices.columns
    if max_assets:
        cols = cols[:max_assets]

    all_features = []
    for ticker in cols:
        feats = compute_features_one_asset(prices[ticker])
        if feats.empty:
            continue
        feats["ticker"] = ticker
        all_features.append(feats)

    if not all_features:
        return pd.DataFrame()

    panel = pd.concat(all_features)
    panel = panel.reset_index().rename(columns={"index": "date"})
    return panel
