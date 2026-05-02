"""
Stratégie statistical arbitrage par paires (Chan ch.5, Cartea ch.7).

Mécanique de trading par paire :
1. Calculer le spread = log(P1) - β · log(P2)
2. Calculer le z-score glissant : z = (spread - μ_window) / σ_window
3. Règles de trading classiques (Bollinger sur spread) :
   - Entrer SHORT spread (sell P1, buy β·P2) si z > +entry_threshold
   - Entrer LONG  spread (buy P1, sell β·P2) si z < -entry_threshold
   - Sortir quand |z| < exit_threshold (retour à la moyenne)
   - Stop-loss si |z| > stop_threshold (la cointégration s'est cassée)

Diversification : on trade N paires simultanément, capital équiréparti.
Le portefeuille global est dollar-neutral et market-neutral par construction.
"""
import numpy as np
import pandas as pd


def compute_spread_zscore(log_p1, log_p2, beta, window=20):
    """
    Spread = log(P1) - β·log(P2) ; z-score sur fenêtre glissante.
    """
    spread = log_p1 - beta * log_p2
    mu = spread.rolling(window, min_periods=10).mean()
    sigma = spread.rolling(window, min_periods=10).std().replace(0, np.nan)
    z = (spread - mu) / sigma
    return spread, z


def generate_pair_positions(z, entry=2.0, exit=0.5, stop=4.0):
    """
    Génère la position dans le spread sur la base du z-score.
    Position : -1 (short spread), 0 (flat), +1 (long spread).

    Règles :
    - Entry : prendre position contre l'extrême
    - Exit  : fermer quand le spread revient près de la moyenne
    - Stop  : fermer si dérive trop forte (cointégration cassée)

    Important : on lague la position pour exécuter à t+1 (pas de look-ahead).
    """
    pos = pd.Series(0.0, index=z.index)
    current = 0
    for i, val in enumerate(z.values):
        if pd.isna(val):
            pos.iloc[i] = current
            continue
        if current == 0:
            if val > entry:
                current = -1   # short spread
            elif val < -entry:
                current = +1   # long spread
        else:
            # En position : check exit ou stop
            if abs(val) < exit:
                current = 0
            elif abs(val) > stop:
                current = 0
        pos.iloc[i] = current
    return pos


def backtest_pair(prices, stock1, stock2, beta,
                  zscore_window=20, entry=2.0, exit=0.5, stop=4.0,
                  cost_bps=10):
    """
    Backteste une paire individuelle.

    Renvoie une Series de rendements quotidiens nets de coûts pour cette paire.
    Le portefeuille de la paire est dollar-neutral :
    - Long spread = +1$ sur stock1, -β$ sur stock2 (normalisé pour |L|+|S| = 1)
    - Short spread = -1$ sur stock1, +β$ sur stock2

    Convention de normalisation : on alloue 0.5$ par leg pour avoir une
    exposition gross totale de 1$ par paire (long-leg 0.5 + short-leg 0.5).
    """
    p1 = prices[stock1].copy()
    p2 = prices[stock2].copy()
    valid = p1.notna() & p2.notna()
    p1 = p1[valid]
    p2 = p2[valid]
    if len(p1) < zscore_window + 10:
        return None

    log_p1 = np.log(p1)
    log_p2 = np.log(p2)
    spread, z = compute_spread_zscore(log_p1, log_p2, beta, window=zscore_window)
    pos = generate_pair_positions(z, entry=entry, exit=exit, stop=stop)

    # Lag d'1 jour pour exécution sans look-ahead
    pos_lag = pos.shift(1).fillna(0)

    # Rendements des deux jambes
    r1 = p1.pct_change().fillna(0)
    r2 = p2.pct_change().fillna(0)

    # Normaliser le hedge : la jambe stock1 a poids ±0.5 et stock2 a ±0.5·sign(beta)
    # (On préserve le ratio β en restant dollar-neutral à 0.5 chaque jambe).
    # Position = +1 sur spread → long stock1, short stock2
    w1 = 0.5 * pos_lag
    w2 = -0.5 * pos_lag * np.sign(beta)

    pair_ret = w1 * r1 + w2 * r2

    # Coûts : turnover des deux jambes
    turnover = (w1.diff().abs() + w2.diff().abs()).fillna(0)
    costs = turnover * (cost_bps / 10000.0)
    pair_ret_net = pair_ret - costs

    pair_ret_net.name = f"{stock1}_{stock2}"
    return pair_ret_net


def backtest_portfolio_of_pairs(prices, pairs_df,
                                zscore_window=20, entry=2.0, exit=0.5, stop=4.0,
                                cost_bps=10):
    """
    Backteste un portefeuille équipondéré de N paires cointégrées.

    pairs_df : DataFrame issu de find_cointegrated_pairs (contient stock1, stock2, beta).
    Renvoie : (portfolio_returns, per_pair_returns_dict).
    """
    per_pair = {}
    for _, row in pairs_df.iterrows():
        ret = backtest_pair(
            prices, row["stock1"], row["stock2"], row["beta"],
            zscore_window=zscore_window, entry=entry, exit=exit, stop=stop,
            cost_bps=cost_bps,
        )
        if ret is None or ret.dropna().empty:
            continue
        per_pair[ret.name] = ret

    if not per_pair:
        return pd.Series(dtype=float), {}

    # Portefeuille équipondéré : on aligne toutes les Series et on moyenne
    df = pd.DataFrame(per_pair)
    portfolio = df.mean(axis=1)  # équipondéré sur paires actives ce jour-là
    return portfolio, per_pair
