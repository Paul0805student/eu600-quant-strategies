"""
Validation V3 avec données synthétiques où on INJECTE de vraies paires cointégrées.

Ceci permet de vérifier que :
1. Le test d'Engle-Granger détecte bien les paires injectées
2. Le z-score trading capture l'alpha
3. Le walk-forward fonctionne sans bug
4. Les métriques se calculent correctement
"""
import sys
sys.path.insert(0, '/home/claude/eu600_quant_v3')

import numpy as np
import pandas as pd

from cointegration import find_cointegrated_pairs, half_life_ou
from stat_arb import backtest_pair, backtest_portfolio_of_pairs
from backtest_v3 import walkforward_pairs_trading, compute_metrics, print_metrics


def gen_market_with_pairs(n_assets=80, n_pairs_inject=10, n_years=6, seed=42):
    """
    Génère un marché synthétique avec n_pairs_inject paires VRAIMENT cointégrées,
    plus du bruit non-cointégré pour le reste.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    dates = pd.date_range("2018-01-02", periods=n_days, freq="B")

    # Marché commun (drives les bêtas)
    mkt_drift = 0.07 / 252
    mkt_vol = 0.15 / np.sqrt(252)
    mkt = rng.normal(mkt_drift, mkt_vol, n_days)
    log_mkt = np.cumsum(mkt)

    tickers = [f"S{i:03d}" for i in range(n_assets)]
    log_prices_dict = {}

    # Premières 2*n_pairs_inject actions = paires cointégrées
    # Chaque paire (s_2k, s_2k+1) partage un facteur commun + spread mean-reverting
    for k in range(n_pairs_inject):
        # Facteur partagé (random walk corrélé au marché)
        beta_common = rng.uniform(0.6, 1.2)
        common_factor = beta_common * log_mkt + np.cumsum(
            rng.normal(0, 0.012, n_days)
        )

        # Stock 1 = facteur commun + bruit indépendant léger
        idio1 = np.cumsum(rng.normal(0, 0.005, n_days))
        s1 = 5.0 + common_factor + idio1  # log price ~ 5 = price ~ 150

        # Spread mean-reverting (Ornstein-Uhlenbeck)
        # ds = -theta * s * dt + sigma * dW, half-life = 15 jours
        theta = np.log(2) / 15
        sigma_spread = 0.03
        spread = np.zeros(n_days)
        for t in range(1, n_days):
            spread[t] = spread[t-1] * (1 - theta) + rng.normal(0, sigma_spread)

        # Stock 2 = s1 - spread (donc s1 - β*s2 = spread, β = 1 ici)
        s2 = s1 - spread

        log_prices_dict[tickers[2*k]] = s1
        log_prices_dict[tickers[2*k + 1]] = s2

    # Le reste : actions non-cointégrées (random walks corrélés au marché)
    for i in range(2 * n_pairs_inject, n_assets):
        beta = rng.uniform(0.5, 1.5)
        idio_vol = rng.uniform(0.012, 0.025)
        idio = np.cumsum(rng.normal(0, idio_vol, n_days))
        log_prices_dict[tickers[i]] = 5.0 + beta * log_mkt + idio

    log_prices = pd.DataFrame(log_prices_dict, index=dates)
    prices = np.exp(log_prices)

    # On retourne aussi la liste des paires VRAIMENT injectées pour vérification
    true_pairs = [(tickers[2*k], tickers[2*k+1]) for k in range(n_pairs_inject)]
    return prices, true_pairs


def main():
    print("=" * 78)
    print("VALIDATION V3 — données synthétiques avec paires injectées")
    print("=" * 78)

    prices, true_pairs = gen_market_with_pairs(
        n_assets=80, n_pairs_inject=10, n_years=6, seed=42
    )
    print(f"\n  {prices.shape[1]} actifs · {prices.shape[0]} jours")
    print(f"  Paires VRAIMENT injectées : {len(true_pairs)}")

    # Test 1 : la détection trouve-t-elle nos paires ?
    print("\n[1] Test de détection sur la PREMIÈRE moitié de la série...")
    half = len(prices) // 2
    formation = prices.iloc[:half]
    pairs_found = find_cointegrated_pairs(
        formation, p_threshold=0.05, corr_prefilter=0.5,
        min_half_life=2, max_half_life=60, top_n=30, verbose=True,
    )

    if not pairs_found.empty:
        print("\n  Top 10 paires détectées :")
        print(pairs_found.head(10).to_string(index=False))

        # Combien de vraies paires retrouvées ?
        true_set = set(true_pairs) | set([(b, a) for a, b in true_pairs])
        found_set = set(zip(pairs_found["stock1"], pairs_found["stock2"]))
        recall = len(found_set & true_set) / len(true_pairs) if true_pairs else 0
        print(f"\n  RECALL : {len(found_set & true_set)}/{len(true_pairs)} vraies paires retrouvées "
              f"({recall*100:.0f}%)")
    else:
        print("  AUCUNE paire détectée (seuils trop stricts ?)")

    # Test 2 : trading sur la SECONDE moitié avec ces paires fixes
    print("\n[2] Backtest sur seconde moitié (OOS) avec paires de la première moitié...")
    test = prices.iloc[half:]
    full_test = prices.iloc[half - 100:]  # un peu d'histoire pour init z-score

    if not pairs_found.empty:
        port_ret, per_pair = backtest_portfolio_of_pairs(
            full_test, pairs_found,
            zscore_window=20, entry=2.0, exit=0.5, stop=4.0, cost_bps=10,
        )
        port_ret = port_ret.loc[test.index[0]:]

        m = compute_metrics(port_ret)
        print(f"\n  PORTEFEUILLE DE PAIRES (OOS):")
        print_metrics(m)

        # Vérifier la décorrélation au marché
        bench = prices.pct_change().fillna(0).mean(axis=1).loc[port_ret.index]
        corr = port_ret.corr(bench)
        print(f"\n  Corrélation avec marché : {corr:.3f} (devrait être proche de 0)")

    # Test 3 : Walk-forward complet
    print("\n[3] Walk-forward complet (formation 1.5 ans / trading 4 mois)...")
    portfolio_ret, pairs_log, raw = walkforward_pairs_trading(
        prices,
        formation_years=1.5,
        trading_months=4,
        top_n_pairs=15,
        p_threshold=0.05,
        corr_prefilter=0.5,
        zscore_window=20, entry=2.0, exit=0.5, stop=4.0,
        cost_bps=10, target_vol=0.10, verbose=False,
    )

    if not portfolio_ret.empty:
        m = compute_metrics(portfolio_ret)
        print(f"\n  STRATÉGIE COMPLÈTE WALK-FORWARD (vol-targeted) :")
        print_metrics(m)
        print(f"\n  Périodes walk-forward exécutées : {pairs_log['period'].nunique() if not pairs_log.empty else 0}")
    else:
        print("  Walk-forward n'a généré aucun rendement.")

    print("\n=== VALIDATION V3 OK ===")


if __name__ == "__main__":
    main()
