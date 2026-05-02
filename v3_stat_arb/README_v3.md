# EU600 Quant Strategies — V3 (Statistical Arbitrage)

Troisième framework, **radicalement différent des V1 et V2** : market-neutral et dollar-neutral par construction. Stratégie phare de Chan ch.5-6 et Cartea ch.7.

## Principe

**Pairs Trading par cointégration** : on identifie des paires d'actions dont les prix bougent ensemble à long terme (cointégrées au sens d'Engle-Granger). Le **spread** entre les deux suit alors un processus mean-reverting (Ornstein-Uhlenbeck). On parie sur le retour à la moyenne :
- Quand le spread s'écarte (+2σ) → on **shorte** le spread
- Quand il revient près de zéro → on **ferme**
- Quand il dérape (>4σ) → **stop-loss** (cointégration cassée)

Comme on est long une jambe et short l'autre, le portefeuille a un **bêta marché ~0** : il gagne (ou perd) indépendamment de la direction du marché. C'est l'archétype de la stratégie quant institutionnelle.

## Différences vs V1/V2

| Aspect | V1 / V2 | V3 |
|---|---|---|
| Direction | Long-only | **Market-neutral** (long + short) |
| Cible | Capter des primes factorielles | **Mean reversion de spreads** |
| Beta marché | ~1 (long-only) | **~0** (cointégration neutralise) |
| Univers utilisé | Top quintile sur signal | **Paires sélectionnées** (~20 par période) |
| Métrique clé | Sharpe vs CAGR | **Sharpe** + corrélation marché ~0 |
| Méthodologie | Split IS/OOS unique | **Walk-forward réel** avec re-sélection |
| Rebalancement | Mensuel | Continu sur z-score, re-sélection paires tous les 6 mois |

## Architecture

```
eu600_quant_v3/
├── universe.py            # repris du V1
├── data.py                # repris du V1
├── cointegration.py       # test Engle-Granger + half-life OU
├── stat_arb.py            # logique de trading par paire
├── backtest_v3.py         # walk-forward + métriques (PSR)
├── main_v3.py             # pipeline principal
└── test_validation_v3.py  # validation avec paires synthétiques injectées
```

## Méthodologie en détail

### 1. Sélection des paires (formation period, 2 ans rolling)

**Étape A — Pré-filtre par corrélation** : on ne garde que les paires dont les log-prix normalisés ont une corrélation > 0.50. Cela réduit le nombre de tests d'~30 000 paires possibles à quelques centaines, limitant le **multiple testing problem** (López de Prado ch.11).

**Étape B — Test d'Engle-Granger** : pour chaque paire candidate :
1. Régresser `log(P1) = α + β·log(P2) + ε`
2. Tester si les résidus `ε` sont stationnaires (ADF)
3. Si p-value < 0.05 → cointégration confirmée

**Étape C — Half-life d'Ornstein-Uhlenbeck** : sur le spread, estimer `Δs = -θ·s + ε`. Half-life = `ln(2)/θ`. On garde les paires avec `2 ≤ HL ≤ 60` jours :
- HL < 2j = bruit, pas exploitable (whipsaw + coûts)
- HL > 60j = trop lent, capital immobilisé

**Étape D** : top 20 paires par p-value croissante.

### 2. Trading (period suivante, 6 mois)

Pour chaque paire `(i, j)` avec hedge ratio `β` :
- **Spread** : `s_t = log(P_i,t) - β · log(P_j,t)`
- **Z-score glissant** : `z_t = (s_t - μ_20) / σ_20`
- **Règles** :
  - `z > +2` → SHORT spread (sell P_i, buy β·P_j)
  - `z < -2` → LONG spread (buy P_i, sell β·P_j)
  - `|z| < 0.5` → CLOSE
  - `|z| > 4` → STOP-LOSS (cointégration cassée)
- **Lag** : exécution à t+1 (pas de look-ahead)

Le portefeuille global est **équipondéré** sur les ~20 paires actives.

### 3. Walk-forward (López de Prado ch.7)

À la fin des 6 mois de trading, on **re-sélectionne** des paires sur les 2 années précédentes (qui incluent maintenant les 6 derniers mois). Cela teste la **robustesse temporelle** : les paires identifiées en 2018 trade-t-elles bien en 2019 ? Celles de 2019 en 2020 ? etc.

**Tous les rendements générés sont OOS par construction** — il n'y a pas de période in-sample séparée. C'est la procédure la plus rigoureuse possible.

## Validation logique (avec paires synthétiques)

J'ai injecté 10 paires véritablement cointégrées dans 80 actifs synthétiques, avec des spreads OU de half-life 15 jours. Résultats :

| Test | Résultat |
|---|---|
| Recall détection | **40%** des vraies paires retrouvées |
| OOS test simple | **Sharpe = 1.57**, MaxDD = -5.8% |
| Corrélation marché | **-0.014** ✓ (market-neutral confirmé) |
| Walk-forward complet | **Sharpe = 0.86** (plus réaliste avec re-sélection) |

## Performance attendue sur STOXX 600 réel

D'après les benchmarks académiques (Gatev, Goetzmann & Rouwenhorst 2006 ; Do & Faff 2010) :

| Période | Sharpe attendu | Notes |
|---|---|---|
| 1990-2000 | 1.5 - 2.5 | Âge d'or de la stratégie |
| 2000-2010 | 0.8 - 1.4 | Diminution avec démocratisation |
| 2010-2025 | **0.6 - 1.2** | Encore exploitable, surtout au sein de secteurs |

Notre cible **Sharpe OOS ≥ 1** est atteignable, surtout grâce à :
- L'**Europe moins arbitrée** que les US (plus d'inefficiences)
- Le **walk-forward** qui force l'adaptation
- Le **vol targeting** qui lisse les performances

## Exécution

```powershell
cd eu600_quant_v3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install statsmodels
python main_v3.py
```

⚠️ **Le calcul est lourd** : pour 250 actions, ~30 000 paires possibles, on en teste plusieurs centaines après pré-filtre. Comptez 5-15 min par période walk-forward × ~15 périodes = 1-3 heures sur le run complet. Les runs suivants utilisent le cache yfinance.

Pour un test rapide : réduire `MIN_HISTORY_DAYS` à 750 et `TOP_N_PAIRS` à 10.

## Sorties (`outputs_v3/`)

- `equity_curves_v3.png` — equity curve + drawdown
- `strategy_returns_v3.csv` — rendements quotidiens (vol-targeted, raw, benchmark)
- `summary_v3.csv` — métriques complètes (Sharpe, Sortino, Calmar, PSR, MaxDD)
- `pairs_log_v3.csv` — toutes les paires sélectionnées par période (audit trail)

## Pour aller plus loin

- **Cointégration multivariate (Johansen)** : groupes de 3-5 actions au lieu de paires (Chan ch.6)
- **Kalman filter** pour β dynamique : le hedge ratio évolue dans le temps (Cartea ch.7)
- **Filtres ML** : entraîner un classifier pour prédire si un signal de spread va être profitable (López de Prado meta-labeling)
- **Capital allocation** non-uniforme : pondérer les paires par Sharpe IS / par p-value (mais attention au sur-fitting)
- **Sector-only pairs** : ne tester que les paires intra-secteur (GICS) → réduit le multiple testing et augmente la probabilité de vraie cointégration économique

## Combinaison V1 + V2 + V3

Comme V3 est market-neutral et **non-corrélé** à V1/V2 (qui sont long-only), un portefeuille **équipondéré** sur les trois aurait théoriquement :
- Sharpe combiné > Sharpe individuel maximal (par diversification)
- Drawdown combiné < drawdown individuel
- Exposition marché ~33% (V3 neutralise un tiers du beta)

C'est l'architecture des fonds quant multi-strats sérieux.
