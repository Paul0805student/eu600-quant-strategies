# EU600 Quant Strategies — V2

Second framework long-only, **complètement distinct du V1**, basé sur 4 stratégies tirées directement de la littérature fournie.

## Différences vs V1

| Aspect | V1 | V2 |
|---|---|---|
| Stratégie 1 | Momentum 12-1 (Jegadeesh-Titman) | **TSMOM** (Moskowitz, Ooi, Pedersen 2012) |
| Stratégie 2 | Low Volatility (Frazzini-Pedersen) | **52-Week High** (George & Hwang 2004) |
| Stratégie 3 | Reversal 5j (Lehmann) | **Idiosyncratic Momentum** (Blitz, Huij, Martens 2011) |
| Stratégie 4 | Quality/Trend (custom) | **Bollinger Mean Reversion** (Chan ch.4) |
| Validation | Split IS/OOS simple | Split IS/OOS + **Walk-Forward 5 folds** |
| Métrique sup. | — | **Probabilistic Sharpe Ratio** (Bailey & López de Prado) |

## Les 4 stratégies en détail

### 1. Time-Series Momentum (TSMOM)
**Source** : Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum", cité dans Cartea ch.10.

Pour **chaque action** : si rendement 12 mois > 0, on tient long ; sinon flat. Pas de comparaison cross-section. Fondamentalement différent du momentum classique : c'est du **trend-following absolu**, pas relatif. Les positions sont équipondérées sur les actions à signal positif. Quand le marché plonge globalement, le portefeuille passe en cash → forte protection contre les bear markets.

### 2. 52-Week High (Nearness to High)
**Source** : George & Hwang (2004), cité dans Chan "Algorithmic Trading".

Signal = `prix_actuel / max(prix_252j)`. Plus on est proche du plus haut 52 semaines, plus le signal est fort. Surperforme historiquement le momentum classique (12-1) car capte un **biais d'ancrage** : les investisseurs hésitent à acheter une action proche de son plus haut, créant une underreaction exploitable.

### 3. Idiosyncratic Momentum (Residual Momentum)
**Source** : Blitz, Huij & Martens (2011), principe central dans López de Prado ch.8.

On régresse chaque action sur le marché en CAPM glissant 252j, on extrait les résidus (= rendement idiosyncratique pur), puis on calcule le momentum 12-1 sur ces résidus. Élimine le bêta indésirable et **réduit drastiquement les momentum crashes** (Daniel & Moskowitz 2016). Sharpe historique ~0.85-1.0 sur actions européennes.

### 4. Bollinger Mean Reversion (long-only)
**Source** : Ernest Chan "Algorithmic Trading: Winning Strategies and Their Rationale" ch.4.

Signal = `(MA20 - prix) / (2σ20)`. Positif = oversold, on entre. On longe les actions les plus oversold en cross-section, en pariant sur un retour à la moyenne sur 5-20 jours. **Diversifie les 3 autres** (orthogonal au momentum).

## Pondérations du composite

```python
composite = 0.20 × TSMOM
          + 0.25 × 52W High
          + 0.35 × Idio Momentum    # le plus robuste académiquement
          + 0.20 × Bollinger MR
```

## Améliorations méthodologiques (López de Prado)

- **Walk-Forward Analysis** (ch.7) : 5 folds successifs, évalue la stabilité du Sharpe dans le temps
- **Probabilistic Sharpe Ratio** (ch.14) : P(vrai Sharpe > 1) ajustée pour skew/kurtosis
- **Embargo** : prêt pour ajouter période d'embargo entre folds train/test

## Architecture

```
eu600_quant_v2/
├── universe.py            # repris du V1 (345 tickers STOXX 600)
├── data.py                # repris du V1 (téléchargement + cache parquet)
├── signals_v2.py          # 4 nouveaux signaux + combine + weight conversion
├── backtest_v2.py         # moteur + walk-forward + PSR
├── main_v2.py             # pipeline principal
├── test_validation_v2.py  # validation logique (synthétique)
└── outputs_v2/            # résultats générés
```

## Exécution (Windows / PowerShell)

```powershell
cd eu600_quant_v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main_v2.py
```

⚠️ **L'idiosyncratic momentum est le calcul le plus lourd** (~1 min sur 250 actions × 2500 jours) : régression CAPM glissante par actif. Le cache yfinance accélère les runs suivants.

## Sorties (`outputs_v2/`)

- `equity_curves_v2.png` — courbes equity log + drawdown ENSEMBLE
- `strategy_returns_v2.csv` — rendements quotidiens
- `summary_v2.csv` — Sharpe IS/OOS, CAGR, MaxDD, **PSR** par stratégie
- `walkforward_v2.csv` — performance par fold

## Niveau de Sharpe attendu (hypothèses réalistes)

| Stratégie | Sharpe OOS attendu | Source |
|---|---|---|
| TSMOM seul | 0.4 - 0.7 | Moskowitz 2012 (multi-asset 0.6) |
| 52W High | 0.5 - 0.9 | George & Hwang 2004 |
| Idio Momentum | 0.7 - 1.0 | Blitz 2011 (EU stocks ~0.85) |
| Bollinger MR | 0.3 - 0.6 | Chan, diversifie le momentum |
| **ENSEMBLE V2 vol-managed** | **1.0 - 1.4** | combinaison + filtre régime |

## Pour aller plus loin

- **Triple Barrier Method + Meta-Labeling** (López de Prado ch.3) : labelliser les events et entraîner un classifier ML pour filtrer les signaux faibles
- **Fractional Differentiation** (ch.5) : features stationnaires gardant la mémoire long-terme
- **CUSUM filter** (ch.2) : réduire le sample bias en ne tradant qu'aux events significatifs
- **Pairs trading** (Cartea ch.7) : ajouter une stratégie dollar-neutral cointegration
