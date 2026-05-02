# EU600 Quant Strategies

Framework de stratégies quantitatives sur le STOXX Europe 600, avec backtest rigoureux IS/OOS visant un Sharpe out-of-sample ≥ 1.

## Architecture

```
eu600_quant/
├── universe.py      # Liste curated de ~250 actions STOXX 600 (tickers Yahoo)
├── data.py          # Téléchargement yfinance + cache parquet
├── strategies.py    # 4 signaux factoriels + composite
├── backtest.py      # Moteur (coûts, lag, vol targeting, IS/OOS split)
├── main.py          # Pipeline complet
└── outputs/         # Graphiques + CSV générés
```

## Stratégies implémentées

1. **Momentum 12-1** (Jegadeesh & Titman 1993, Asness 2013) — return 12 mois en sautant le dernier mois
2. **Low Volatility** (Frazzini & Pedersen 2014) — long les actions à faible vol 60j
3. **Reversal court terme** (Lehmann 1990) — long les perdants à 5 jours
4. **Quality / Trend** — long les actions proches de leur plus haut 12 mois
5. **ENSEMBLE composite** — combinaison pondérée des 4 (35% LV / 30% Mom / 20% QT / 15% Rev)

Tous les signaux sont z-scorés en cross-section, écrêtés à ±3σ.

## Améliorations clés pour le Sharpe OOS

- **Filtre de régime** : exposition réduite à 30% quand l'index est sous sa SMA 200j
- **Vol targeting** à 10% annuel (levier capé à 2x)
- **Coûts réalistes** : 10 bps par côté
- **Long-only top quintile** : structurellement plus stable que long-short en Europe
- **Lag d'exécution** d'1 jour : pas de look-ahead

## Installation (Windows / PowerShell)

```powershell
cd eu600_quant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si l'activation est bloquée par la policy :
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Exécution

```powershell
python main.py
```

Le premier run télécharge ~250 tickers (peut prendre 5-10 min). Les runs suivants utilisent le cache `data_cache/`.

## Sorties

Dans `outputs/` :
- `equity_curves.png` — courbes d'equity log + drawdown ensemble
- `strategy_returns.csv` — rendements quotidiens nets par stratégie
- `summary.csv` — résumé Sharpe IS/OOS, CAGR, MaxDD

## Limites honnêtes (à corriger pour de la prod)

1. **Survivorship bias** : on utilise la composition actuelle du STOXX 600, pas l'historique des entrées/sorties. Pour un travail rigoureux, ré-importer les compositions historiques (Bloomberg/Refinitiv).
2. **Pas de borrowing/funding cost** : si tu actives `LONG_SHORT=True`, ajoute un coût de portage côté short (~2-5% annuel en Europe).
3. **Liquidité** : on suppose qu'on peut entrer/sortir au close. Pour les positions concentrées, ajouter un modèle d'impact de marché (cf. Almgren-Chriss, Kissell ch. 4-5).
4. **Single asset class** : pour augmenter encore le Sharpe, combiner avec FX, taux, vol carry (cf. Asness "Value & Momentum Everywhere").
5. **Pas de neutralisation sectorielle** : un screen low-vol peut surcharger les utilities/consumer staples. Ajouter une contrainte de neutralité GICS.

## Pistes pour améliorer encore

- **Volatility scaling cross-sectional** : pondérer chaque position par 1/vol_i (risk parity intra-portefeuille)
- **Multi-horizon momentum** : combiner 1m / 3m / 12m
- **Earnings momentum / quality fondamental** : nécessite données fondamentales
- **Gestion explicite des crashes momentum** (cf. Daniel & Moskowitz 2016)
- **ML stacking** : utiliser GBM/Random Forest pour combiner les signaux non-linéairement (cf. *Advances in Financial ML*, López de Prado)
