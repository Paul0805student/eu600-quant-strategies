# Guide complet — Tester les stratégies avec VS Code + GitHub

Tu as maintenant 4 frameworks de stratégies quant (V1, V2, V3, V4). Voici comment les déployer proprement chez toi avec VS Code, créer un repo GitHub, et les exécuter.

---

## Partie 1 — Mise en place initiale (à faire UNE fois)

### 1.1 Vérifier les prérequis

Ouvre **PowerShell** (Win+R, tape `powershell`) et vérifie :

```powershell
python --version    # doit afficher 3.9 ou plus (idéalement 3.11+)
git --version       # doit afficher git 2.x
code --version      # doit afficher la version de VS Code
```

Si l'un manque :
- **Python 3.11+** : https://www.python.org/downloads/ (coche bien "Add to PATH")
- **Git** : https://git-scm.com/download/win
- **VS Code** : https://code.visualstudio.com/

### 1.2 Extensions VS Code recommandées

Dans VS Code, ouvre l'onglet Extensions (Ctrl+Shift+X) et installe :
- **Python** (Microsoft) — interpréteur, debugger, IntelliSense
- **Pylance** (Microsoft) — type-checking et autocomplétion
- **Jupyter** (Microsoft) — pour exécuter du code cellule par cellule
- **GitLens** (GitKraken) — visualisation de l'historique Git
- **GitHub Pull Requests** (GitHub) — gérer les PRs depuis VS Code

### 1.3 Configurer Git (si pas déjà fait)

```powershell
git config --global user.name "John"
git config --global user.email "ton.email@example.com"
```

---

## Partie 2 — Créer le repo GitHub

### 2.1 Sur GitHub.com

1. Va sur https://github.com → bouton vert **"New"** (en haut à gauche)
2. Repository name : `eu600-quant-strategies`
3. Description : "Stratégies quantitatives long-only et market-neutral sur STOXX Europe 600"
4. **Private** (recommandé pour du code de stratégie)
5. **Coche** "Add a README file"
6. Add .gitignore : choisis **Python**
7. License : MIT (ou aucune si privé)
8. **Create repository**

### 2.2 Cloner en local

Dans VS Code : **View → Command Palette** (Ctrl+Shift+P), tape `Git: Clone`, colle l'URL HTTPS de ton repo (ex: `https://github.com/waoile45/eu600-quant-strategies.git`), choisis un dossier (par exemple `C:\Users\John\Projects\`).

VS Code te propose d'ouvrir le repo cloné — accepte.

---

## Partie 3 — Intégrer les frameworks

### 3.1 Structure recommandée du repo

Dans le terminal intégré de VS Code (**Terminal → New Terminal**, ou Ctrl+`), crée la structure :

```powershell
mkdir v1_factor_long_only
mkdir v2_long_only_alt
mkdir v3_stat_arb
mkdir v4_ml_meta_labeling
mkdir docs
mkdir notebooks
```

### 3.2 Copier les fichiers

Place les fichiers de chaque V dans son dossier respectif :

```
eu600-quant-strategies/
├── v1_factor_long_only/      ← contenu de eu600_quant/
├── v2_long_only_alt/          ← contenu de eu600_quant_v2/
├── v3_stat_arb/               ← contenu de eu600_quant_v3/
├── v4_ml_meta_labeling/       ← contenu de eu600_quant_v4/
├── docs/                      ← ce guide + autres docs
├── notebooks/                 ← Jupyter pour analyses ad hoc
├── .gitignore                 ← déjà créé par GitHub
├── README.md                  ← description du repo (à éditer)
└── requirements.txt           ← dépendances communes
```

### 3.3 Compléter le .gitignore

Ouvre `.gitignore` dans VS Code et **ajoute en bas** :

```gitignore
# Données et caches
data_cache/
outputs/
outputs_v2/
outputs_v3/
outputs_v4/
*.parquet
*.csv

# Environnement virtuel
.venv/
venv/

# VS Code
.vscode/
.idea/

# Système
.DS_Store
Thumbs.db
```

⚠️ **Important** : on ne commit JAMAIS les données brutes ni les outputs (le repo grossirait sans raison et tu pourrais publier des résultats sensibles). Le code suffit, les autres peuvent re-télécharger les données.

### 3.4 Créer un requirements.txt commun

À la racine du repo, crée `requirements.txt` :

```
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
tqdm>=4.65.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
pyarrow>=14.0.0
```

---

## Partie 4 — Environnement Python isolé

### 4.1 Créer un virtualenv

Dans le terminal VS Code, à la racine du repo :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si l'activation est bloquée :
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
puis réessayer `Activate.ps1`.

Tu vois maintenant `(.venv)` au début de la ligne de commande.

### 4.2 Installer les dépendances

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Ça prend 2-5 minutes. Si une erreur sur `lightgbm` (Visual C++ requis sur Windows) :
- Soit installer Visual C++ Redistributable
- Soit retirer `lightgbm` du requirements (pas critique, on a sklearn)

### 4.3 Sélectionner l'interpréteur dans VS Code

`Ctrl+Shift+P` → tape `Python: Select Interpreter` → choisis celui qui contient `.venv` (l'icône a une petite étoile).

VS Code utilisera désormais cet env pour exécuter et débugger.

---

## Partie 5 — Tester chaque stratégie

### 5.1 Test rapide V1 (factor long-only)

```powershell
cd v1_factor_long_only
python check_setup.py    # vérifie que tout est en place
python main.py            # lance le backtest complet
```

Premier run : **5-10 min** (téléchargement Yahoo de ~250 tickers). Les suivants : 30s grâce au cache parquet.

Sortie : `v1_factor_long_only/outputs/equity_curves.png` + `summary.csv`.

### 5.2 Test V2 (long-only divers)

```powershell
cd ..\v2_long_only_alt
python main_v2.py
```

⚠️ L'idiosyncratic momentum est lent (~1 min de calcul de régressions CAPM glissantes).

### 5.3 Test V3 (stat arb)

```powershell
cd ..\v3_stat_arb
python main_v3.py
```

⚠️ **Très lourd** : 1-3h sur le full universe à cause du test exhaustif des paires. Pour un test rapide, édite `main_v3.py` :
```python
MIN_HISTORY_DAYS = 750     # au lieu de 1000
TOP_N_PAIRS = 10           # au lieu de 20
TRADING_MONTHS = 12        # au lieu de 6 (moins de re-sélections)
```
Ça descend à ~15 min.

### 5.4 Test V4 (ML)

```powershell
cd ..\v4_ml_meta_labeling
python main_v4.py
```

Durée : ~10-20 min selon la machine (le panel de features est gros).

---

## Partie 6 — Workflow VS Code efficace

### 6.1 Debugger interactivement

Place un breakpoint en cliquant à gauche du numéro de ligne dans `main.py` (point rouge).

`F5` lance le debugger. Tu peux inspecter les variables, exécuter pas à pas (F10), entrer dans les fonctions (F11), continuer (F5).

Très utile pour comprendre comment les signaux sont calculés.

### 6.2 Notebooks Jupyter pour exploration

Dans VS Code, crée un fichier `notebooks/explore_v1.ipynb`. Tu peux importer ton code et explorer interactivement :

```python
# Cellule 1
import sys
sys.path.insert(0, '../v1_factor_long_only')
from main import main
strategies, summary = main()

# Cellule 2 (après le run)
summary

# Cellule 3
import matplotlib.pyplot as plt
strategies['ENSEMBLE (composite)'].cumsum().plot(figsize=(12,5))
plt.title("Equity ensemble V1")
plt.show()
```

### 6.3 Exécution de cellules dans un .py

Tu n'es pas obligé d'utiliser des notebooks. Dans n'importe quel fichier `.py`, ajoute :
```python
# %%
import pandas as pd
df = pd.read_csv('outputs/strategy_returns.csv')

# %%
df.head()
```

VS Code propose un bouton "Run Cell" au-dessus de chaque `# %%`. Tu obtiens une console interactive sans avoir besoin de Jupyter.

---

## Partie 7 — Workflow Git basique

### 7.1 Vérifier l'état

```powershell
git status      # voir les changements
git diff        # voir les diffs détaillés
```

Ou dans VS Code : onglet Source Control (Ctrl+Shift+G) — tout est visuel.

### 7.2 Commiter

Dans VS Code Source Control :
1. Clique sur `+` à côté des fichiers à inclure (stage)
2. Tape un message de commit (ex: "Add V4 meta-labeling pipeline")
3. Ctrl+Enter pour commit

En ligne de commande équivalente :
```powershell
git add .
git commit -m "Add V4 meta-labeling pipeline"
git push
```

### 7.3 Branches pour expérimenter

Avant de modifier une stratégie qui marche, crée une branche :
```powershell
git checkout -b feature/v5-pairs-johansen
# ... fais tes modifs ...
git add .
git commit -m "Try Johansen multivariate cointegration"
git push -u origin feature/v5-pairs-johansen
```

Sur GitHub, tu peux ouvrir une Pull Request pour merger dans `main` quand c'est validé.

### 7.4 Tags pour les versions stables

Quand un V donne de bons résultats OOS :
```powershell
git tag -a v1.0-stable -m "V1 with Sharpe OOS = 1.15 on 2020-2024"
git push --tags
```

Tu peux toujours revenir à cette version même si tu casses tout après.

---

## Partie 8 — Métriques à surveiller

Pour chaque stratégie, regarde dans `summary.csv` (ou similaire) :

| Métrique | Interprétation |
|---|---|
| **Sharpe OOS** | Cible ≥ 1. Critique. |
| **MaxDD OOS** | Devrait être < 25% pour du long-only avec vol target 10% |
| **Sortino OOS** | Si > Sharpe, asymétrie favorable (bon) |
| **Calmar OOS** | CAGR / |MaxDD|. > 0.5 = correct |
| **Hit rate mensuel** | > 55% = signal robuste |
| **PSR** (V2-V4) | > 0.95 = très fort |
| **Corrélation marché** (V3) | < 0.2 = market-neutral OK |
| **Sharpe par fold** (V2) | Stabilité dans le temps |

**Drapeau rouge** : Sharpe IS très élevé (>2) mais OOS faible (<0.5) → **overfitting**.
**Drapeau vert** : Sharpe IS et OOS proches, MaxDD raisonnable, courbe d'equity régulière.

---

## Partie 9 — Pièges courants à éviter

### 9.1 Look-ahead bias
Tu utilises de l'information future sans le savoir. Symptômes : Sharpe IS irréaliste (>3). 
**Vérification** : à n'importe quelle date t, est-ce que mon signal n'utilise QUE des données ≤ t ? Le `shift(1)` partout dans nos pipelines protège contre ça.

### 9.2 Survivorship bias
Tu n'utilises que les actions qui existent aujourd'hui — tu rates les faillites. 
**Vérification** : on l'admet honnêtement, c'est dans tous les README. Pour aller plus loin, il faudrait une source comme Bloomberg ou Refinitiv pour la composition historique.

### 9.3 Data snooping (multiple testing)
Tu testes 100 paramètres et tu prends celui avec le meilleur Sharpe → biais de sélection énorme.
**Vérification** : nos paramètres sont fixes (issus de la littérature), pas optimisés sur les données. Si tu commences à ajuster les hyperparams pour booster le Sharpe, applique la **Deflated Sharpe Ratio** (López ch.14).

### 9.4 Overfitting du ML
Le RF apprend par cœur les patterns du train.
**Vérification** : Sharpe train >> test. Solution : réduire `max_depth`, augmenter `min_samples_leaf`, ajouter des purges plus fortes.

### 9.5 Coûts sous-estimés
Tu trades 100 lignes par jour à 5 bps mais le vrai coût avec slippage est 25 bps.
**Vérification** : nos pipelines utilisent 10 bps par côté = 20 bps round-trip, réaliste pour des grandes caps européennes. Sur des smid caps, monte à 30-50 bps.

---

## Partie 10 — Pistes d'amélioration

Une fois tes tests faits :

1. **Comparer les 4 V** : lance les 4, compare leurs Sharpe OOS, leurs corrélations entre eux
2. **Combiner** : portefeuille équipondéré V1+V2+V3+V4, vol-targeted globalement → souvent Sharpe combiné > meilleur Sharpe individuel
3. **Tester d'autres univers** : S&P 500, MSCI World, marchés émergents
4. **Tester d'autres horizons** : weekly rebalance vs mensuel, intraday avec données 1min
5. **Ajouter des features fondamentales** : P/E, P/B, ROE (nécessite données type Compustat ou simfin.com)
6. **Forward test** : laisse tourner en paper trading pendant 6 mois pour voir si les stats OOS tiennent

---

## Partie 11 — Aide quand quelque chose casse

### Erreur d'import
```
ModuleNotFoundError: No module named 'yfinance'
```
→ Tu n'as pas activé le venv. `.\.venv\Scripts\Activate.ps1`

### Téléchargement Yahoo échoue
```
HTTP Error 429: Too Many Requests
```
→ Yahoo te rate-limite. Attends 5-10 min, ou réduis `batch_size` dans `data.py`.

### Pandas warning sur `'M'`
```
FutureWarning: 'M' is deprecated...
```
→ Tu as une vieille version. Code déjà compatible mais tu peux upgrader : `pip install --upgrade pandas`.

### Out of memory
→ Réduis le nombre de tickers ou la période. Pour V3 surtout : `MIN_HISTORY_DAYS = 750` et `TOP_N_PAIRS = 10`.

### Le run V1 prend 30 min
→ C'est le téléchargement initial. Le cache parquet rend les runs suivants instantanés. Vérifie que `data_cache/` contient un fichier `.parquet`.

---

## Récapitulatif des Sharpe OOS attendus

| Stratégie | Sharpe OOS attendu | Source |
|---|---|---|
| V1 ENSEMBLE composite | 1.0 - 1.4 | Pondération de 4 facteurs robustes |
| V2 ENSEMBLE V2 | 1.0 - 1.4 | Idio Mom + 52W + TSMOM + Bollinger |
| V3 Stat Arb | 0.6 - 1.2 | Walk-forward réaliste |
| V4 Meta-Labeled | 0.9 - 1.4 | Sur le sous-ensemble filtré ML |
| **V1+V2+V3+V4 combo** | **1.3 - 1.8** | Diversification multi-strat |

Si tu n'atteins pas Sharpe ≥ 1 sur au moins une stratégie, c'est probablement un problème de :
1. Période OOS trop courte (au moins 3 ans nécessaires)
2. Bear market 2022 dans l'OOS qui pénalise les long-only (V1, V2)
3. Coûts trop élevés pour ton broker (passer à 5 bps si tu as un compte premium)

Bon trading. ✓
