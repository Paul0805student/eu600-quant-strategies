"""
Pipeline V4 — Triple Barrier + Meta-Labeling avec ML
Source : López de Prado, "Advances in Financial Machine Learning" ch.3-7.

Méthodologie :
1. SIGNAL PRIMAIRE : momentum cross-section (long top quintile)
2. EVENTS : dates où on rentre en position selon le signal primaire
3. TRIPLE BARRIER : pour chaque event, label {-1, 0, +1} via path-based barriers
4. META-LABELS : binaire {0, 1} = trade gagnant ou non
5. FEATURES : panel de 18 features financières par actif
6. WALK-FORWARD : entraîne le ML sur passé, prédit sur futur (avec purge + embargo)
7. TRADING : ne prend le trade primaire QUE si ML.proba > seuil

L'idée clé : on ne change pas le signal primaire (qui définit la direction).
Le ML décide seulement si ce signal est "fiable" pour ce setup particulier.
Cela améliore la PRÉCISION sans sacrifier la robustesse.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

from universe import get_universe
from data import download_prices, filter_universe
from labels import get_events, get_daily_volatility
from features import compute_features_one_asset
from meta_labeling import (
    make_meta_labels, train_meta_model, evaluate_meta_model,
    feature_importance_mdi,
)


# ============================================================================
# CONFIG
# ============================================================================
START_DATE = "2014-01-01"
END_DATE = None
MIN_HISTORY_DAYS = 1000

# Signal primaire : momentum 12-1
MOM_LOOKBACK = 252
MOM_SKIP = 21
MOM_TOP_QUANTILE = 0.20

# Triple barrier
PT_MULT = 2.0     # take profit à 2σ
SL_MULT = 1.5     # stop loss à 1.5σ
TIMEOUT_DAYS = 21 # 1 mois timeout
VOL_SPAN = 60

# Meta-modèle
TRAIN_FRAC = 0.6  # 60% pour train
PROBA_THRESHOLD = 0.55  # seuil de proba pour prendre le trade

# Sizing
TARGET_VOL = 0.10
COST_BPS = 10

OUTPUT_DIR = Path(__file__).parent / "outputs_v4"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# SIGNAL PRIMAIRE : momentum 12-1 cross-section
# ============================================================================

def primary_signal_dates(prices, lookback=252, skip=21, top_q=0.20):
    """
    À chaque fin de mois, identifier les actions du top quintile momentum.
    Renvoie un dict {ticker: liste de dates où l'action est long-signal}.
    """
    log_p = np.log(prices)
    mom = log_p.shift(skip) - log_p.shift(lookback)

    # Rebal mensuel
    s = pd.Series(prices.index, index=prices.index)
    rebal_dates = s.groupby([s.index.year, s.index.month]).max().values

    signals_per_ticker = {tk: [] for tk in prices.columns}

    for date in rebal_dates:
        if date not in mom.index:
            continue
        row = mom.loc[date].dropna()
        if len(row) < 20:
            continue
        n_top = max(1, int(len(row) * top_q))
        top = row.nlargest(n_top).index
        for tk in top:
            signals_per_ticker[tk].append(date)

    # Convertir en Series par ticker
    return {tk: pd.DatetimeIndex(dates) for tk, dates in signals_per_ticker.items()}


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    print("=" * 78)
    print("PIPELINE V4 — TRIPLE BARRIER + META-LABELING")
    print("Méthodologie López de Prado")
    print("=" * 78)

    # ------------------------------------------------------------------------
    print("\n[1/6] Données...")
    tickers = get_universe()
    prices = download_prices(tickers, start=START_DATE, end=END_DATE)
    prices = filter_universe(prices, min_history_days=MIN_HISTORY_DAYS, min_price=1.0)
    print(f"  {prices.shape[1]} actifs · {prices.shape[0]} jours")

    # ------------------------------------------------------------------------
    print("\n[2/6] Signal primaire (momentum 12-1)...")
    signals = primary_signal_dates(prices, MOM_LOOKBACK, MOM_SKIP, MOM_TOP_QUANTILE)
    n_total_signals = sum(len(d) for d in signals.values())
    print(f"  Total events long générés : {n_total_signals}")

    # ------------------------------------------------------------------------
    print("\n[3/6] Triple Barrier Method (labellisation)...")
    all_events = []
    for ticker in tqdm(list(signals.keys()), desc="  Tickers"):
        dates = signals[ticker]
        if len(dates) == 0:
            continue
        p = prices[ticker].dropna()
        if len(p) < 252:
            continue
        events = get_events(
            p, dates,
            side="long",
            pt_mult=PT_MULT, sl_mult=SL_MULT,
            num_days=TIMEOUT_DAYS, vol_span=VOL_SPAN,
        )
        if events.empty:
            continue
        events["ticker"] = ticker
        all_events.append(events.reset_index().rename(columns={"index": "t0"}))

    events_df = pd.concat(all_events, ignore_index=True)
    print(f"  Events labellisés : {len(events_df)}")
    print(f"  Distribution des labels :")
    print(events_df["bin"].value_counts().sort_index().to_string())

    # Méta-labels : 1 si gagnant, 0 sinon
    events_df["meta_label"] = (events_df["bin"] == 1).astype(int)
    base_rate = events_df["meta_label"].mean()
    print(f"  Base rate (P(trade gagnant)) : {base_rate*100:.1f}%")

    # ------------------------------------------------------------------------
    print("\n[4/6] Construction des features...")
    feature_panels = []
    for ticker in tqdm(events_df["ticker"].unique(), desc="  Features"):
        feats = compute_features_one_asset(prices[ticker])
        if feats.empty:
            continue
        feats["ticker"] = ticker
        feature_panels.append(feats.reset_index().rename(columns={"index": "date"}))

    features_df = pd.concat(feature_panels, ignore_index=True)
    print(f"  Panel de features : {features_df.shape}")

    # Joindre features aux events
    events_df["t0"] = pd.to_datetime(events_df["t0"])
    features_df["date"] = pd.to_datetime(features_df["date"])
    merged = events_df.merge(
        features_df,
        left_on=["ticker", "t0"],
        right_on=["ticker", "date"],
        how="inner",
    )
    feature_cols = [c for c in features_df.columns if c not in ["date", "ticker"]]
    merged = merged.dropna(subset=feature_cols)
    print(f"  Events avec features : {len(merged)}")

    # ------------------------------------------------------------------------
    print("\n[5/6] Train/Test split + entraînement ML...")
    merged = merged.sort_values("t0").reset_index(drop=True)
    cut_idx = int(len(merged) * TRAIN_FRAC)
    cut_date = merged.iloc[cut_idx]["t0"]
    print(f"  Coupure train/test : {cut_date.date()}")

    # Purge : on retire du train les events qui se terminent APRÈS le début du test
    merged["t1"] = pd.to_datetime(merged["t1"])
    train_mask = (merged["t0"] < cut_date) & (merged["t1"] < cut_date)
    test_mask = merged["t0"] >= cut_date

    X_train = merged.loc[train_mask, feature_cols]
    y_train = merged.loc[train_mask, "meta_label"]
    X_test = merged.loc[test_mask, feature_cols]
    y_test = merged.loc[test_mask, "meta_label"]

    print(f"  Train : {len(X_train)} samples | Test : {len(X_test)} samples")
    print(f"  Base rate train : {y_train.mean()*100:.1f}% | test : {y_test.mean()*100:.1f}%")

    print("\n  Entraînement Random Forest...")
    model = train_meta_model(X_train, y_train, n_estimators=200, max_depth=8)

    print("\n  Évaluation OOS du méta-modèle :")
    metrics = evaluate_meta_model(model, X_test, y_test, threshold=PROBA_THRESHOLD)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<20} : {v:.3f}")
        else:
            print(f"    {k:<20} : {v}")

    # Feature importance
    importance = feature_importance_mdi(model, feature_cols)
    print("\n  Top 10 features les plus importantes :")
    print(importance.head(10).to_string())

    # ------------------------------------------------------------------------
    print("\n[6/6] Backtest du méta-modèle vs primaire seul...")

    # Probas prédites sur le test set
    test_subset = merged.loc[test_mask].copy()
    test_subset["proba"] = model.predict_proba(X_test)[:, 1]
    test_subset["take_trade"] = (test_subset["proba"] > PROBA_THRESHOLD).astype(int)

    # === STRATÉGIE PRIMAIRE SEULE (sans filtre ML) ===
    primary_returns = test_subset.groupby("t0")["ret"].mean()  # tous les trades pris
    # === STRATÉGIE META-LABELÉE (filtrée par ML) ===
    meta_subset = test_subset[test_subset["take_trade"] == 1]
    if len(meta_subset) > 0:
        meta_returns = meta_subset.groupby("t0")["ret"].mean()
    else:
        meta_returns = pd.Series(dtype=float)

    # Stats
    def stats(ret_series, name):
        if len(ret_series) < 10:
            print(f"  {name}: pas assez de trades")
            return
        # Annualiser : moyenne des trades × nb trades par an
        n_per_year = 252 / TIMEOUT_DAYS
        avg = ret_series.mean()
        std = ret_series.std()
        sharpe_per_trade = avg / std if std > 0 else np.nan
        annualized_sharpe = sharpe_per_trade * np.sqrt(n_per_year)
        win_rate = (ret_series > 0).mean()
        print(f"\n  {name}:")
        print(f"    Trades              : {len(ret_series)}")
        print(f"    Avg return / trade  : {avg*100:>6.3f} %")
        print(f"    Win rate            : {win_rate*100:>6.1f} %")
        print(f"    Sharpe / trade      : {sharpe_per_trade:>6.2f}")
        print(f"    Sharpe annualisé    : {annualized_sharpe:>6.2f}")

    stats(primary_returns, "PRIMAIRE seul (momentum top quintile)")
    stats(meta_returns, "META-LABELED (filtré par RF)")

    # ------------------------------------------------------------------------
    # Reconstruction d'une équité quotidienne approchée pour visualiser
    print("\n  Reconstruction d'equity curves...")

    def reconstruct_equity(events_subset, prices, label):
        """
        Pour chaque trade, on tient la position de t0 à t1.
        On simule un portefeuille équipondéré à 20 positions max.
        """
        eq = pd.Series(1.0, index=prices.index)
        # Calcul simplifié : daily return = moyenne des returns des trades actifs ce jour
        active_returns = pd.DataFrame(0.0, index=prices.index, columns=range(len(events_subset)))
        for i, (_, row) in enumerate(events_subset.iterrows()):
            t0, t1, tk = row["t0"], row["t1"], row["ticker"]
            if tk not in prices.columns:
                continue
            path = prices[tk].loc[t0:t1].pct_change().fillna(0)
            active_returns.loc[path.index, i] = path.values

        # Daily return = moyenne des positions actives
        n_active = (active_returns != 0).sum(axis=1)
        daily_ret = active_returns.sum(axis=1) / n_active.replace(0, np.nan)
        daily_ret = daily_ret.fillna(0)
        # Coûts approximatifs
        daily_ret -= COST_BPS / 10000.0 / TIMEOUT_DAYS  # amortis sur la durée

        equity = (1 + daily_ret).cumprod()
        return equity, daily_ret

    eq_primary, ret_primary = reconstruct_equity(test_subset, prices, "PRIMARY")
    if len(meta_subset) > 0:
        eq_meta, ret_meta = reconstruct_equity(meta_subset, prices, "META")
    else:
        eq_meta, ret_meta = None, None

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    ax = axes[0]
    test_idx = (eq_primary.index >= cut_date)
    ax.plot(eq_primary.index[test_idx], eq_primary[test_idx],
            label=f"PRIMAIRE seul ({len(test_subset)} trades)",
            color="steelblue", linewidth=1.5)
    if eq_meta is not None:
        ax.plot(eq_meta.index[test_idx], eq_meta[test_idx],
                label=f"META-LABELED RF ({len(meta_subset)} trades)",
                color="black", linewidth=2)
    ax.axvline(cut_date, color="red", linestyle="--", alpha=0.5, label="Train/Test")
    ax.set_yscale("log")
    ax.set_title("V4 — Triple Barrier + Meta-Labeling (test set OOS)", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2 : feature importance
    ax = axes[1]
    importance.head(15).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Feature Importances (MDI)", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "v4_results.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  Graphique : {fig_path}")

    # Exports
    importance.to_csv(OUTPUT_DIR / "feature_importance_v4.csv")
    test_subset.to_csv(OUTPUT_DIR / "test_predictions_v4.csv", index=False)

    print("\n" + "=" * 78)
    print("TERMINÉ V4")
    print("=" * 78)

    return model, importance, test_subset


if __name__ == "__main__":
    model, importance, test_subset = main()
