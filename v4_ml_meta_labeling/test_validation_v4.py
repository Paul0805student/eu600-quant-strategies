"""
Validation V4 avec données synthétiques.
"""
import sys
sys.path.insert(0, '/home/claude/eu600_quant_v4')

import numpy as np
import pandas as pd

from labels import get_events, get_daily_volatility
from features import compute_features_one_asset
from meta_labeling import train_meta_model, evaluate_meta_model, feature_importance_mdi


def gen_market(n_assets=80, n_years=8, seed=42):
    """Marché avec momentum exploitable + bruit."""
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    dates = pd.date_range("2016-01-04", periods=n_days, freq="B")
    tickers = [f"S{i:03d}" for i in range(n_assets)]

    mkt = rng.normal(0.07/252, 0.16/np.sqrt(252), n_days)
    betas = rng.normal(1.0, 0.25, n_assets).clip(0.5, 1.5)
    idio_vol = rng.uniform(0.012, 0.022, n_assets)
    mom_q = rng.normal(0, 1, n_assets)
    alphas = mom_q * (0.05/252)

    R = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        R[:, i] = alphas[i] + betas[i] * mkt + rng.normal(0, idio_vol[i], n_days)

    # Momentum persistant
    for t in range(252, n_days):
        past = R[t-252:t-21, :].sum(axis=0)
        z = (past - past.mean()) / (past.std() + 1e-9)
        R[t, :] += z * 0.0004

    prices = pd.DataFrame(100 * np.exp(np.cumsum(R, axis=0)), index=dates, columns=tickers)
    return prices


def main():
    print("=" * 78)
    print("VALIDATION V4 — Triple Barrier + Meta-Labeling")
    print("=" * 78)

    prices = gen_market(n_assets=60, n_years=7, seed=42)
    print(f"\n  {prices.shape[1]} actifs · {prices.shape[0]} jours")

    # Signaux primaires : top quintile momentum mensuels
    print("\n[1] Génération des events primaires (momentum 12-1)...")
    log_p = np.log(prices)
    mom = log_p.shift(21) - log_p.shift(252)
    s = pd.Series(prices.index, index=prices.index)
    rebal = s.groupby([s.index.year, s.index.month]).max().values

    signal_dates_per_ticker = {tk: [] for tk in prices.columns}
    for date in rebal:
        if date not in mom.index:
            continue
        row = mom.loc[date].dropna()
        if len(row) < 20:
            continue
        n_top = max(1, int(len(row) * 0.20))
        for tk in row.nlargest(n_top).index:
            signal_dates_per_ticker[tk].append(date)
    n_total = sum(len(d) for d in signal_dates_per_ticker.values())
    print(f"  Total events long : {n_total}")

    # Triple barrier
    print("\n[2] Triple Barrier...")
    all_events = []
    for tk, dates in signal_dates_per_ticker.items():
        if len(dates) == 0:
            continue
        events = get_events(prices[tk], pd.DatetimeIndex(dates),
                            side="long", pt_mult=2.0, sl_mult=1.5,
                            num_days=21, vol_span=60)
        if events.empty:
            continue
        events["ticker"] = tk
        all_events.append(events.reset_index().rename(columns={"index": "t0"}))

    events_df = pd.concat(all_events, ignore_index=True)
    events_df["meta_label"] = (events_df["bin"] == 1).astype(int)
    print(f"  Events labellisés : {len(events_df)}")
    print(f"  Distribution bin :\n{events_df['bin'].value_counts().sort_index().to_string()}")
    print(f"  Base rate (gagnants) : {events_df['meta_label'].mean()*100:.1f}%")

    # Features
    print("\n[3] Construction features...")
    feats_panels = []
    for tk in events_df["ticker"].unique():
        f = compute_features_one_asset(prices[tk])
        if f.empty:
            continue
        f["ticker"] = tk
        feats_panels.append(f.reset_index().rename(columns={"index": "date"}))
    features_df = pd.concat(feats_panels, ignore_index=True)
    feature_cols = [c for c in features_df.columns if c not in ["date", "ticker"]]

    events_df["t0"] = pd.to_datetime(events_df["t0"])
    features_df["date"] = pd.to_datetime(features_df["date"])
    merged = events_df.merge(features_df, left_on=["ticker", "t0"],
                             right_on=["ticker", "date"], how="inner")
    merged = merged.dropna(subset=feature_cols)
    print(f"  Events avec features : {len(merged)}")

    # Train/test split avec purge
    print("\n[4] Train/Test + entraînement RF...")
    merged = merged.sort_values("t0").reset_index(drop=True)
    cut_idx = int(len(merged) * 0.6)
    cut_date = merged.iloc[cut_idx]["t0"]

    merged["t1"] = pd.to_datetime(merged["t1"])
    train_mask = (merged["t0"] < cut_date) & (merged["t1"] < cut_date)
    test_mask = merged["t0"] >= cut_date

    X_train, y_train = merged.loc[train_mask, feature_cols], merged.loc[train_mask, "meta_label"]
    X_test, y_test = merged.loc[test_mask, feature_cols], merged.loc[test_mask, "meta_label"]
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    model = train_meta_model(X_train, y_train, n_estimators=200, max_depth=8)
    metrics = evaluate_meta_model(model, X_test, y_test, threshold=0.55)
    print("\n  Metrics OOS :")
    for k, v in metrics.items():
        print(f"    {k:<22} : {v}")

    # Feature importance
    importance = feature_importance_mdi(model, feature_cols)
    print("\n  Top 10 features :")
    print(importance.head(10).to_string())

    # Comparaison primaire vs meta-filtré
    print("\n[5] Backtest primaire vs meta-labeled...")
    test_subset = merged.loc[test_mask].copy()
    test_subset["proba"] = model.predict_proba(X_test)[:, 1]

    primary_mean_ret = test_subset["ret"].mean()
    primary_winrate = (test_subset["ret"] > 0).mean()

    meta_mask = test_subset["proba"] > 0.55
    if meta_mask.sum() > 0:
        meta_mean_ret = test_subset.loc[meta_mask, "ret"].mean()
        meta_winrate = (test_subset.loc[meta_mask, "ret"] > 0).mean()
    else:
        meta_mean_ret, meta_winrate = np.nan, np.nan

    print(f"\n  PRIMAIRE seul : {len(test_subset)} trades, "
          f"avg ret = {primary_mean_ret*100:.3f}%, winrate = {primary_winrate*100:.1f}%")
    print(f"  META-LABELED  : {meta_mask.sum()} trades, "
          f"avg ret = {meta_mean_ret*100:.3f}%, winrate = {meta_winrate*100:.1f}%")

    if not np.isnan(meta_mean_ret):
        improvement = (meta_mean_ret - primary_mean_ret) * 100
        print(f"\n  Amélioration ML : +{improvement:.3f}% par trade")
        if improvement > 0:
            print("  → ML filtre EFFICACEMENT les mauvais trades ✓")
        else:
            print("  → ML pas efficace sur cet échantillon")

    print("\n=== VALIDATION V4 OK ===")


if __name__ == "__main__":
    main()
