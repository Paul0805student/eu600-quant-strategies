"""
Triple Barrier Method (López de Prado, "Advances in Financial Machine Learning" ch.3).

Problème classique du ML appliqué à la finance :
- Si on labellise simplement "rendement positif à H jours" → labels biaisés
- On rate les setups où l'action a TP rapidement puis chute, ou stop-loss puis rebond
- Les labels fixes ignorent la volatilité du moment (un +2% en marché calme ≠ marché vol)

Solution Triple Barrier :
Pour chaque event (setup détecté à t0) :
- Barrière HAUTE : take-profit à pt × σ (ex: +2σ)
- Barrière BASSE : stop-loss à -sl × σ (ex: -2σ)
- Barrière VERTICALE : timeout après h jours
On regarde laquelle des 3 est touchée EN PREMIER, et le label est :
  +1 si TP touché, -1 si SL touché, 0 si timeout (ou signe du rendement)

Avantages :
- Volatility-aware : barrières adaptées au régime
- Path-dependent : capte la dynamique réelle du trade
- Réaliste : reflète comment un trader exit ses positions
"""
import numpy as np
import pandas as pd


def get_daily_volatility(prices, span=100):
    """
    Volatilité daily exponentiellement pondérée (ewm).
    On utilise les returns à 1j, et on lisse sur span jours.
    """
    returns = prices.pct_change()
    return returns.ewm(span=span, min_periods=20).std()


def get_vertical_barriers(events, prices, num_days=10):
    """
    Pour chaque event date, calculer la date de timeout en business days.
    Renvoie une Series indexée par event date, avec valeur = date de timeout.
    """
    out = {}
    price_index = prices.index
    for t0 in events.index:
        # Trouver l'index de t0 dans les prix
        if t0 not in price_index:
            continue
        i0 = price_index.get_loc(t0)
        i1 = min(i0 + num_days, len(price_index) - 1)
        out[t0] = price_index[i1]
    return pd.Series(out)


def apply_triple_barrier(prices_one_asset, events, pt_sl, vol, vertical_barriers,
                          min_ret=0):
    """
    Pour UN actif, applique le triple barrier method.

    Paramètres :
    - prices_one_asset : Series des prix de l'actif
    - events : Series indexée par les event dates (valeur = side, +1 long ou -1 short)
    - pt_sl : tuple (pt_mult, sl_mult) — multiplicateurs de vol
    - vol : Series de volatilité daily
    - vertical_barriers : Series des dates de timeout par event
    - min_ret : seuil minimal de rendement attendu pour garder l'event

    Renvoie un DataFrame avec colonnes :
    - t1 : date de sortie (premier des 3 barrières touchée)
    - ret : rendement réalisé sur ce trade
    - bin : label {-1, 0, +1}
    """
    out = pd.DataFrame(index=events.index, columns=["t1", "ret", "bin"])

    for t0 in events.index:
        if t0 not in vol.index or pd.isna(vol.loc[t0]):
            continue
        v = vol.loc[t0]
        if v == 0 or pd.isna(v):
            continue

        side = events.loc[t0]  # +1 ou -1
        # Date timeout
        if t0 not in vertical_barriers.index:
            continue
        t1_v = vertical_barriers.loc[t0]

        # Path des prix entre t0 et t1
        path = prices_one_asset.loc[t0:t1_v]
        if len(path) < 2:
            continue

        # Returns cumulés depuis t0 (signés selon side)
        cum_ret = (path / path.iloc[0] - 1) * side

        # Barrière haute (TP) et basse (SL)
        pt = pt_sl[0] * v if pt_sl[0] > 0 else np.inf
        sl = -pt_sl[1] * v if pt_sl[1] > 0 else -np.inf

        # Première date où TP est touché
        tp_hits = cum_ret[cum_ret >= pt]
        sl_hits = cum_ret[cum_ret <= sl]

        first_tp = tp_hits.index[0] if len(tp_hits) > 0 else pd.NaT
        first_sl = sl_hits.index[0] if len(sl_hits) > 0 else pd.NaT
        timeout = path.index[-1]

        # Quelle barrière est touchée en premier ?
        candidates = [(first_tp, +1), (first_sl, -1), (timeout, 0)]
        candidates = [(t, lbl) for t, lbl in candidates if pd.notna(t)]
        candidates.sort(key=lambda x: x[0])

        if not candidates:
            continue
        exit_time, label = candidates[0]
        exit_ret = (path.loc[exit_time] / path.iloc[0] - 1) * side

        # Si timeout, le label peut être le signe du return final (option)
        if label == 0 and abs(exit_ret) > min_ret:
            label = int(np.sign(exit_ret))

        out.loc[t0, "t1"] = exit_time
        out.loc[t0, "ret"] = exit_ret
        out.loc[t0, "bin"] = label

    return out.dropna(subset=["bin"])


def get_events(prices_one_asset, signal_dates, side="long",
               pt_mult=2.0, sl_mult=1.5, num_days=10, vol_span=100):
    """
    Wrapper : pour un actif, à partir des dates de signal, retourner le DataFrame
    de labels triple barrier.

    side : "long" (side=+1) ou "short" (side=-1).
    """
    side_val = +1 if side == "long" else -1
    # Filtrer les dates qui sont dans l'index des prix
    valid_dates = signal_dates.intersection(prices_one_asset.index)
    if len(valid_dates) == 0:
        return pd.DataFrame()
    events = pd.Series(side_val, index=valid_dates)

    vol = get_daily_volatility(prices_one_asset, span=vol_span)
    vertical = get_vertical_barriers(events, prices_one_asset, num_days=num_days)

    return apply_triple_barrier(
        prices_one_asset, events, (pt_mult, sl_mult), vol, vertical,
    )
