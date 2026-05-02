"""
Module de chargement des données.
- Télécharge les prix ajustés via yfinance
- Cache en parquet pour éviter les re-téléchargements
- Gère les tickers défaillants proprement
"""
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def download_prices(tickers, start="2010-01-01", end=None, force_refresh=False):
    """
    Télécharge les prix Close ajustés pour la liste de tickers.
    Renvoie un DataFrame indexé par date avec une colonne par ticker.
    """
    cache_file = CACHE_DIR / f"prices_{start}_{end or 'latest'}.parquet"

    if cache_file.exists() and not force_refresh:
        print(f"Chargement depuis le cache : {cache_file}")
        df = pd.read_parquet(cache_file)
        # Vérifier qu'on a bien tous les tickers demandés
        missing = set(tickers) - set(df.columns)
        if not missing:
            return df
        print(f"{len(missing)} tickers manquants dans le cache, refresh partiel...")

    print(f"Téléchargement de {len(tickers)} tickers depuis Yahoo Finance...")

    all_data = {}
    failed = []

    # Téléchargement par batchs de 50 pour respecter Yahoo
    batch_size = 50
    for i in tqdm(range(0, len(tickers), batch_size), desc="Batchs"):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            for tk in batch:
                try:
                    if len(batch) == 1:
                        series = data["Close"]
                    else:
                        series = data[tk]["Close"]
                    if series.notna().sum() > 100:  # Au moins 100 jours de data
                        all_data[tk] = series
                    else:
                        failed.append(tk)
                except (KeyError, AttributeError):
                    failed.append(tk)
        except Exception as e:
            print(f"Erreur batch {i}: {e}")
            failed.extend(batch)

        time.sleep(0.5)  # Politesse envers Yahoo

    if not all_data:
        raise RuntimeError("Aucune donnée téléchargée. Vérifie ta connexion.")

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"Succès : {len(all_data)} | Échecs : {len(failed)}")
    if failed:
        print(f"Tickers en échec (10 premiers) : {failed[:10]}")

    df.to_parquet(cache_file)
    print(f"Cache sauvegardé : {cache_file}")

    return df


def compute_returns(prices, method="simple"):
    """Renvoie les rendements quotidiens."""
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def filter_universe(prices, min_history_days=750, min_price=1.0):
    """
    Filtre les actions avec historique suffisant et prix raisonnable.
    Évite les penny stocks et les nouvelles introductions.
    """
    # Au moins min_history_days observations non-NaN
    valid_history = prices.notna().sum() >= min_history_days
    # Prix moyen au-dessus de min_price
    valid_price = prices.mean() >= min_price
    keep = valid_history & valid_price
    print(f"Filtrage univers : {keep.sum()}/{len(keep)} actions retenues")
    return prices.loc[:, keep]


if __name__ == "__main__":
    from universe import get_universe
    tickers = get_universe()
    prices = download_prices(tickers, start="2014-01-01")
    prices = filter_universe(prices)
    print(prices.tail())
    print(f"Forme finale : {prices.shape}")
