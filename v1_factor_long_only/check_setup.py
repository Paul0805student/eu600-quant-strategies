"""
Quick check : vérifie que tout est prêt avant de lancer main.py
"""
import sys

def check():
    print("=== Vérification de l'environnement ===\n")

    # Python version
    print(f"Python : {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  /!\\ Python 3.9+ recommandé")

    # Packages
    pkgs = {
        "yfinance": "0.2.40",
        "pandas": "2.0",
        "numpy": "1.24",
        "matplotlib": "3.7",
        "scipy": "1.10",
        "tqdm": "4.65",
    }
    missing = []
    for pkg, min_v in pkgs.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg:<15} {ver}")
        except ImportError:
            missing.append(pkg)
            print(f"  {pkg:<15} MANQUANT")

    if missing:
        print(f"\n/!\\ Installer : pip install {' '.join(missing)}")
        return False

    # Test connexion Yahoo
    print("\nTest connexion Yahoo Finance...")
    try:
        import yfinance as yf
        data = yf.download("AAPL", period="5d", progress=False, auto_adjust=True)
        if not data.empty:
            print(f"  OK : {len(data)} jours téléchargés pour AAPL")
        else:
            print("  /!\\ Pas de données reçues (firewall ?)")
            return False
    except Exception as e:
        print(f"  /!\\ Erreur : {e}")
        return False

    # Test imports locaux
    print("\nTest imports locaux...")
    try:
        from universe import get_universe
        from data import download_prices
        from strategies import signal_momentum
        from backtest import compute_metrics
        print(f"  OK : universe contient {len(get_universe())} tickers")
    except ImportError as e:
        print(f"  /!\\ Erreur : {e}")
        return False

    print("\n=== Tout est prêt — lance : python main.py ===")
    return True


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
