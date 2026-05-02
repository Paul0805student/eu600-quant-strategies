"""
Meta-Labeling (López de Prado ch.3.4).

Concept :
- On a un MODÈLE PRIMAIRE qui décide de la DIRECTION (long/short)
  → ici, momentum cross-section : long les top quintile
- On entraîne un MÉTA-MODÈLE qui décide si on PREND ou non le trade
  → c'est un classifier binaire {0, 1} où 1 = "trade gagnant à venir"

Avantages :
- Augmente la précision (réduit les faux positifs)
- Permet de sizer les positions par la probabilité prédite
- Sépare ALPHA (direction) et RISK (taille)
- Robuste : le primaire est interprétable, le ML ne fait que filtrer

Workflow :
1. Définir signal primaire → events (dates où on aurait pris une position long)
2. Triple barrier sur ces events → labels {-1, 0, +1}
3. Convertir en méta-label binaire : 1 si label > 0 (trade gagnant), 0 sinon
4. Entraîner ML (Random Forest / GBM) sur features → P(trade gagnant)
5. Au moment du trading : prendre le trade SI ML proba > seuil

Cross-validation : ATTENTION au leakage temporel. On utilise PURGED K-FOLD
(López ch.7) : on retire de train les samples dont la fenêtre de label
chevauche la fenêtre de test.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def make_meta_labels(triple_barrier_df):
    """
    Convertit les labels triple barrier en méta-labels binaires.
    1 = trade gagnant (label = +1), 0 = sinon (label 0 ou -1).
    """
    out = pd.DataFrame(index=triple_barrier_df.index)
    out["meta_label"] = (triple_barrier_df["bin"] == 1).astype(int)
    out["t1"] = triple_barrier_df["t1"]
    out["ret"] = triple_barrier_df["ret"]
    return out


def purged_train_test_split(events, test_start, embargo_pct=0.01):
    """
    Split train/test avec PURGE et EMBARGO (López ch.7.4).

    - Purge : retirer du train les events dont la fenêtre [t0, t1] chevauche test
    - Embargo : retirer du train les events qui finissent juste avant le début du test

    events : DataFrame indexé par t0, avec colonne 't1' (date de fin de label)
    test_start : date de début du test set
    """
    embargo_days = int(len(events) * embargo_pct)
    embargo_end = test_start + pd.Timedelta(days=embargo_days * 2)

    # Test set : t0 dans [test_start, fin]
    test_mask = events.index >= test_start

    # Train set : t0 < test_start ET t1 < test_start (label finalisé avant test)
    train_mask = (events.index < test_start) & (events["t1"] < test_start)

    return events.loc[train_mask], events.loc[test_mask]


def train_meta_model(X_train, y_train, n_estimators=200, max_depth=8,
                     min_samples_leaf=20, random_state=42):
    """
    Entraîne un Random Forest comme méta-modèle.
    Class weight balancé car les méta-labels sont souvent déséquilibrés.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_meta_model(model, X_test, y_test, threshold=0.5):
    """Évalue le méta-modèle. Renvoie un dict de métriques."""
    if len(X_test) == 0:
        return {}
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > threshold).astype(int)
    metrics = {
        "n_samples": len(y_test),
        "base_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "n_signals_filtered": int((pred == 1).sum()),
    }
    return metrics


def feature_importance_mdi(model, feature_names):
    """Mean Decrease Impurity feature importance."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
