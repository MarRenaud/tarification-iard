"""
src/data.py
-----------
Chargement et préparation du dataset freMTPL2.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data() -> pd.DataFrame:
    """Charge freMTPL2_clean.csv depuis data/."""
    path = DATA_DIR / "freMTPL2_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    df = pd.read_csv(path)
    return df


def prepare_features_poisson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les features pour le GLM Poisson (Marion).
    Variables : tranche_age, Region_v3, Exposure (offset)
    """
    df = df.copy()

    # Tranche d'âge
    if "tranche_age" not in df.columns:
        bins   = [17, 25, 35, 45, 55, 65, 120]
        labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        df["tranche_age"] = pd.cut(df["DrivAge"], bins=bins, labels=labels)

    # Region_v3 = Region si déjà présent
    if "Region_v3" not in df.columns:
        df["Region_v3"] = df["Region"]

    return df


def prepare_features_gamma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les features pour le GLM Gamma (Mario).
    Variables : BonusMalus, VehAge, DrivAge, Region
    Filtre sur les polices avec sinistre > 0.
    """
    df = df.copy()
    df = df[df["ClaimAmount"] > 0].copy()

    # Clipping selon les bornes du modèle sauvegardé
    df["BonusMalus"] = df["BonusMalus"].clip(50, 196)
    df["VehAge"]     = df["VehAge"].clip(0, 84)
    df["DrivAge"]    = df["DrivAge"].clip(18, 99)

    return df


def get_severite_fallback(df: pd.DataFrame) -> float:
    """Sévérité moyenne calculée sur les données — fallback si le modèle Gamma échoue."""
    sinistres = df[df["ClaimAmount"] > 0]
    if len(sinistres) == 0:
        return 1800.0
    return sinistres["ClaimAmount"].mean() / sinistres["ClaimNb"].replace(0, 1).mean()
