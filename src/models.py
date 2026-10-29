import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

REGIONS_NON_SIG = {"R21", "R22", "R23", "R24", "R25", "R26", "R42", "R43"}

def load_poisson():
    return joblib.load(MODELS_DIR / "glm_poisson.pkl")

def load_gamma():
    return joblib.load(MODELS_DIR / "glm_gamma_v4.pkl")

def load_gamma_infos() -> dict:
    path = MODELS_DIR / "infos_modele_gamma.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_prime_pure(poisson_model, gamma_model,
                       tranche_age, region, bonus_malus,
                       veh_age, driv_age, veh_power=7,
                       density=1000, exposure=1.0):

    # ── Fréquence ─────────────────────────────────────────────────────
    region_clean = str(region).strip().replace("'", "")
    region_v3    = "Autre" if region_clean in REGIONS_NON_SIG else region_clean
    bm           = float(np.clip(bonus_malus, 50, 350))
    log_bm       = np.log(bm)

    df_poisson = pd.DataFrame([{
        "log_bonus_malus": log_bm,
        "tranche_age":     str(tranche_age).strip(),
        "Region_v3":       region_v3,
        "VehAge":          float(veh_age),
        "VehPower":        float(veh_power),
        "Density":         float(density),
        "VehGas":          "Regular",
        "Exposure":        float(exposure),
    }])

    try:
        pred  = float(poisson_model.predict(df_poisson)[0])
        freq  = pred / float(exposure) if exposure > 0 else pred
        freq  = float(np.clip(freq, 0, 5))
    except Exception as e:
        print(f"[WARN] freq fallback : {e}")
        freq = 0.053

    # ── Sévérité ──────────────────────────────────────────────────────
    infos   = load_gamma_infos()
    bm_g    = float(np.clip(bonus_malus, infos.get("BonusMalus_min", 50),  infos.get("BonusMalus_max", 196)))
    va_g    = float(np.clip(veh_age,     infos.get("VehAge_min", 0),        infos.get("VehAge_max", 84)))
    da_g    = float(np.clip(driv_age,    infos.get("DrivAge_min", 18),       infos.get("DrivAge_max", 99)))
    vp_g    = float(np.clip(veh_power,   4, 15))
    regions = infos.get("regions", [])
    reg_g   = region_clean if (not regions or region_clean in regions) else "R82"

    df_gamma = pd.DataFrame([{
        "BonusMalus": bm_g,
        "VehAge":     va_g,
        "DrivAge":    da_g,
        "VehPower":   vp_g,
        "Region":     reg_g,
    }])

    try:
        sev = float(np.clip(gamma_model.predict(df_gamma)[0], 100, 50000))
    except Exception as e:
        print(f"[WARN] sev fallback : {e}")
        sev = 1800.0

    return freq, sev, freq * sev
