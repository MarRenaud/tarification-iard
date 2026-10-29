"""
src/metrics.py
--------------
Métriques de validation pour les modèles de tarification IARD.
- Coefficient de Gini
- Courbe de Lorenz
- Lift curve
- Résidus de Pearson
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# COEFFICIENT DE GINI
# ─────────────────────────────────────────────

def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray,
                     exposure: np.ndarray = None) -> float:
    """
    Calcule le coefficient de Gini normalisé (actuariel).

    G = 2 * AUC(Lorenz) - 1, normalisé par le Gini du modèle parfait.

    Paramètres
    ----------
    y_true    : sinistres réels (ClaimNb ou montant)
    y_pred    : prédictions du modèle
    exposure  : poids (exposition) — optionnel

    Retourne
    --------
    float ∈ [-1, 1] — plus proche de 1 = meilleur pouvoir discriminant
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if exposure is None:
        exposure = np.ones(len(y_true))
    exposure = np.asarray(exposure, dtype=float)

    # Tri par prédiction croissante
    order  = np.argsort(y_pred)
    y_true = y_true[order]
    y_pred = y_pred[order]
    w      = exposure[order]

    # Cumuls pondérés
    cum_exposure = np.cumsum(w) / w.sum()
    cum_claims   = np.cumsum(y_true * w) / (y_true * w).sum()

    # AUC par trapèzes
    auc_lorenz = np.trapz(cum_claims, cum_exposure)
    gini_raw   = 2 * auc_lorenz - 1

    # Normalisation par le modèle parfait
    cum_claims_perf = np.cumsum(np.sort(y_true) * w) / (y_true * w).sum()
    auc_perfect     = np.trapz(cum_claims_perf, cum_exposure)
    gini_perfect    = 2 * auc_perfect - 1

    if gini_perfect == 0:
        return 0.0
    return float(gini_raw / gini_perfect)


# ─────────────────────────────────────────────
# COURBE DE LORENZ (plotly)
# ─────────────────────────────────────────────

def plot_lorenz(y_true: np.ndarray, y_pred: np.ndarray,
                exposure: np.ndarray = None,
                model_name: str = "Modèle") -> go.Figure:
    """
    Trace la courbe de Lorenz du modèle vs la référence (diagonale).

    Retourne un objet plotly Figure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if exposure is None:
        exposure = np.ones(len(y_true))
    exposure = np.asarray(exposure, dtype=float)

    order        = np.argsort(y_pred)
    y_true_s     = y_true[order]
    w            = exposure[order]

    cum_exp    = np.concatenate([[0], np.cumsum(w) / w.sum()])
    cum_claims = np.concatenate([[0], np.cumsum(y_true_s * w) / (y_true_s * w).sum()])

    gini = gini_coefficient(y_true, y_pred, exposure)

    fig = go.Figure()

    # Diagonale (modèle nul)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#cccccc", dash="dash", width=1.5),
        name="Modèle nul",
    ))

    # Courbe du modèle
    fig.add_trace(go.Scatter(
        x=cum_exp, y=cum_claims,
        mode="lines",
        line=dict(color="#1a6faf", width=2.5),
        fill="tonexty",
        fillcolor="rgba(26,111,175,0.08)",
        name=f"{model_name} (Gini = {gini:.3f})",
    ))

    fig.update_layout(
        title=f"Courbe de Lorenz — {model_name}",
        xaxis_title="Fraction de polices (triées par prime croissante)",
        yaxis_title="Fraction des sinistres cumulés",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="DM Sans",
        height=420,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", range=[0, 1])
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", range=[0, 1])

    return fig


# ─────────────────────────────────────────────
# LIFT CURVE (plotly)
# ─────────────────────────────────────────────

def plot_lift(y_true: np.ndarray, y_pred: np.ndarray,
              exposure: np.ndarray = None,
              n_buckets: int = 10,
              model_name: str = "Modèle") -> go.Figure:
    """
    Trace la lift curve : fréquence observée vs prédite par décile de risque.

    Retourne un objet plotly Figure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if exposure is None:
        exposure = np.ones(len(y_true))
    exposure = np.asarray(exposure, dtype=float)

    df = pd.DataFrame({
        "y_true":   y_true,
        "y_pred":   y_pred,
        "exposure": exposure,
    })

    df["bucket"] = pd.qcut(df["y_pred"], q=n_buckets, labels=False, duplicates="drop")

    agg = df.groupby("bucket").apply(lambda g: pd.Series({
        "freq_obs":  g["y_true"].sum() / g["exposure"].sum(),
        "freq_pred": (g["y_pred"] * g["exposure"]).sum() / g["exposure"].sum(),
        "n":         len(g),
    })).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agg["bucket"] + 1,
        y=agg["freq_obs"],
        mode="markers+lines",
        name="Fréquence observée",
        line=dict(color="#d62728", width=2),
        marker=dict(size=8),
    ))

    fig.add_trace(go.Scatter(
        x=agg["bucket"] + 1,
        y=agg["freq_pred"],
        mode="markers+lines",
        name="Fréquence prédite",
        line=dict(color="#1a6faf", width=2, dash="dot"),
        marker=dict(size=8, symbol="diamond"),
    ))

    fig.update_layout(
        title=f"Lift Curve — {model_name}",
        xaxis_title=f"Décile de risque (1 = faible risque, {n_buckets} = fort risque)",
        yaxis_title="Fréquence de sinistres",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="DM Sans",
        height=400,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig


# ─────────────────────────────────────────────
# COMPARAISON MULTI-MODÈLES (plotly)
# ─────────────────────────────────────────────

def plot_gini_comparison(scores: dict) -> go.Figure:
    """
    Bar chart de comparaison des coefficients de Gini entre modèles.

    Paramètres
    ----------
    scores : dict — ex. {"GLM Poisson×Gamma": 0.32, "Tweedie": 0.29, "XGBoost": 0.41}

    Retourne un objet plotly Figure.
    """
    models = list(scores.keys())
    ginis  = list(scores.values())
    colors = ["#1a6faf", "#e07b39", "#2ca02c", "#d62728"][:len(models)]

    fig = go.Figure(go.Bar(
        x=models,
        y=ginis,
        marker_color=colors,
        text=[f"{g:.3f}" for g in ginis],
        textposition="outside",
    ))

    fig.update_layout(
        title="Comparaison des coefficients de Gini",
        yaxis_title="Coefficient de Gini normalisé",
        yaxis_range=[0, max(ginis) * 1.25],
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="DM Sans",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig


# ─────────────────────────────────────────────
# RÉSIDUS DE PEARSON (plotly)
# ─────────────────────────────────────────────

def plot_pearson_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Modèle") -> go.Figure:
    """
    Scatter plot des résidus de Pearson vs valeurs prédites.

    Résidu de Pearson : r_i = (y_i - mu_i) / sqrt(mu_i)
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    residus = (y_true - y_pred) / np.sqrt(np.maximum(y_pred, 1e-8))

    # Échantillon pour la lisibilité
    n     = min(5000, len(residus))
    idx   = np.random.choice(len(residus), n, replace=False)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_pred[idx],
        y=residus[idx],
        mode="markers",
        marker=dict(color="#1a6faf", size=4, opacity=0.4),
        name="Résidus de Pearson",
    ))

    # Ligne y=0
    fig.add_hline(y=0, line=dict(color="#d62728", width=1.5, dash="dash"))

    fig.update_layout(
        title=f"Résidus de Pearson — {model_name}",
        xaxis_title="Valeurs prédites",
        yaxis_title="Résidu de Pearson",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="DM Sans",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig
