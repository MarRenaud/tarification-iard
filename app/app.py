import streamlit as st
import pandas as pd
import numpy as np
import sys
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Modules projet ───────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
from src.models import load_poisson, load_gamma, predict_prime_pure
from llm_explainer import render_llm_explainer

# ── Configuration page ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tarification IARD",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — minimal, sans forcer les couleurs de la sidebar ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
}

.prime-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.prime-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #f0c040 !important;
    line-height: 1;
}
.prime-label {
    font-size: 0.85rem;
    color: #aab4c8 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1a1a2e !important;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 3px solid #f0c040;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ── Chargement des modèles ───────────────────────────────────────────────────
@st.cache_resource
def get_models():
    try:
        return load_poisson(), load_gamma()
    except Exception as e:
        st.error(f"Erreur chargement modèles : {e}")
        return None, None


@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "data" / "freMTPL2_clean.csv"
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Erreur chargement données : {e}")
        return None


poisson_model, gamma_model = get_models()
df = load_data()

SEVERITE_MOYENNE = 1800.0
if df is not None:
    sinistres = df[df["ClaimAmount"] > 0]
    if len(sinistres) > 0:
        SEVERITE_MOYENNE = sinistres["ClaimAmount"].mean() / sinistres["ClaimNb"].mean()

REGIONS = {
    "R11": {"nom": "Île-de-France",              "lat": 48.85, "lon": 2.35},
    "R21": {"nom": "Champagne-Ardenne",           "lat": 49.04, "lon": 4.36},
    "R22": {"nom": "Picardie",                    "lat": 49.89, "lon": 2.30},
    "R23": {"nom": "Haute-Normandie",             "lat": 49.44, "lon": 1.10},
    "R24": {"nom": "Centre",                      "lat": 47.75, "lon": 1.67},
    "R25": {"nom": "Basse-Normandie",             "lat": 48.99, "lon": -0.12},
    "R26": {"nom": "Bourgogne",                   "lat": 47.05, "lon": 4.38},
    "R31": {"nom": "Nord-Pas-de-Calais",          "lat": 50.48, "lon": 2.79},
    "R41": {"nom": "Lorraine",                    "lat": 48.69, "lon": 6.18},
    "R42": {"nom": "Alsace",                      "lat": 48.35, "lon": 7.44},
    "R43": {"nom": "Franche-Comté",               "lat": 47.28, "lon": 6.02},
    "R52": {"nom": "Pays de la Loire",            "lat": 47.76, "lon": -0.33},
    "R53": {"nom": "Bretagne",                    "lat": 48.11, "lon": -1.68},
    "R54": {"nom": "Poitou-Charentes",            "lat": 46.58, "lon": 0.34},
    "R72": {"nom": "Aquitaine",                   "lat": 44.84, "lon": -0.58},
    "R73": {"nom": "Midi-Pyrénées",               "lat": 43.60, "lon": 1.44},
    "R74": {"nom": "Limousin",                    "lat": 45.83, "lon": 1.26},
    "R82": {"nom": "Rhône-Alpes",                 "lat": 45.75, "lon": 4.85},
    "R83": {"nom": "Auvergne",                    "lat": 45.78, "lon": 3.08},
    "R91": {"nom": "Languedoc-Roussillon",        "lat": 43.61, "lon": 3.88},
    "R93": {"nom": "Provence-Alpes-Côte d'Azur",  "lat": 43.93, "lon": 6.07},
    "R94": {"nom": "Corse",                       "lat": 42.04, "lon": 9.01},
}

TRANCHES_AGE = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
VEH_GAS      = ["Regular", "Diesel"]
AREAS        = ["A", "B", "C", "D", "E", "F"]


def predict_prime(region, area, veh_power, veh_age, driv_age,
                  bonus_malus, veh_brand, veh_gas, density, tranche_age,
                  exposure=1.0):
    if poisson_model is None or gamma_model is None:
        freq = 0.053 * (bonus_malus / 100) * (1.2 if tranche_age == "18-25" else 1.0)
        return freq, SEVERITE_MOYENNE, freq * SEVERITE_MOYENNE
    return predict_prime_pure(
        poisson_model, gamma_model,
        tranche_age=tranche_age, region=region,
        bonus_malus=bonus_malus, veh_age=veh_age,
        driv_age=driv_age, veh_power=veh_power,
        density=density, exposure=exposure,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Profil client")
    st.markdown("---")

    st.markdown("**Conducteur**")
    driv_age    = st.slider("Âge du conducteur", 18, 85, 35)
    # Tranche d'âge déduite automatiquement de l'âge
    if driv_age <= 25:   tranche_age = "18-25"
    elif driv_age <= 35: tranche_age = "26-35"
    elif driv_age <= 45: tranche_age = "36-45"
    elif driv_age <= 55: tranche_age = "46-55"
    elif driv_age <= 65: tranche_age = "56-65"
    else:                tranche_age = "65+"
    bonus_malus = st.slider("Bonus-Malus", 50, 350, 100)

    st.markdown("---")
    st.markdown("**Véhicule**")
    veh_power = st.slider("Puissance (CV)", 4, 15, 7)
    veh_age   = st.slider("Âge du véhicule (ans)", 0, 20, 5)
    veh_gas   = st.selectbox("Carburant", VEH_GAS)
    veh_brand = st.selectbox("Marque", ["B1","B2","B3","B4","B5","B6","B10","B11","B12","B13","B14"])

    st.markdown("---")
    st.markdown("**Localisation**")
    region_key = st.selectbox(
        "Région",
        list(REGIONS.keys()),
        format_func=lambda x: f"{x} — {REGIONS[x]['nom']}",
        index=list(REGIONS.keys()).index("R82"),
    )
    area    = st.selectbox("Zone (A=rural → F=urbain dense)", AREAS, index=3)
    density = st.number_input("Densité de population (hab/km²)", 10, 30000, 1000)


# ── Calcul prime ─────────────────────────────────────────────────────────────
freq, severite, prime = predict_prime(
    region=region_key, area=area, veh_power=veh_power, veh_age=veh_age,
    driv_age=driv_age, bonus_malus=bonus_malus, veh_brand=veh_brand,
    veh_gas=veh_gas, density=density, tranche_age=tranche_age,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# Moteur de Tarification IARD")
st.markdown("*Assurance automobile — Modèle GLM Poisson × Gamma · Dataset freMTPL2*")
st.markdown("---")

# ── Résultat principal ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"""
    <div class="prime-card">
        <div class="prime-label">Prime pure annuelle estimée</div>
        <div class="prime-value">{prime:.2f} €</div>
        <div class="prime-label" style="margin-top:1rem">
            {REGIONS[region_key]['nom']} · BM {bonus_malus} · {driv_age} ans
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Fréquence prédite", f"{freq:.4f}", help="Nombre de sinistres espéré par an")

with col3:
    st.metric("Sévérité moyenne", f"{severite:.0f} €", help="Coût moyen par sinistre")

st.markdown("")

# ── Waterfall ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Décomposition de la prime par facteur</div>', unsafe_allow_html=True)
st.markdown("")


def get_contributions(poisson_model, gamma_model, tranche_age, region_key,
                      bonus_malus, veh_age, driv_age, veh_power, area, severite):
    ref = dict(tranche_age="36-45", region="R82", bonus_malus=100,
               veh_age=5, driv_age=35, veh_power=7)

    def pred(tranche_age, region, bonus_malus, veh_age, driv_age, veh_power):
        try:
            _, _, p = predict_prime_pure(
                poisson_model, gamma_model,
                tranche_age=tranche_age, region=region,
                bonus_malus=bonus_malus, veh_age=veh_age,
                driv_age=driv_age, veh_power=veh_power)
            return p
        except Exception:
            return None

    p_base = pred(**ref) or (0.053 * severite)
    p_age  = pred(tranche_age, ref['region'],  ref['bonus_malus'], ref['veh_age'], ref['driv_age'], ref['veh_power']) or p_base
    p_reg  = pred(tranche_age, region_key,     ref['bonus_malus'], ref['veh_age'], ref['driv_age'], ref['veh_power']) or p_age
    p_bm   = pred(tranche_age, region_key,     bonus_malus,        ref['veh_age'], ref['driv_age'], ref['veh_power']) or p_reg
    p_da   = pred(tranche_age, region_key,     bonus_malus,        ref['veh_age'], driv_age,        ref['veh_power']) or p_bm
    p_vp   = pred(tranche_age, region_key,     bonus_malus,        ref['veh_age'], driv_age,        veh_power)        or p_da
    p_va   = pred(tranche_age, region_key,     bonus_malus,        veh_age,        driv_age,        veh_power)        or p_vp

    return {
        "Base":           p_base,
        "Tranche d'âge":  p_age - p_base,
        "Région":         p_reg - p_age,
        "Bonus-Malus":    p_bm  - p_reg,
        "Âge conducteur": p_da  - p_bm,
        "Puissance":      p_vp  - p_da,
        "Âge véhicule":   p_va  - p_vp,
    }


contributions = get_contributions(
    poisson_model, gamma_model, tranche_age, region_key,
    bonus_malus, veh_age, driv_age, veh_power, area, severite
)
labels = list(contributions.keys())
values = list(contributions.values())

fig_waterfall = go.Figure(go.Waterfall(
    name="Prime",
    orientation="v",
    measure=["absolute"] + ["relative"] * (len(labels) - 1),
    x=labels,
    y=values,
    connector={"line": {"color": "#cccccc"}},
    increasing={"marker": {"color": "#f0c040"}},
    decreasing={"marker": {"color": "#3498db"}},
    totals={"marker": {"color": "#1a1a2e"}},
))
fig_waterfall.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="DM Sans", color="#1a1a2e", size=12),
    height=400,
    margin=dict(l=40, r=20, t=30, b=80),
    yaxis_title="Contribution à la prime (€)",
    xaxis=dict(tickfont=dict(color="#1a1a2e", size=11), tickangle=-20),
    yaxis=dict(tickfont=dict(color="#1a1a2e", size=11), gridcolor="#eeeeee", zerolinecolor="#cccccc"),
    showlegend=False,
)
st.plotly_chart(fig_waterfall, use_container_width=True)

# ── Carte ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Carte des primes moyennes par région</div>', unsafe_allow_html=True)
st.markdown("")

if df is not None:
    primes_region = (
        df.groupby("Region")
        .apply(lambda g: (g["ClaimNb"].sum() / g["Exposure"].sum()) * SEVERITE_MOYENNE)
        .reset_index().rename(columns={0: "prime_moyenne"})
    )
else:
    primes_region = pd.DataFrame({
        "Region": list(REGIONS.keys()),
        "prime_moyenne": [80 + np.random.uniform(-20, 40) for _ in REGIONS],
    })

primes_region["lat"] = primes_region["Region"].map(lambda r: REGIONS.get(r, {}).get("lat", 46.5))
primes_region["lon"] = primes_region["Region"].map(lambda r: REGIONS.get(r, {}).get("lon", 2.3))
primes_region["nom"] = primes_region["Region"].map(lambda r: REGIONS.get(r, {}).get("nom", r))

fig_map = px.scatter_map(
    primes_region, lat="lat", lon="lon",
    size="prime_moyenne", color="prime_moyenne",
    hover_name="nom",
    hover_data={"prime_moyenne": ":.2f", "lat": False, "lon": False},
    color_continuous_scale=["#3498db", "#f0c040", "#e74c3c"],
    size_max=40, zoom=4.5,
    center={"lat": 46.5, "lon": 2.3},
    map_style="carto-positron",
    labels={"prime_moyenne": "Prime moyenne (€)"},
)
fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))

region_data = primes_region[primes_region["Region"] == region_key]
if not region_data.empty:
    fig_map.add_trace(go.Scattermap(
        lat=region_data["lat"], lon=region_data["lon"],
        mode="markers",
        marker=dict(size=20, color="#ff6b35"),
        name="Région sélectionnée",
        hovertext=f"📍 {REGIONS[region_key]['nom']}",
    ))

st.plotly_chart(fig_map, use_container_width=True)

# ── LLM Explainer ────────────────────────────────────────────────────────────
profil = {
    "DrivAge":    driv_age,
    "BonusMalus": bonus_malus,
    "VehPower":   veh_power,
    "VehAge":     veh_age,
    "VehGas":     veh_gas,
    "VehBrand":   veh_brand,
    "Region":     region_key,
    "RegionNom":  REGIONS[region_key]["nom"],
    "Area":       area,
    "Density":    density,
    "tranche_age": tranche_age,
}
render_llm_explainer(freq, severite, prime, profil)

# ── À propos ──────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 À propos du projet", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Projet MAM 4A — Tarification IARD**
*Université Claude Bernard Lyon 1*
Superviseur : M. Soler

---

**Équipe**
- Marion — Exploration des données & GLM Poisson
- Mario — GLM Gamma (sévérité)
- Laflame — Comparaison des modèles & Application

---

**Dataset**
freMTPL2 — 678 013 polices d'assurance automobile françaises (OpenML)
        """)
    with col_b:
        st.markdown(r"""
**Méthodologie**

$$\text{Prime Pure} = \hat{\lambda} \times \hat{\mu}$$

| Composante | Modèle | Distribution |
|------------|--------|--------------|
| Fréquence | GLM Poisson | offset log(Exposure) |
| Sévérité | GLM Gamma | lien log |

**Modèles comparés**
- GLM Poisson × Gamma — référence actuarielle ✅
- GLM Tweedie — alternative compacte
- XGBoost — benchmark ML (Gini = 0.3676)
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<small style='color:#888'>Projet actuariel — MAM 4A — Université Claude Bernard Lyon 1 · "
    "GLM Poisson × Gamma · Dataset freMTPL2 (678 013 polices)</small>",
    unsafe_allow_html=True,
)
