"""
app/llm_explainer.py
--------------------
Module LLM : Explication de la prime pure — double mode Grand public / Actuaire
Utilise l'API Groq (gratuite) avec le modèle llama-3.3-70b-versatile
"""

import streamlit as st
import json
import os

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─────────────────────────────────────────────
# PROMPTS SYSTÈME
# ─────────────────────────────────────────────

SYSTEM_GRAND_PUBLIC = """Tu es un conseiller en assurance automobile, pédagogue et bienveillant.
Tu expliques à un assuré, en langage clair et sans jargon technique, pourquoi sa prime d'assurance
a été calculée ainsi. Tu utilises des analogies concrètes de la vie quotidienne.
Tu ne dépasses jamais 200 mots. Tu es rassurant et transparent.
Commence toujours par reformuler le montant de la prime de façon simple.
Réponds en français."""

SYSTEM_ACTUAIRE = """Tu es un actuaire senior spécialisé en tarification non-vie (IARD).
Tu expliques la décomposition technique de la prime pure à destination d'un actuaire junior
ou d'un auditeur technique. Tu dois impérativement :
- Rappeler la formule : Prime Pure = Fréquence × Coût Moyen
- Expliquer que la fréquence est modélisée par un GLM Poisson avec offset log(Exposure)
- Expliquer que la sévérité est modélisée par un GLM Gamma avec lien log
- Interpréter les facteurs de risque dominants du profil fourni (BonusMalus, âge, région)
- Mentionner au moins une limite du modèle
Tu es précis, rigoureux. Notation statistique standard. Maximum 350 mots.
Réponds en français."""

# ─────────────────────────────────────────────
# PROMPTS UTILISATEUR
# ─────────────────────────────────────────────

def _prompt_grand_public(freq, sev, prime, profil):
    return f"""Un assuré a reçu une prime d'assurance automobile de {prime:.2f} € par an.

Voici son profil :
- Âge du conducteur : {profil.get('DrivAge', 'N/A')} ans
- Bonus-Malus : {profil.get('BonusMalus', 'N/A')} (100 = neutre, >100 = malus)
- Puissance du véhicule : {profil.get('VehPower', 'N/A')} CV
- Âge du véhicule : {profil.get('VehAge', 'N/A')} ans
- Carburant : {profil.get('VehGas', 'N/A')}
- Région : {profil.get('RegionNom', profil.get('Region', 'N/A'))}
- Zone de résidence : {profil.get('Area', 'N/A')} (A=rural, F=urbain dense)

Explique-lui simplement pourquoi il paie ce montant, quels facteurs jouent en sa faveur
ou en sa défaveur, et ce que cela représente concrètement."""


def _prompt_actuaire(freq, sev, prime, profil):
    profil_str = json.dumps(profil, ensure_ascii=False, indent=2)
    return f"""Voici les résultats de tarification pour un assuré du portefeuille freMTPL2 :

Résultats du modèle :
- Fréquence estimée (GLM Poisson) : {freq:.5f} sinistres/an
- Coût moyen estimé (GLM Gamma)   : {sev:.2f} €/sinistre
- Prime pure = Fréquence x Coût moyen = {prime:.2f} €

Profil de l'assuré :
{profil_str}

Fournis une analyse actuarielle complète : modélisation utilisée, interprétation des facteurs
de risque dominants pour ce profil précis, et limites à signaler à l'auditeur."""


# ─────────────────────────────────────────────
# APPEL API GROQ (streaming)
# ─────────────────────────────────────────────

def _stream_groq(system_prompt, user_prompt, api_key):
    """Génère l'explication en streaming via l'API Groq."""
    client = Groq(api_key=api_key)

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.4,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ─────────────────────────────────────────────
# COMPOSANT STREAMLIT
# ─────────────────────────────────────────────

def render_llm_explainer(freq: float, sev: float, prime: float, profil: dict):
    """
    Affiche le bloc LLM dans la page Streamlit.

    Paramètres
    ----------
    freq   : float — fréquence prédite par le GLM Poisson (sinistres/an)
    sev    : float — coût moyen prédit par le GLM Gamma (€/sinistre)
    prime  : float — prime pure = freq × sev (€)
    profil : dict  — variables de l'assuré
    """
    st.markdown("---")
    st.markdown('<div class="section-title">Explication IA de la prime</div>',
                unsafe_allow_html=True)
    st.markdown("")

    # Récupération clé API
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        st.warning("⚠️ Clé API Groq non configurée — définissez GROQ_API_KEY dans votre environnement.")
        return

    if not GROQ_AVAILABLE:
        st.error("❌ Package `groq` non installé — lancez `pip install groq`")
        return

    # Sélecteur de mode
    mode_label = st.radio(
        "Mode d'explication",
        options=["👤 Grand public — explication simplifiée",
                 "📐 Actuaire — analyse technique"],
        horizontal=True,
    )
    mode = "grand_public" if "Grand" in mode_label else "actuaire"

    if st.button("✨ Générer l'explication", use_container_width=True):

        if mode == "grand_public":
            system = SYSTEM_GRAND_PUBLIC
            prompt = _prompt_grand_public(freq, sev, prime, profil)
            border = "#f0c040"
            bg     = "#1a1a2e08"
            label  = "💬 Explication pour l'assuré"
        else:
            system = SYSTEM_ACTUAIRE
            prompt = _prompt_actuaire(freq, sev, prime, profil)
            border = "#3498db"
            bg     = "#3498db08"
            label  = "📊 Analyse actuarielle"

        st.markdown(f"**{label}**")
        style = (
            f"background:{bg}; padding:1.4rem 1.6rem; border-radius:10px; "
            f"border-left:4px solid {border}; font-size:0.95rem; line-height:1.7;"
        )

        placeholder = st.empty()
        full_text = ""

        try:
            for chunk in _stream_groq(system, prompt, api_key):
                full_text += chunk
                placeholder.markdown(
                    f'<div style="{style}">{full_text}▌</div>',
                    unsafe_allow_html=True,
                )
            placeholder.markdown(
                f'<div style="{style}">{full_text}</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"❌ Erreur Groq : {e}")
