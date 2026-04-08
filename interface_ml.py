import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Détection Cancer Pulmonaire", layout="centered")
st.title("Détection du Cancer Pulmonaire")

st.write("Entrez les informations du patient et uploadez l'image du nodule :")

# --- 1️ Formulaire pour les données tabulaires ---
with st.form(key='tabulaire_form'):
    age = st.number_input("Âge", min_value=0, max_value=120, value=50)
    sexe_masculin = st.radio("Sexe", options=[0,1], format_func=lambda x: "Femme" if x==0 else "Homme")
    presence_nodule = st.radio("Présence nodule", options=[0,1])
    subtilite_nodule = st.slider("Subtilité du nodule", min_value=0.0, max_value=10.0, step=0.1)
    taille_nodule_px = st.number_input("Taille du nodule (px)", min_value=1, max_value=1000, value=10)
    x_nodule_norm = st.slider("Position X normalisée", min_value=0.0, max_value=1.0, step=0.01)
    y_nodule_norm = st.slider("Position Y normalisée", min_value=0.0, max_value=1.0, step=0.01)
    tabagisme_paquets_annee = st.number_input("Tabagisme (paquets/année)", min_value=0.0, max_value=100.0, step=0.1)
    toux_chronique = st.radio("Toux chronique", options=[0,1])
    antecedent_familial = st.number_input("Antécédent familial (%)", min_value=0, max_value=100, value=0)

    uploaded_file = st.file_uploader("Upload image du nodule (PNG/JPG)", type=["png","jpg","jpeg"])
    submit_button = st.form_submit_button(label=' Prédire le risque')

# --- 2 Appel à l'API FastAPI ---
if submit_button:
    if uploaded_file is None:
        st.warning("Veuillez uploader une image pour la prédiction.")
    else:
        # Préparer les données pour l'API
        files = {"file": uploaded_file.getvalue()}
        params = {
            "age": age,
            "sexe_masculin": sexe_masculin,
            "presence_nodule": presence_nodule,
            "subtilite_nodule": subtilite_nodule,
            "taille_nodule_px": taille_nodule_px,
            "x_nodule_norm": x_nodule_norm,
            "y_nodule_norm": y_nodule_norm,
            "tabagisme_paquets_annee": tabagisme_paquets_annee,
            "toux_chronique": toux_chronique,
            "antecedent_familial": antecedent_familial
        }

        #  Adapter l'URL selon où ton FastAPI tourne
        api_url = "http://127.0.0.1:8000/predict_patient"

        response = requests.post(api_url, params=params, files={"file": uploaded_file})
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"Résultat : {data['résultat']}")
            st.write(f"Probabilité de cancer : {data['prob_cancer']*100:.2f}%")
            st.write("Probabilités du modèle tabulaire :", data['proba_tabulaire'])
        else:
            st.error(f"Erreur API : {response.status_code}")