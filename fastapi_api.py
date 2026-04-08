from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import io

# Charger les modèles
tab_model = joblib.load("modele_tabulaire.pkl")
scaler = joblib.load("scaler.pkl")
multimodal_model = tf.keras.models.load_model("cnn_multimodal_model.h5")

IMG_SIZE = (224, 224)

app = FastAPI(title="Détection Cancer Pulmonaire")

# Prétraitement image
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 1)
    return img

# Prétraitement tabulaire
def preprocess_tabulaire(data_dict):
    # Transformer en array et scaler
    # ⚠️ dyspnee supprimé pour correspondre à l'entraînement
    X = np.array([[data_dict['age'],
                   data_dict['sexe_masculin'],
                   data_dict['presence_nodule'],
                   data_dict['subtilite_nodule'],
                   data_dict['taille_nodule_px'],
                   data_dict['x_nodule_norm'],
                   data_dict['y_nodule_norm'],
                   data_dict['tabagisme_paquets_annee'],
                   data_dict['toux_chronique'],
                   data_dict['antecedent_familial']]])
    X_scaled = scaler.transform(X)
    return X_scaled

# Endpoint de prédiction finale
@app.post("/predict_patient")
async def predict_patient(
    file: UploadFile = File(...),
    age: float = 0,
    sexe_masculin: int = 0,
    presence_nodule: int = 0,
    subtilite_nodule: float = 0,
    taille_nodule_px: float = 0,
    x_nodule_norm: float = 0,
    y_nodule_norm: float = 0,
    tabagisme_paquets_annee: float = 0,
    toux_chronique: int = 0,
    antecedent_familial: int = 0
):
    # --- 1️⃣ Prétraiter les données tabulaires ---
    tab_data = preprocess_tabulaire({
        'age': age,
        'sexe_masculin': sexe_masculin,
        'presence_nodule': presence_nodule,
        'subtilite_nodule': subtilite_nodule,
        'taille_nodule_px': taille_nodule_px,
        'x_nodule_norm': x_nodule_norm,
        'y_nodule_norm': y_nodule_norm,
        'tabagisme_paquets_annee': tabagisme_paquets_annee,
        'toux_chronique': toux_chronique,
        'antecedent_familial': antecedent_familial
    })

    # Probabilités du modèle tabulaire
    proba_tab = tab_model.predict_proba(tab_data)[0]

    # --- 2️⃣ Prétraiter l'image ---
    img_bytes = await file.read()
    img = preprocess_image(img_bytes)

    # --- 3️⃣ Prédiction modèle multimodal ---
    # reshaper la probabilité tabulaire pour l'entrée multimodale
    multimodal_input = [img, proba_tab.reshape(1, -1)]
    prob_cancer = multimodal_model.predict(multimodal_input)[0][0]
    result = "Cancer détecté" if prob_cancer > 0.5 else "Pas de cancer"

    return JSONResponse({
        "prob_cancer": float(prob_cancer),
        "résultat": result,
        "proba_tabulaire": proba_tab.tolist()
    })