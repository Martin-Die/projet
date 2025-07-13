from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from pathlib import Path

# Variables globales pour le modèle et les métadonnées
model = None
class_names = None
model_metadata = None

def load_latest_model():
    """
    Charge le modèle le plus récent
    """
    global model, class_names, model_metadata
    
    print("Chargement du modele...")
    
    # Chercher le modèle le plus récent - depuis le dossier racine du projet
    # L'API est dans API/, donc on remonte d'un niveau pour aller à la racine
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    print(f"Recherche dans: {models_dir.absolute()}")
    
    if not models_dir.exists():
        print("Dossier models non trouve - creation du dossier...")
        models_dir.mkdir(exist_ok=True)
        raise Exception("Aucun modele trouve dans le dossier models")
    
    # Chercher les dossiers de modèles entraînés (supporte les deux types de noms)
    model_paths = [p for p in models_dir.iterdir() if p.is_dir() and (
        p.name.startswith("potato_disease_model_v") or 
        p.name.startswith("multi_crop_disease_model_v") or
        p.name.startswith("disease_model_v")
    )]
    
    if not model_paths:
        print("Aucun modele trouve dans le dossier models")
        print("Contenu du dossier models:")
        for item in models_dir.iterdir():
            print(f"  - {item.name} ({'dossier' if item.is_dir() else 'fichier'})")
        print("\nPour entrainer un modele, executez: python main.py (option 1)")
        raise Exception("Aucun modele trouve - veuillez d'abord entrainer un modele")
    
    # Prendre le modèle avec la version la plus élevée
    # Extraire le numéro de version en gérant les différents formats de noms
    def extract_version(path):
        name = path.name
        if name.startswith("potato_disease_model_v"):
            return int(name.split("v")[1])
        elif name.startswith("multi_crop_disease_model_v"):
            return int(name.split("v")[1])
        elif name.startswith("disease_model_v"):
            return int(name.split("v")[1])
        else:
            return 0
    
    latest_model_path = max(model_paths, key=extract_version)
    
    print(f"Chargement du modele: {latest_model_path}")
    
    # Charger le modèle - essayer différents formats
    model = None
    model_file = None
    
    # Essayer d'abord le format natif Keras (.keras)
    keras_file = latest_model_path / "model.keras"
    if keras_file.exists():
        try:
            model = tf.keras.models.load_model(str(keras_file))
            model_file = keras_file
            print(f"Modele charge avec format natif Keras: {keras_file}")
        except Exception as e:
            print(f"Erreur chargement format Keras: {e}")
    
    # Si échec, essayer le format HDF5 (.h5)
    if model is None:
        h5_file = latest_model_path / "model.h5"
        if h5_file.exists():
            try:
                model = tf.keras.models.load_model(str(h5_file))
                model_file = h5_file
                print(f"Modele charge avec format HDF5: {h5_file}")
            except Exception as e:
                print(f"Erreur chargement format HDF5: {e}")
    
    # Si échec, essayer le format SavedModel
    if model is None:
        saved_model_dir = latest_model_path / "saved_model"
        if saved_model_dir.exists():
            try:
                model = tf.keras.models.load_model(str(saved_model_dir))
                model_file = saved_model_dir
                print(f"Modele charge avec format SavedModel: {saved_model_dir}")
            except Exception as e:
                print(f"Erreur chargement format SavedModel: {e}")
    
    if model is None:
        raise Exception(f"Aucun modele valide trouve dans {latest_model_path}")
    
    # Charger les métadonnées
    metadata_path = latest_model_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        class_names = model_metadata['class_names']
    else:
        # Fallback si pas de métadonnées
        class_names = ["Plante_saine", "Mildiou_precoce", "Mildiou_tardif"]
    
    print(f"Modele charge avec succes!")
    print(f"   Classes: {class_names}")
    print(f"   Version: {latest_model_path.name}")

def preprocess_image(image_bytes):
    """
    Prépare l'image pour la prédiction
    """
    # Charger l'image avec PIL
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner à 256x256
    image = image.resize((256, 256))
    
    # Convertir en array numpy
    image_array = np.array(image)
    
    # Normaliser (0-1)
    image_array = image_array.astype(np.float32) / 255.0
    
    # Ajouter la dimension du batch
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application (remplace @app.on_event)
    """
    # Startup
    try:
        load_latest_model()
    except Exception as e:
        print(f"Erreur lors du chargement du modele: {e}")
        print("L'API demarrera sans modele charge")
        print("Pour entrainer un modele, executez: python main.py (option 1)")
    
    yield
    
    # Shutdown (nettoyage si nécessaire)
    print("Arret de l'API...")

# Créer l'application FastAPI avec le gestionnaire de cycle de vie
app = FastAPI(
    title="API Diagnostic Maladies Pommes de Terre",
    description="API pour le diagnostic automatique des maladies de pommes de terre",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Endpoint racine - informations générales
    """
    return {
        "message": "API Diagnostic Maladies Pommes de Terre",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "model_status": "Charge" if model is not None else "Non charge - entrainez d'abord un modele",
        "endpoints": {
            "ping": "/ping",
            "model_info": "/model-info",
            "predict": "/predict",
            "docs": "/docs"
        },
        "instructions": {
            "if_model_not_loaded": "Executez: python main.py (option 1) pour entrainer un modele",
            "test_api": "Visitez /docs pour tester l'API interactivement"
        }
    }

@app.get("/ping")
async def ping():
    """
    Endpoint de test pour vérifier que le serveur fonctionne
    """
    return {
        "message": "pong",
        "status": "ok",
        "model_loaded": model is not None
    }

@app.get("/model-info")
async def get_model_info():
    """
    Retourne les informations sur le modèle chargé
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Modele non charge",
                "message": "Aucun modele trouve. Veuillez d'abord entrainer un modele avec: python main.py (option 1)",
                "solution": "Executez le pipeline complet pour creer un modele"
            }
        )
    
    return {
        "model_loaded": True,
        "class_names": class_names,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "metadata": model_metadata
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint de prédiction - accepte une image et retourne le diagnostic
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Modele non charge",
                "message": "Aucun modele trouve. Veuillez d'abord entrainer un modele",
                "solution": "Executez: python main.py (option 1) pour entrainer un modele"
            }
        )
    
    # Vérifier le type de fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit etre une image")
    
    try:
        # Lire les bytes de l'image
        image_bytes = await file.read()
        
        # Prétraiter l'image
        processed_image = preprocess_image(image_bytes)
        
        # Faire la prédiction
        predictions = model.predict(processed_image, verbose=0)
        
        # Obtenir la classe prédite et la confiance
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = class_names[predicted_class_idx]
        
        # Obtenir toutes les probabilités
        probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(class_names, predictions[0])
        }
        
        # Déterminer le statut de santé
        if "saine" in predicted_class.lower():
            health_status = "saine"
            recommendation = "Continuez à surveiller vos cultures régulièrement."
        elif "mildiou" in predicted_class.lower():
            health_status = "malade"
            if "precoce" in predicted_class.lower():
                recommendation = "Traitement préventif recommandé. Surveillez l'évolution."
            else:
                recommendation = "Traitement curatif urgent nécessaire. Consultez un expert."
        else:
            health_status = "incertain"
            recommendation = "Diagnostic incertain. Prenez une nouvelle photo ou consultez un expert."
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "health_status": health_status,
            "recommendation": recommendation,
            "probabilities": probabilities,
            "all_classes": class_names
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Endpoint de vérification de santé
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 