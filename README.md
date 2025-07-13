# 🥔 Application de Diagnostic des Maladies de Pommes de Terre

Une application mobile complète permettant aux producteurs de pommes de terre de photographier une feuille de plante et de recevoir un diagnostic automatique grâce à un modèle Deep Learning entraîné pour distinguer plante saine, mildiou précoce et mildiou tardif.

## Objectif

Construire une application mobile complète qui permet aux producteurs de pommes de terre de photographier une feuille de plante et de recevoir un diagnostic automatique grâce à un modèle Deep Learning entraîné pour distinguer :
- **Plante saine**
- **Mildiou précoce** 
- **Mildiou tardif**

## Architecture Technique

### 1. Collecte des données et préparation
- Utilisation du dataset PlantVillage (Kaggle)
- Extraction et organisation des données de pommes de terre
- Chargement avec `tf.keras.preprocessing.image_dataset_from_directory`
- Split en train/validation/test (80%/10%/10%)
- Data augmentation et normalisation
- Pipeline optimisé avec cache, shuffle et prefetch

### 2. Construction et entraînement du modèle
- Modèle CNN avec `tf.keras.Sequential`
- Intégration des couches de prétraitement
- Entraînement avec callbacks (EarlyStopping, ReduceLROnPlateau)
- Évaluation et visualisation des performances
- Sauvegarde avec versioning automatique

### 3. Déploiement ML Ops avec FastAPI
- API REST avec FastAPI
- Endpoint `/predict` pour les prédictions
- Endpoint `/ping` pour les tests de santé
- Documentation interactive automatique
- Gestion des erreurs et validation

### 4. Interface utilisateur React.js
- Application web moderne et responsive
- Upload d'images et affichage des résultats
- Interface intuitive pour les producteurs

## Structure du Projet

```
projet/
├── data/                          # Données brutes
│   └── archive.zip               # Dataset PlantVillage
├── plantvillage/                 # Images organisées par classe
├── models/                       # Modèles entraînés
│   ├── preprocessing_layers.pkl  # Couches de prétraitement
│   └── potato_disease_model_v1/  # Modèle versionné
├── API/                          # Serveur FastAPI
│   └── main.py                   # API principale
├── frontend/                     # Application React (à configurer)
├── extract_data.py               # Extraction des données
├── data_preparation.py           # Préparation des données
├── train_model.py                # Entraînement du modèle
├── main.py                       # Script principal
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation
```

## Installation et Utilisation

### Prérequis
- Python 3.8+
- Node.js et npm (pour le frontend)
- Git

### 1. Installation des dépendances

```bash
# Installer les dépendances Python
pip install -r requirements.txt
```

### 2. Exécution du pipeline complet

```bash
# Lancer le script principal
python main.py

# Choisir l'option 1 pour exécuter tout le pipeline
```

### 3. Lancement de l'API

```bash
# Option 1: Via le script principal
python main.py
# Choisir l'option 5

# Option 2: Directement
cd API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test de l'API

```bash
# Test de santé
curl http://localhost:8000/ping

# Documentation interactive
# Ouvrir http://localhost:8000/docs dans le navigateur
```

### 5. Configuration du frontend

```bash
# Créer la configuration frontend
python main.py
# Choisir l'option 7

# Installer et lancer le frontend (après avoir reçu le code)
cd frontend
npm install
npm start
```

## Fonctionnalités

### Extraction et Préparation des Données
- Extraction automatique de l'archive PlantVillage
- Organisation des dossiers de pommes de terre
- Chargement avec TensorFlow Dataset API
- Visualisation d'images d'exemple
- Split train/validation/test
- Data augmentation (flip, rotation, contraste)
- Pipeline optimisé (cache, shuffle, prefetch)

### Entraînement du Modèle
- Architecture CNN avec TensorFlow/Keras
- Intégration des couches de prétraitement
- Callbacks pour optimiser l'entraînement
- Visualisation des courbes d'entraînement
- Évaluation complète (matrice de confusion, rapport)
- Test de prédictions sur images d'exemple
- Sauvegarde avec versioning automatique

### API FastAPI
- Endpoint `/ping` pour les tests de santé
- Endpoint `/predict` pour les prédictions
- Validation des fichiers d'entrée
- Gestion des erreurs
- Documentation interactive automatique
- CORS configuré pour le frontend
- Chargement automatique du modèle le plus récent

### Interface Utilisateur
- Configuration automatique
- Interface moderne et responsive
- Upload d'images
- Affichage des résultats de diagnostic
- Recommandations personnalisées

## 🔧 Scripts Disponibles

### `extract_data.py`
- Extraction de l'archive PlantVillage
- Organisation des dossiers de pommes de terre
- Création de la structure du projet

### `data_preparation.py`
- Chargement des images avec TensorFlow
- Visualisation et analyse des données
- Split en datasets d'entraînement/validation/test
- Création des couches de prétraitement
- Optimisation du pipeline

### `train_model.py`
- Construction de l'architecture CNN
- Entraînement avec callbacks
- Évaluation et visualisation
- Sauvegarde avec versioning

### `API/main.py`
- Serveur FastAPI complet
- Endpoints de prédiction et santé
- Gestion des erreurs et validation
- Documentation automatique

### `main.py`
- Script principal avec menu interactif
- Exécution séquentielle des étapes
- Tests et configuration automatique

## 📈 Performances du Modèle

Le modèle CNN entraîné offre :
- **Architecture** : CNN avec couches de convolution et dropout
- **Optimiseur** : Adam avec learning rate adaptatif
- **Métriques** : Accuracy, Loss, Matrice de confusion
- **Data Augmentation** : Flip horizontal, rotation, contraste
- **Callbacks** : EarlyStopping, ReduceLROnPlateau

## 🌐 API Endpoints

### GET `/ping`
Test de santé de l'API
```json
{
  "message": "pong",
  "status": "ok",
  "model_loaded": true
}
```

### GET `/model-info`
Informations sur le modèle chargé
```json
{
  "model_loaded": true,
  "class_names": ["Plante_saine", "Mildiou_precoce", "Mildiou_tardif"],
  "input_shape": [null, 256, 256, 3],
  "output_shape": [null, 3]
}
```

### POST `/predict`
Prédiction sur une image
```json
{
  "predicted_class": "Plante_saine",
  "confidence": 0.95,
  "health_status": "saine",
  "recommendation": "Continuez à surveiller vos cultures régulièrement.",
  "probabilities": {
    "Plante_saine": 0.95,
    "Mildiou_precoce": 0.03,
    "Mildiou_tardif": 0.02
  }
}
```

## 🎨 Interface Utilisateur

L'application React offre :
- **Upload d'images** : Glisser-déposer ou sélection de fichier
- **Diagnostic en temps réel** : Affichage immédiat des résultats
- **Recommandations** : Conseils personnalisés selon le diagnostic
- **Historique** : Sauvegarde des prédictions précédentes
- **Interface responsive** : Compatible mobile et desktop

## Tests et Validation

### Tests de l'API
```bash
# Test de santé
curl http://localhost:8000/ping

# Test de prédiction (avec Postman ou curl)
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

### Validation du Modèle
- Matrice de confusion
- Rapport de classification
- Courbes d'entraînement
- Test sur images d'exemple

## Déploiement

### Développement
```bash
# API
uvicorn API.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm start
```

### Production
- Utiliser Gunicorn pour l'API
- Configurer un reverse proxy (Nginx)
- Déployer le frontend sur un CDN
- Utiliser Docker pour la conteneurisation

## 📚 Documentation

- **API Documentation** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **Health Check** : http://localhost:8000/ping

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Équipe Deep Learning** - *Développement initial*

## 🙏 Remerciements

- Dataset PlantVillage pour les images
- TensorFlow/Keras pour le framework Deep Learning
- FastAPI pour l'API REST
- React.js pour l'interface utilisateur

---

**🥔 Application de Diagnostic des Maladies de Pommes de Terre** - Une solution complète pour l'agriculture intelligente. 