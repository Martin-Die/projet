# Résultats de l'entraînement - Diagnostic Maladies Pommes de Terre

## Fichiers générés

### Visualisations
- `training_curves.png` - Courbes d'accuracy et de loss
- `confusion_matrix.png` - Matrice de confusion
- `test_predictions.png` - Exemples de prédictions

### Données
- `training_history.json` - Historique complet de l'entraînement
- `evaluation_results.json` - Résultats d'évaluation détaillés
- `test_predictions_data.json` - Données des prédictions de test
- `training_summary.json` - Résumé complet de l'entraînement

## Performance du modèle

- **Classes détectées**: Pepper__bell___Bacterial_spot, Pepper__bell___healthy, Potato___Early_blight, Potato___Late_blight, Potato___healthy, Tomato_Bacterial_spot, Tomato_Early_blight, Tomato_Late_blight, Tomato_Leaf_Mold, Tomato_Septoria_leaf_spot, Tomato_Spider_mites_Two_spotted_spider_mite, Tomato__Target_Spot, Tomato__Tomato_YellowLeaf__Curl_Virus, Tomato__Tomato_mosaic_virus, Tomato_healthy
- **Précision finale (validation)**: 62.29%
- **Perte finale (validation)**: 1.3677
- **Meilleure précision**: 64.62% (epoch 2)
- **Paramètres totaux**: 1,328,399

## Entraînement effectué le: 2025-07-11 16:47:32

## Utilisation

Le modèle entraîné est sauvegardé dans le dossier `models/` et peut être utilisé avec l'API FastAPI.
