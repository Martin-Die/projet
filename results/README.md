# R�sultats de l'entra�nement - Diagnostic Maladies Pommes de Terre

## Fichiers g�n�r�s

### Visualisations
- `training_curves.png` - Courbes d'accuracy et de loss
- `confusion_matrix.png` - Matrice de confusion
- `test_predictions.png` - Exemples de pr�dictions

### Donn�es
- `training_history.json` - Historique complet de l'entra�nement
- `evaluation_results.json` - R�sultats d'�valuation d�taill�s
- `test_predictions_data.json` - Donn�es des pr�dictions de test
- `training_summary.json` - R�sum� complet de l'entra�nement

## Performance du mod�le

- **Classes d�tect�es**: Pepper__bell___Bacterial_spot, Pepper__bell___healthy, Potato___Early_blight, Potato___Late_blight, Potato___healthy, Tomato_Bacterial_spot, Tomato_Early_blight, Tomato_Late_blight, Tomato_Leaf_Mold, Tomato_Septoria_leaf_spot, Tomato_Spider_mites_Two_spotted_spider_mite, Tomato__Target_Spot, Tomato__Tomato_YellowLeaf__Curl_Virus, Tomato__Tomato_mosaic_virus, Tomato_healthy
- **Pr�cision finale (validation)**: 62.29%
- **Perte finale (validation)**: 1.3677
- **Meilleure pr�cision**: 64.62% (epoch 2)
- **Param�tres totaux**: 1,328,399

## Entra�nement effectu� le: 2025-07-11 16:47:32

## Utilisation

Le mod�le entra�n� est sauvegard� dans le dossier `models/` et peut �tre utilis� avec l'API FastAPI.
