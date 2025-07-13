import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
import json
import time


def train_model():
    """
    Script 2: Construction et entraînement du modèle
    """
    print("Debut de l'entrainement du modele...")

    try:
        # Charger les données et couches de prétraitement
        train_ds, val_ds, test_ds, class_names, preprocessing_layers = load_training_data()

        if train_ds is None:
            print(
                "Impossible de charger les donnees. Executez d'abord data_preparation.py")
            return None, None

        # Construire le modèle
        model = build_model(preprocessing_layers, len(class_names))

        # Compiler le modèle
        compile_model(model)

        # Entraîner le modèle
        history = train_model_epochs(model, train_ds, val_ds)

        # Évaluer le modèle
        evaluate_model(model, test_ds, class_names)

        # Visualiser les courbes d'entraînement
        plot_training_curves(history)

        # Tester le modèle sur quelques images
        test_predictions(model, test_ds, class_names)

        # Sauvegarder le modèle avec versioning
        print("\n" + "="*50)
        print("SAUVEGARDE DU MODELE")
        print("="*50)
        # Récupérer le type de dataset utilisé
        dataset_type = os.environ.get('DATASET_CHOICE', 'potato_only')
        save_success = save_model_with_versioning(
            model, class_names, dataset_type)

        if save_success:
            print("\n" + "="*50)
            print("ENTRAINEMENT TERMINE AVEC SUCCES!")
            print("="*50)
            print("Le modele a ete sauvegarde et est pret pour le deploiement.")

            # Créer un résumé des résultats
            create_training_summary(model, class_names, history, dataset_type)

            # Vérifier le contenu du dossier models
            models_dir = Path('models')
            if models_dir.exists():
                print("\nContenu du dossier models:")
                for item in models_dir.iterdir():
                    if item.is_dir():
                        print(f"  - {item.name}")
                    else:
                        print(f"  - {item.name}")

            # Vérifier le contenu du dossier results
            results_dir = Path('results')
            if results_dir.exists():
                print("\nContenu du dossier results:")
                for item in results_dir.iterdir():
                    if item.is_file():
                        print(f"  - {item.name}")
        else:
            print("\nERREUR: Le modele n'a pas pu etre sauvegarde!")
            print("Veuillez verifier les permissions du dossier models/")

        return model, history

    except Exception as e:
        print(f"\nERREUR lors de l'entrainement: {e}")
        print("L'entrainement a echoue. Veuillez verifier les donnees et reessayer.")
        return None, None


def load_training_data():
    """
    Charge les données d'entraînement et les couches de prétraitement
    """
    print("Chargement des donnees d'entrainement...")

    # Vérifier que les fichiers existent
    if not os.path.exists('models/preprocessing_layers.pkl'):
        print("Couches de pretraitement non trouvees. Executez d'abord data_preparation.py")
        return None, None, None, None, None

    # Charger les couches de prétraitement
    with open('models/preprocessing_layers.pkl', 'rb') as f:
        preprocessing_layers = pickle.load(f)

    # Lire le choix de dataset depuis la variable d'environnement
    dataset_choice = os.environ.get('DATASET_CHOICE', 'potato_only')
    print(f"Choix de dataset: {dataset_choice}")

    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 32

    if dataset_choice == 'potato_only':
        # Utiliser uniquement les données de pommes de terre (comportement par défaut)
        print("Chargement des donnees de pommes de terre uniquement...")
        data_path = 'plantvillage'
    elif dataset_choice == 'all_datasets':
        # Utiliser tous les datasets disponibles
        print("Chargement de tous les datasets disponibles...")
        data_path = 'data/PlantVillage'
    else:
        print(
            f"Choix de dataset invalide: {dataset_choice}, utilisation des donnees de pommes de terre")
        data_path = 'plantvillage'

    # Utiliser la nouvelle structure training/train et training/test créée par data_preparation.py
    print("Chargement depuis la structure training/train et training/test...")

    # Vérifier que les dossiers training/train et training/test existent
    if not os.path.exists('training/train') or not os.path.exists('training/test'):
        print("Dossiers training/train/ et training/test/ non trouves. Executez d'abord data_preparation.py")
        return None, None, None, None, None

    # Charger le dataset d'entraînement
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'training/train',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        label_mode='categorical'
    )

    # Charger le dataset de test
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'training/test',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=42,
        label_mode='categorical'
    )

    class_names = train_dataset.class_names
    print(f"Classes trouvees: {len(class_names)}")
    for i, class_name in enumerate(class_names):
        print(f"  {i+1}. {class_name}")

    # Compter les fichiers sans charger en mémoire
    train_count = count_files_in_directory('training/train')
    test_count = count_files_in_directory('training/test')

    print(f"Nombre d'images:")
    print(f"  Entrainement: {train_count} images")
    print(f"  Test: {test_count} images")

    # Split du dataset d'entraînement en train/validation (20% pour validation)
    # Nombre de batches pour validation
    val_size = int(train_count * 0.2 / BATCH_SIZE)

    train_ds = train_dataset.skip(val_size)
    val_ds = train_dataset.take(val_size)
    test_ds = test_dataset

    print(f"Repartition des batches:")
    print(
        f"  Entrainement: {val_size} batches pour validation, reste pour entrainement")
    print(f"  Test: utilise directement le dossier training/test")

    # Préparer les pipelines
    train_ds = prepare_pipeline(train_ds, shuffle=True)
    val_ds = prepare_pipeline(val_ds, shuffle=False)
    test_ds = prepare_pipeline(test_ds, shuffle=False)

    return train_ds, val_ds, test_ds, class_names, preprocessing_layers


def build_model(preprocessing_layers, num_classes):
    """
    Construit le modèle avec tf.keras.Sequential
    """
    print("Construction du modele...")

    model = tf.keras.Sequential([
        # Couches de prétraitement
        preprocessing_layers['resize_and_rescale'],
        preprocessing_layers['data_augmentation'],

        # Couches de convolution
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # Couches de dropout pour éviter l'overfitting
        tf.keras.layers.Dropout(0.2),

        # Couches de convolution supplémentaires
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # Couches de dropout
        tf.keras.layers.Dropout(0.2),

        # Couches de classification
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    print("Modele construit!")
    return model


def compile_model(model):
    """
    Compile le modèle avec les paramètres optimaux
    """
    print("Compilation du modele...")

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Modele compile!")
    print("   Optimiseur: Adam")
    print("   Loss: Categorical Crossentropy")
    print("   Metrique: Accuracy")


def train_model_epochs(model, train_ds, val_ds, epochs=3):
    """
    Entraîne le modèle
    """
    print(f"Debut de l'entrainement ({epochs} epochs)...")

    # Callbacks pour améliorer l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
    ]

    # Entraînement
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("Entrainement termine!")
    return history


def evaluate_model(model, test_ds, class_names):
    """
    Évalue le modèle sur le jeu de test
    """
    print("Evaluation du modele...")

    # Créer le dossier results s'il n'existe pas
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Évaluation
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")

    # Prédictions sur le test set
    predictions = model.predict(test_ds, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Obtenir les vraies classes
    true_classes = []
    for images, labels in test_ds:
        true_classes.extend(np.argmax(labels.numpy(), axis=1))

    # Calculer la matrice de confusion
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(true_classes, predicted_classes)

    print("\nMatrice de confusion:")
    print(cm)

    print("\nRapport de classification:")
    report = classification_report(
        true_classes, predicted_classes, target_names=class_names, output_dict=True)
    print(classification_report(true_classes,
          predicted_classes, target_names=class_names))

    # Sauvegarder les résultats d'évaluation
    evaluation_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': cm.tolist(),
        # Convertir en string pour éviter les problèmes JSON
        'classification_report': str(report),
        'class_names': [str(name) for name in class_names],
        'total_samples': len(true_classes),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    evaluation_path = results_dir / 'evaluation_results.json'
    with open(evaluation_path, 'w') as f:
        json.dump(clean_for_json(evaluation_results), f, indent=2)
    print(f"\nResultats d'evaluation sauvegardes: {evaluation_path}")

    # Créer et sauvegarder une visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [name.replace('_', ' ')
               for name in class_names], rotation=45)
    plt.yticks(tick_marks, [name.replace('_', ' ') for name in class_names])

    # Ajouter les valeurs dans les cases
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')

    confusion_matrix_path = results_dir / 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Matrice de confusion sauvegardee: {confusion_matrix_path}")


def plot_training_curves(history):
    """
    Trace les courbes de précision et de perte
    """
    print("Generation des courbes d'entrainement...")

    # Créer le dossier results s'il n'existe pas
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Courbe de précision
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Courbe de perte
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Sauvegarder dans le dossier results
    curves_path = results_dir / 'training_curves.png'
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Courbes sauvegardees: {curves_path}")

    # Sauvegarder aussi les données numériques
    training_data = {
        'epochs': list(range(1, len(history.history['accuracy']) + 1)),
        'training_accuracy': history.history['accuracy'],
        'validation_accuracy': history.history['val_accuracy'],
        'training_loss': history.history['loss'],
        'validation_loss': history.history['val_loss']
    }

    training_data_path = results_dir / 'training_history.json'
    with open(training_data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Donnees d'entrainement sauvegardees: {training_data_path}")


def test_predictions(model, test_ds, class_names, num_samples=8):
    """
    Teste le modèle sur quelques images et affiche les résultats
    """
    print("Test de predictions sur des images d'exemple...")

    # Créer le dossier results s'il n'existe pas
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(20, 10))

    sample_count = 0
    predictions_data = []

    for images, labels in test_ds:
        if sample_count >= num_samples:
            break

        for i in range(min(len(images), num_samples - sample_count)):
            # Prédiction
            prediction = model.predict(tf.expand_dims(images[i], 0), verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            # Vraie classe
            true_class = np.argmax(labels[i])

            # Stocker les données de prédiction
            predictions_data.append({
                'sample_id': sample_count + 1,
                'true_class': class_names[true_class],
                'predicted_class': class_names[predicted_class],
                'confidence': float(confidence),
                # conversion explicite
                'correct': bool(predicted_class == true_class)
            })

            # Affichage
            plt.subplot(2, 4, sample_count + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            title = f"Vrai: {class_names[true_class]}\nPredit: {class_names[predicted_class]}\nConfiance: {confidence:.2f}"
            plt.title(title, fontsize=10)
            plt.axis("off")

            sample_count += 1

    plt.tight_layout()

    # Sauvegarder dans le dossier results
    predictions_path = results_dir / 'test_predictions.png'
    plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Predictions sauvegardees: {predictions_path}")

    # Correction : conversion explicite des booléens pour JSON
    for d in predictions_data:
        if "correct" in d:
            d["correct"] = bool(d["correct"])

    # Sauvegarder les données de prédiction
    predictions_data_path = results_dir / 'test_predictions_data.json'
    with open(predictions_data_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"Donnees de predictions sauvegardees: {predictions_data_path}")

    # Calculer et afficher les statistiques
    correct_predictions = sum(1 for p in predictions_data if p['correct'])
    accuracy = correct_predictions / len(predictions_data)
    print(
        f"Precision sur les echantillons de test: {accuracy:.2%} ({correct_predictions}/{len(predictions_data)})")


def save_model_with_versioning(model, class_names, dataset_type=None):
    """
    Sauvegarde le modèle avec versioning automatique
    """
    print("Sauvegarde du modele...")

    try:
        # Créer le dossier models s'il n'existe pas
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        print(f"Dossier models verifie: {models_dir.absolute()}")

        # Vérifier les permissions d'écriture
        if not os.access(models_dir, os.W_OK):
            raise Exception(
                f"Pas de permission d'ecriture dans le dossier {models_dir}")

        print("Permissions d'ecriture verifiees")

        # Déterminer le type de dataset utilisé
        if dataset_type is None:
            dataset_type = os.environ.get('DATASET_CHOICE', 'potato_only')

        # Créer le nom de base selon le type de dataset
        if dataset_type == 'potato_only':
            base_name = 'potato_disease_model'
        elif dataset_type == 'all_datasets':
            base_name = 'multi_crop_disease_model'
        else:
            base_name = 'disease_model'

        # Trouver la prochaine version
        version = 1
        while os.path.exists(f'models/{base_name}_v{version}'):
            version += 1

        # Créer le dossier pour cette version
        model_dir = f'models/{base_name}_v{version}'
        os.makedirs(model_dir, exist_ok=True)

        # Sauvegarder le modèle avec le format natif Keras (.keras) - plus stable
        model_path = f'{model_dir}/model.keras'
        print(f"Sauvegarde dans: {model_path}")

        # Sauvegarder le modèle avec gestion d'erreur détaillée
        try:
            # Utiliser le format natif Keras (recommandé)
            model.save(model_path)
            print("Modele sauvegarde avec succes (format natif Keras)!")
        except Exception as save_error:
            print(f"Erreur lors de la sauvegarde: {save_error}")
            print("Tentative avec format HDF5...")
            try:
                # Fallback vers HDF5 si nécessaire
                h5_path = f'{model_dir}/model.h5'
                model.save(h5_path, save_format='h5')
                model_path = h5_path
                print("Modele sauvegarde avec format HDF5!")
            except Exception as h5_error:
                print(f"Erreur avec format HDF5: {h5_error}")
                print("Tentative avec format SavedModel...")
                try:
                    # Dernier recours: format SavedModel
                    saved_model_path = f'{model_dir}/saved_model'
                    model.save(saved_model_path, save_format='tf')
                    model_path = saved_model_path
                    print("Modele sauvegarde avec format SavedModel!")
                except Exception as tf_error:
                    print(f"Erreur avec format SavedModel: {tf_error}")
                    raise Exception(
                        f"Impossible de sauvegarder le modele avec aucun format: {tf_error}")

        # Vérifier que le modèle a bien été sauvegardé
        if not os.path.exists(model_path):
            raise Exception("Le modele n'a pas ete sauvegarde correctement")

        # Sauvegarder les métadonnées
        metadata = {
            'version': version,
            'dataset_type': dataset_type,
            'base_name': base_name,
            'class_names': [str(name) for name in class_names],
            'model_path': str(model_path),
            'model_dir': str(model_dir),
            'input_shape': [int(x) if x is not None else -1 for x in model.input_shape],
            'output_shape': [int(x) if x is not None else -1 for x in model.output_shape],
            'total_params': int(model.count_params()),
            'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
        }

        metadata_path = f'{model_dir}/metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(clean_for_json(metadata), f, indent=2)

        print(f"Metadonnees sauvegardees: {metadata_path}")

        # Vérification finale
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            print(f"Modele sauvegarde avec succes: {model_dir}")
            print(f"   Fichier modele: {model_path}")
            print(f"   Version: {version}")
            print(f"   Type de dataset: {dataset_type}")
            print(f"   Classes: {class_names}")
            print(f"   Parametres: {model.count_params():,}")
            return True
        else:
            raise Exception("Erreur lors de la sauvegarde")

    except Exception as e:
        print(f"ERREUR lors de la sauvegarde: {e}")
        print("Tentative de sauvegarde de secours...")

        try:
            # Sauvegarde de secours avec format natif Keras
            backup_dir = f'models/model_backup_{int(time.time())}'
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = f'{backup_dir}/model.keras'
            model.save(backup_path)
            print(f"Sauvegarde de secours reussie: {backup_path}")
            return True
        except Exception as backup_error:
            print(f"ERREUR de sauvegarde de secours: {backup_error}")
            # Dernière tentative avec SavedModel
            try:
                saved_model_backup = f'{backup_dir}/saved_model'
                model.save(saved_model_backup, save_format='tf')
                print(
                    f"Sauvegarde de secours SavedModel reussie: {saved_model_backup}")
                return True
            except Exception as final_error:
                print(f"ERREUR finale de sauvegarde: {final_error}")
                return False


def create_training_summary(model, class_names, history, dataset_type=None):
    """
    Crée un résumé complet de l'entraînement
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Déterminer le type de dataset utilisé
    if dataset_type is None:
        dataset_type = os.environ.get('DATASET_CHOICE', 'potato_only')

    # Créer le nom du modèle selon le type de dataset
    if dataset_type == 'potato_only':
        model_name = 'Potato Disease Classifier'
    elif dataset_type == 'all_datasets':
        model_name = 'Multi-Crop Disease Classifier'
    else:
        model_name = 'Disease Classifier'

    # Créer le résumé
    summary = {
        'training_info': {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'architecture': 'CNN Sequential',
            'total_parameters': int(model.count_params()),
            'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
            'classes': [str(name) for name in class_names],
            'num_classes': len(class_names)
        },
        'training_results': {
            'final_training_accuracy': float(history.history['accuracy'][-1]),
            'final_validation_accuracy': float(history.history['val_accuracy'][-1]),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history['val_loss'][-1]),
            'total_epochs': len(history.history['accuracy']),
            'best_validation_accuracy': float(max(history.history['val_accuracy'])),
            'best_validation_epoch': int(np.argmax(history.history['val_accuracy']) + 1)
        },
        'model_structure': {
            'input_shape': [int(x) if x is not None else -1 for x in model.input_shape],
            'output_shape': [int(x) if x is not None else -1 for x in model.output_shape],
            'layers': [str(layer.name) for layer in model.layers]
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'files_generated': [
            'training_curves.png',
            'training_history.json',
            'confusion_matrix.png',
            'evaluation_results.json',
            'test_predictions.png',
            'test_predictions_data.json'
        ]
    }

    # Sauvegarder le résumé
    summary_path = results_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(clean_for_json(summary), f, indent=2)

    print(f"Resume d'entrainement sauvegarde: {summary_path}")

    # Créer aussi un fichier README pour le dossier results
    readme_content = f"""# Résultats de l'entraînement - Diagnostic Maladies Pommes de Terre

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

- **Classes détectées**: {', '.join(class_names)}
- **Précision finale (validation)**: {summary['training_results']['final_validation_accuracy']:.2%}
- **Perte finale (validation)**: {summary['training_results']['final_validation_loss']:.4f}
- **Meilleure précision**: {summary['training_results']['best_validation_accuracy']:.2%} (epoch {summary['training_results']['best_validation_epoch']})
- **Paramètres totaux**: {summary['training_info']['total_parameters']:,}

## Entraînement effectué le: {summary['timestamp']}

## Utilisation

Le modèle entraîné est sauvegardé dans le dossier `models/` et peut être utilisé avec l'API FastAPI.
"""

    readme_path = results_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"README cree: {readme_path}")


def count_files_in_directory(directory_path):
    """
    Compte le nombre de fichiers d'images dans un répertoire sans les charger en mémoire
    """
    count = 0
    if os.path.exists(directory_path):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    count += 1
    return count


def clean_for_json(obj):
    """
    Nettoie un objet pour la sérialisation JSON
    """
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)


def prepare_pipeline(dataset, shuffle=True):
    """
    Prépare le pipeline optimisé
    """
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)

    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    # Exécuter l'entraînement
    model, history = train_model()

    if model is not None:
        print("\nModele entraine avec succes!")
        print("   Le modele est pret pour le deploiement.")
