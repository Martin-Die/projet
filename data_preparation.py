import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from pathlib import Path
import pickle

def load_and_prepare_data():
    """
    Script 1: Collecte des données et préparation
    """
    print("Debut de la preparation des donnees...")
    
    # Paramètres
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 32
    SEED = 42
    
    # Lire le choix de dataset depuis la variable d'environnement
    dataset_choice = os.environ.get('DATASET_CHOICE', 'potato_only')
    print(f"Choix de dataset pour la preparation: {dataset_choice}")
    
    # Déterminer le dossier à utiliser
    if dataset_choice == 'potato_only':
        data_path = 'plantvillage'
        print("Preparation avec les donnees de pommes de terre uniquement")
    elif dataset_choice == 'all_datasets':
        # Pour tous les datasets, on doit combiner data/PlantVillage et plantvillage/
        data_path = 'data/PlantVillage'
        print("Preparation avec tous les datasets disponibles (incluant pommes de terre)")
    else:
        data_path = 'plantvillage'
        print(f"Choix de dataset invalide: {dataset_choice}, utilisation des donnees de pommes de terre")
    
    # Vérifier que le dossier existe
    if not os.path.exists(data_path):
        print(f"Dossier '{data_path}' non trouve. Executez d'abord extract_data.py")
        return None, None, None
    
    # Créer la structure train/test
    create_train_test_structure(data_path, SEED)
    
    # Charger les images depuis la nouvelle structure
    print(f"Chargement des images depuis la structure training/train et training/test...")
    
    # Charger le dataset d'entraînement
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'training/train',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        label_mode='categorical'
    )
    
    # Charger le dataset de test
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'training/test',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        label_mode='categorical'
    )
    
    # Vérifier le nombre total d'images et les classes détectées
    class_names = train_dataset.class_names
    print(f"Classes detectees: {class_names}")
    print(f"Nombre de classes: {len(class_names)}")
    
    # Compter le nombre total d'images sans charger en mémoire
    train_images = count_files_in_directory('training/train')
    test_images = count_files_in_directory('training/test')
    total_images = train_images + test_images
    print(f"Nombre total d'images: {total_images}")
    print(f"  - Entrainement: {train_images}")
    print(f"  - Test: {test_images}")
    
    # Parcourir un batch du dataset et afficher les shapes et labels
    print("\nAnalyse d'un batch d'entrainement:")
    for images, labels in train_dataset.take(1):
        print(f"   Shape des images: {images.shape}")
        print(f"   Shape des labels: {labels.shape}")
        print(f"   Labels uniques: {np.unique(tf.argmax(labels, axis=1).numpy())}")
        break
    
    # Visualiser quelques images annotées
    visualize_sample_images(train_dataset, class_names)
    
    # Split du dataset d'entraînement en train/validation
    train_ds, val_ds = split_train_validation(train_dataset, val_split=0.2, seed=SEED)
    
    # Calculer les tailles sans charger en mémoire
    train_batches = int(train_images / BATCH_SIZE)
    val_batches = int(train_batches * 0.2)
    test_batches = int(test_images / BATCH_SIZE)
    
    print(f"\nRepartition des donnees:")
    print(f"   Entrainement: {train_batches - val_batches} batches")
    print(f"   Validation: {val_batches} batches")
    print(f"   Test: {test_batches} batches")
    
    # Créer les couches de prétraitement
    preprocessing_layers = create_preprocessing_layers(IMAGE_SIZE)
    
    # Préparer les pipelines optimisés
    train_ds = prepare_pipeline(train_ds, shuffle=True)
    val_ds = prepare_pipeline(val_ds, shuffle=False)
    test_ds = prepare_pipeline(test_dataset, shuffle=False)
    
    # Sauvegarder les couches de prétraitement
    save_preprocessing_layers(preprocessing_layers)
    
    print("Preparation des donnees terminee!")
    return train_ds, val_ds, test_ds, class_names, preprocessing_layers

def create_train_test_structure(source_path, seed=42):
    """
    Crée la structure train/test en copiant les fichiers
    """
    print("Creation de la structure train/test...")
    
    # Créer le dossier training et les sous-dossiers train et test
    training_dir = Path('training')
    train_dir = training_dir / 'train'
    test_dir = training_dir / 'test'
    
    # Supprimer le dossier training existant s'il existe
    if training_dir.exists():
        shutil.rmtree(training_dir)
    
    training_dir.mkdir()
    train_dir.mkdir()
    test_dir.mkdir()
    
    # Déterminer si on doit inclure les pommes de terre
    dataset_choice = os.environ.get('DATASET_CHOICE', 'potato_only')
    include_potato = dataset_choice == 'all_datasets'
    
    # Liste des sources de données
    data_sources = [source_path]
    if include_potato and os.path.exists('plantvillage'):
        data_sources.append('plantvillage')
        print("Inclusion des donnees de pommes de terre depuis plantvillage/")
    
    # Traiter chaque source de données
    for data_source in data_sources:
        print(f"Traitement de la source: {data_source}")
        
        # Parcourir toutes les classes dans le dossier source
        for class_name in os.listdir(data_source):
            class_path = os.path.join(data_source, class_name)
            
            if os.path.isdir(class_path):
                print(f"Traitement de la classe: {class_name}")
                
                # Créer les dossiers de classe dans train et test
                train_class_dir = train_dir / class_name
                test_class_dir = test_dir / class_name
                train_class_dir.mkdir(exist_ok=True)
                test_class_dir.mkdir(exist_ok=True)
                
                # Lister tous les fichiers d'images
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Mélanger les fichiers
                np.random.seed(seed)
                np.random.shuffle(image_files)
                
                # Diviser en train (80%) et test (20%)
                split_idx = int(len(image_files) * 0.8)
                train_files = image_files[:split_idx]
                test_files = image_files[split_idx:]
                
                print(f"  - {len(train_files)} images pour l'entrainement")
                print(f"  - {len(test_files)} images pour le test")
                
                # Copier les fichiers d'entraînement
                for file_name in train_files:
                    src_file = os.path.join(class_path, file_name)
                    dst_file = train_class_dir / file_name
                    shutil.copy2(src_file, dst_file)
                
                # Copier les fichiers de test
                for file_name in test_files:
                    src_file = os.path.join(class_path, file_name)
                    dst_file = test_class_dir / file_name
                    shutil.copy2(src_file, dst_file)
    
    print("Structure train/test creee avec succes!")

def split_train_validation(train_dataset, val_split=0.2, seed=42):
    """
    Divise le dataset d'entraînement en train et validation
    """
    print("Division du dataset d'entrainement en train/validation...")
    
    # Calculer les tailles sans charger en mémoire
    train_count = count_files_in_directory('training/train')
    total_batches = int(train_count / 32)  # BATCH_SIZE = 32
    val_batches = int(total_batches * val_split)
    
    # Split avec .take() et .skip()
    val_ds = train_dataset.take(val_batches)
    train_ds = train_dataset.skip(val_batches)
    
    return train_ds, val_ds

def visualize_sample_images(dataset, class_names, num_images=12):
    """
    Visualise quelques images annotées avec matplotlib
    """
    print("Visualisation d'images d'exemple...")
    
    plt.figure(figsize=(15, 12))
    
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            plt.subplot(3, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            # Obtenir le label
            label_idx = tf.argmax(labels[i]).numpy()
            label_name = class_names[label_idx]
            
            plt.title(f"Classe: {label_name}")
            plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Image sauvegardee: sample_images.png")

def create_preprocessing_layers(image_size):
    """
    Crée les couches de prétraitement
    """
    print("Creation des couches de pretraitement...")
    
    # Couche de redimensionnement et normalisation
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(image_size[0], image_size[1]),
        tf.keras.layers.Rescaling(1./255)  # Normalisation entre 0 et 1
    ])
    
    # Couche de data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])
    
    preprocessing_layers = {
        'resize_and_rescale': resize_and_rescale,
        'data_augmentation': data_augmentation
    }
    
    return preprocessing_layers

def prepare_pipeline(dataset, shuffle=True):
    """
    Prépare le pipeline optimisé avec cache, shuffle et prefetch
    """
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    # Cache pour éviter la relecture depuis le disque
    dataset = dataset.cache()
    
    # Prefetch pour accélérer l'entraînement en pipeline GPU/CPU
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

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

def save_preprocessing_layers(preprocessing_layers):
    """
    Sauvegarde les couches de prétraitement
    """
    print("Sauvegarde des couches de pretraitement...")
    
    # Créer le dossier models s'il n'existe pas
    Path('models').mkdir(exist_ok=True)
    
    # Sauvegarder les couches
    with open('models/preprocessing_layers.pkl', 'wb') as f:
        pickle.dump(preprocessing_layers, f)
    
    print("Couches de pretraitement sauvegardees dans models/preprocessing_layers.pkl")

if __name__ == "__main__":
    # Exécuter la préparation des données
    train_ds, val_ds, test_ds, class_names, preprocessing_layers = load_and_prepare_data()
    
    if train_ds is not None:
        print("\nDonnees pretes pour l'entrainement!")
        print(f"   Classes: {class_names}")
        print(f"   Taille des images: 256x256")
        print(f"   Batch size: 32")
        print(f"   Structure: training/train/ (entrainement), training/test/ (test)") 