import zipfile
import os
import shutil
from pathlib import Path

def extract_and_organize_data(extract_all=False):
    """
    Extrait l'archive et organise les données
    """
    print("Extraction et organisation des données...")
    
    # Créer la structure du projet
    project_dirs = [
        'plantvillage',
        'models',
        'API',
        'frontend'
    ]
    
    for dir_name in project_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Dossier créé: {dir_name}")
    
    # Extraire l'archive
    zip_path = 'data/archive.zip'
    extract_path = 'data/'
    
    if os.path.exists(zip_path):
        print(f"Extraction de {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Archive extraite avec succès!")
        
        if extract_all:
            # Extraire tous les datasets
            print("Extraction de tous les datasets disponibles...")
            
            # Chercher tous les dossiers de données
            all_data_dirs = []
            for root, dirs, files in os.walk(extract_path):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    # Vérifier si le dossier contient des images
                    image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        all_data_dirs.append((dir_path, dir_name))
            
            print(f"Datasets trouves: {len(all_data_dirs)}")
            
            # Garder tous les datasets dans data/ pour l'option "all_datasets"
            for dir_path, dir_name in all_data_dirs:
                print(f"  {dir_name}: {len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])} images")
            
            # Chercher spécifiquement les dossiers de pommes de terre pour plantvillage/
            potato_dirs = []
            for root, dirs, files in os.walk(extract_path):
                for dir_name in dirs:
                    if 'potato' in dir_name.lower():
                        potato_dirs.append(os.path.join(root, dir_name))
            
            print(f"\nDossiers de pommes de terre trouves: {len(potato_dirs)}")
            
            # Déplacer les dossiers de pommes de terre vers plantvillage/
            for potato_dir in potato_dirs:
                dir_name = os.path.basename(potato_dir)
                dest_path = os.path.join('plantvillage', dir_name)
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.move(potato_dir, dest_path)
                print(f"Deplace: {dir_name} -> plantvillage/")
            
        else:
            # Extraire uniquement les pommes de terre (comportement par défaut)
            print("Extraction des donnees de pommes de terre uniquement...")
            
            # Chercher les dossiers de pommes de terre
            potato_dirs = []
            for root, dirs, files in os.walk(extract_path):
                for dir_name in dirs:
                    if 'potato' in dir_name.lower():
                        potato_dirs.append(os.path.join(root, dir_name))
            
            print(f"Dossiers de pommes de terre trouves: {len(potato_dirs)}")
            
            # Déplacer les dossiers de pommes de terre vers plantvillage/
            for potato_dir in potato_dirs:
                dir_name = os.path.basename(potato_dir)
                dest_path = os.path.join('plantvillage', dir_name)
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.move(potato_dir, dest_path)
                print(f"Deplace: {dir_name} -> plantvillage/")
        
        # Lister le contenu final
        print("\nContenu du dossier plantvillage:")
        for item in os.listdir('plantvillage'):
            item_path = os.path.join('plantvillage', item)
            if os.path.isdir(item_path):
                num_files = len([f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {item}: {num_files} images")
        
        if extract_all:
            print("\nTous les datasets sont disponibles dans le dossier data/")
            print("Ils seront utilises lors de l'entrainement avec l'option 'all_datasets'")
        
    else:
        print(f"Fichier {zip_path} non trouve!")

if __name__ == "__main__":
    # Lire la variable d'environnement pour déterminer si extraire tous les datasets
    import os
    extract_all = os.environ.get('EXTRACT_ALL', '0') == '1'
    extract_and_organize_data(extract_all) 