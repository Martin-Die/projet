#!/usr/bin/env python3
"""
Script principal pour l'application de diagnostic des maladies de pommes de terre
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """
    Affiche la bannière du projet
    """
    print("=" * 80)
    print("APPLICATION DE DIAGNOSTIC DES MALADIES DE POMMES DE TERRE")
    print("=" * 80)
    print("Deep Learning - FastAPI - React.js")
    print("=" * 80)

def check_dependencies():
    """
    Vérifie que les dépendances sont installées
    """
    print("Verification des dependances...")
    
    try:
        import tensorflow as tf
        import fastapi
        import numpy as np
        import matplotlib
        import PIL
        print("Toutes les dependances Python sont installees")
        return True
    except ImportError as e:
        print(f"Dependance manquante: {e}")
        print("Installez les dependances avec: pip install -r requirements.txt")
        return False

def run_extract_data(extract_all=False):
    """
    Exécute l'extraction des données
    """
    print("\nETAPE 1: Extraction et organisation des donnees")
    print("-" * 50)
    
    if not os.path.exists('extract_data.py'):
        print("Fichier extract_data.py non trouve")
        return False
    
    try:
        print("Extraction des donnees en cours...")
        
        # Passer le paramètre extract_all via variable d'environnement
        env = os.environ.copy()
        env['EXTRACT_ALL'] = '1' if extract_all else '0'
        
        result = subprocess.run([sys.executable, 'extract_data.py'], env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"Erreur lors de l'extraction: {e}")
        return False

def run_data_preparation(dataset_choice=None):
    """
    Exécute la préparation des données
    """
    print("\nETAPE 2: Preparation des donnees")
    print("-" * 50)
    
    if not os.path.exists('data_preparation.py'):
        print("Fichier data_preparation.py non trouve")
        return False
    
    try:
        print("Preparation des donnees en cours...")
        
        # Passer le paramètre dataset_choice via variable d'environnement
        env = os.environ.copy()
        if dataset_choice:
            env['DATASET_CHOICE'] = dataset_choice
        
        result = subprocess.run([sys.executable, 'data_preparation.py'], env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"Erreur lors de la preparation: {e}")
        return False

def run_model_training(dataset_choice=None):
    """
    Exécute l'entraînement du modèle
    """
    print("\nETAPE 3: Entrainement du modele")
    print("-" * 50)
    
    if not os.path.exists('train_model.py'):
        print("Fichier train_model.py non trouve")
        return False
    
    # Si aucun choix n'est fourni, demander à l'utilisateur
    if dataset_choice is None:
        while True:
            show_dataset_choice()
            try:
                choice = input("\nVotre choix (1-3): ").strip()
                
                if choice == '1':
                    dataset_choice = 'potato_only'
                    print("Modele sera entraine uniquement avec les donnees de pommes de terre")
                    break
                elif choice == '2':
                    dataset_choice = 'all_datasets'
                    print("Modele sera entraine avec tous les datasets disponibles")
                    break
                elif choice == '3':
                    print("Retour au menu principal")
                    return False
                else:
                    print("Choix invalide. Veuillez choisir un nombre entre 1 et 3.")
            except KeyboardInterrupt:
                print("\nRetour au menu principal")
                return False
    
    try:
        print("Lancement de l'entrainement du modele...")
        print("Vous verrez la progression en temps reel:")
        print("  - Chargement des donnees")
        print("  - Construction du modele")
        print("  - Compilation")
        print("  - Entrainement (epochs avec accuracy/loss)")
        print("  - Evaluation")
        print("  - Sauvegarde")
        print("-" * 50)
        
        # Exécuter avec le choix de dataset en variable d'environnement
        env = os.environ.copy()
        env['DATASET_CHOICE'] = dataset_choice
        
        # Exécuter sans capture pour voir la progression en temps réel
        result = subprocess.run([sys.executable, 'train_model.py'], env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"Erreur lors de l'entrainement: {e}")
        return False

def run_api_server():
    """
    Lance le serveur API
    """
    print("\nETAPE 4: Lancement du serveur API")
    print("-" * 50)
    
    if not os.path.exists('API/main.py'):
        print("Fichier API/main.py non trouve")
        return False
    
    print("Demarrage du serveur FastAPI...")
    print("L'API sera accessible sur: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("Test de sante: http://localhost:8000/ping")
    print("\nAppuyez sur Ctrl+C pour arreter le serveur")
    
    try:
        # Changer vers le dossier API
        os.chdir('API')
        subprocess.run([sys.executable, '-m', 'uvicorn', 'main:app', 
                       '--host', '0.0.0.0', '--port', '8000', '--reload'])
    except KeyboardInterrupt:
        print("\nServeur arrete")
        os.chdir('..')
    except Exception as e:
        print(f"Erreur lors du lancement de l'API: {e}")
        os.chdir('..')
        return False
    
    return True

def test_api():
    """
    Teste l'API avec une requête simple
    """
    print("\nTest de l'API...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/ping", timeout=5)
        if response.status_code == 200:
            print("API fonctionne correctement")
            return True
        else:
            print(f"API retourne le code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Impossible de contacter l'API: {e}")
        return False

def setup_frontend():
    """
    Configure le frontend React
    """
    print("\nETAPE 5: Configuration du frontend")
    print("-" * 50)
    
    if not os.path.exists('frontend'):
        print("Creation du dossier frontend...")
        os.makedirs('frontend', exist_ok=True)
    
    # Créer un fichier .env pour le frontend
    env_content = """# Configuration de l'API
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=30000
"""
    
    with open('frontend/.env', 'w') as f:
        f.write(env_content)
    
    print("Configuration frontend creee")
    print("Pour installer le frontend React:")
    print("   1. cd frontend")
    print("   2. npm install")
    print("   3. npm start")
    
    return True

def show_menu():
    """
    Affiche le menu principal
    """
    print("\nMENU PRINCIPAL")
    print("=" * 50)
    print("1. Executer tout le pipeline (extraction -> entrainement)")
    print("2. Extraction des donnees")
    print("3. Preparation des donnees")
    print("4. Entrainement du modele")
    print("5. Lancer l'API FastAPI")
    print("6. Tester l'API")
    print("7. Configurer le frontend")
    print("8. Afficher la documentation")
    print("9. Quitter")
    print("=" * 50)

def show_dataset_choice():
    """
    Affiche le menu de choix du dataset
    """
    print("\nCHOIX DU DATASET POUR L'ENTRAINEMENT")
    print("=" * 50)
    print("1. Entrainer uniquement avec les donnees de pommes de terre (potato)")
    print("2. Entrainer avec tous les datasets disponibles")
    print("3. Retour au menu principal")
    print("=" * 50)

def show_documentation():
    """
    Affiche la documentation du projet
    """
    print("\nDOCUMENTATION DU PROJET")
    print("=" * 80)
    print("APPLICATION DE DIAGNOSTIC DES MALADIES DE POMMES DE TERRE")
    print("=" * 80)
    
    print("\nOBJECTIF:")
    print("   Construire une application mobile complete permettant aux producteurs")
    print("   de pommes de terre de photographier une feuille et recevoir un")
    print("   diagnostic automatique grace a un modele Deep Learning.")
    
    print("\nARCHITECTURE:")
    print("   1. Extraction et organisation des donnees (PlantVillage dataset)")
    print("   2. Preparation des donnees (split, augmentation, pipeline)")
    print("   3. Entrainement du modele CNN (TensorFlow/Keras)")
    print("   4. API REST (FastAPI)")
    print("   5. Interface utilisateur (React.js)")
    
    print("\nSTRUCTURE DU PROJET:")
    print("   ├── data/                    # Donnees brutes")
    print("   ├── plantvillage/            # Images organisees (pommes de terre)")
    print("   ├── training/                # Structure train/test")
    print("   │   ├── train/               # Donnees d'entrainement")
    print("   │   └── test/                # Donnees de test")
    print("   ├── models/                  # Modeles entraines")
    print("   ├── API/                     # Serveur FastAPI")
    print("   ├── frontend/                # Application React")
    print("   ├── extract_data.py          # Extraction des donnees")
    print("   ├── data_preparation.py      # Preparation des donnees")
    print("   ├── train_model.py           # Entrainement du modele")
    print("   └── main.py                  # Script principal")
    
    print("\nUTILISATION:")
    print("   1. python main.py")
    print("   2. Choisir l'option 1 pour executer tout le pipeline")
    print("   3. Lancer l'API avec l'option 5")
    print("   4. Tester avec l'option 6")
    print("   5. Configurer le frontend avec l'option 7")
    
    print("\nLIENS UTILES:")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - API Health Check: http://localhost:8000/ping")
    print("   - Frontend: http://localhost:3000 (apres configuration)")
    
    print("\n" + "=" * 80)

def main():
    """
    Fonction principale
    """
    print_banner()
    
    # Vérifier les dépendances
    if not check_dependencies():
        print("\nPour installer les dependances:")
        print("   pip install -r requirements.txt")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("\nVotre choix (1-9): ").strip()
            
            if choice == '1':
                print("\nEXECUTION COMPLETE DU PIPELINE")
                print("=" * 50)
                
                # Demander le choix du dataset pour l'extraction et l'entraînement
                while True:
                    show_dataset_choice()
                    try:
                        dataset_choice = input("\nVotre choix pour le dataset (1-3): ").strip()
                        
                        if dataset_choice == '1':
                            extract_all = False
                            dataset_choice = 'potato_only'
                            print("Pipeline utilisera uniquement les donnees de pommes de terre")
                            break
                        elif dataset_choice == '2':
                            extract_all = True
                            dataset_choice = 'all_datasets'
                            print("Pipeline utilisera tous les datasets disponibles")
                            break
                        elif dataset_choice == '3':
                            print("Retour au menu principal")
                            continue
                        else:
                            print("Choix invalide. Veuillez choisir un nombre entre 1 et 3.")
                    except KeyboardInterrupt:
                        print("\nRetour au menu principal")
                        continue
                
                steps = [
                    ("Extraction des donnees", lambda: run_extract_data(extract_all)),
                    ("Preparation des donnees", lambda: run_data_preparation(dataset_choice)),
                    ("Entrainement du modele", lambda: run_model_training(dataset_choice)),
                    ("Configuration frontend", setup_frontend)
                ]
                
                for step_name, step_func in steps:
                    print(f"\n{step_name}...")
                    if not step_func():
                        print(f"Echec a l'etape: {step_name}")
                        break
                    print(f"{step_name} termine")
                
                print("\nPipeline termine!")
                print("Vous pouvez maintenant lancer l'API avec l'option 5")
                
            elif choice == '2':
                # Demander le choix du dataset pour l'extraction
                while True:
                    print("\nCHOIX DU DATASET POUR L'EXTRACTION")
                    print("=" * 50)
                    print("1. Extraire uniquement les donnees de pommes de terre (potato)")
                    print("2. Extraire tous les datasets disponibles")
                    print("3. Retour au menu principal")
                    print("=" * 50)
                    
                    try:
                        extract_choice = input("\nVotre choix (1-3): ").strip()
                        
                        if extract_choice == '1':
                            extract_all = False
                            print("Extraction des donnees de pommes de terre uniquement")
                            run_extract_data(extract_all)
                            break
                        elif extract_choice == '2':
                            extract_all = True
                            print("Extraction de tous les datasets disponibles")
                            run_extract_data(extract_all)
                            break
                        elif extract_choice == '3':
                            print("Retour au menu principal")
                            break
                        else:
                            print("Choix invalide. Veuillez choisir un nombre entre 1 et 3.")
                    except KeyboardInterrupt:
                        print("\nRetour au menu principal")
                        break
                
            elif choice == '3':
                # Demander le choix du dataset pour la préparation
                while True:
                    print("\nCHOIX DU DATASET POUR LA PREPARATION")
                    print("=" * 50)
                    print("1. Preparer uniquement les donnees de pommes de terre (potato)")
                    print("2. Preparer tous les datasets disponibles")
                    print("3. Retour au menu principal")
                    print("=" * 50)
                    
                    try:
                        prep_choice = input("\nVotre choix (1-3): ").strip()
                        
                        if prep_choice == '1':
                            dataset_choice = 'potato_only'
                            print("Preparation des donnees de pommes de terre uniquement")
                            run_data_preparation(dataset_choice)
                            break
                        elif prep_choice == '2':
                            dataset_choice = 'all_datasets'
                            print("Preparation de tous les datasets disponibles")
                            run_data_preparation(dataset_choice)
                            break
                        elif prep_choice == '3':
                            print("Retour au menu principal")
                            break
                        else:
                            print("Choix invalide. Veuillez choisir un nombre entre 1 et 3.")
                    except KeyboardInterrupt:
                        print("\nRetour au menu principal")
                        break
                
            elif choice == '4':
                run_model_training()
                
            elif choice == '5':
                run_api_server()
                
            elif choice == '6':
                test_api()
                
            elif choice == '7':
                setup_frontend()
                
            elif choice == '8':
                show_documentation()
                
            elif choice == '9':
                print("\n👋 Au revoir!")
                break
                
            else:
                print("Choix invalide. Veuillez choisir un nombre entre 1 et 9.")
                
        except KeyboardInterrupt:
            print("\n\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")

if __name__ == "__main__":
    main()

