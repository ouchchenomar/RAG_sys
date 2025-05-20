import subprocess
import sys
import os

def install_dependencies():
    """Installer les dépendances nécessaires pour le système RAG"""
    print("Installation des dépendances pour le système RAG...")
    
    # Dépendances de base
    base_dependencies = [
        "fastapi==0.104.1",
        "uvicorn==0.23.2",
        "python-multipart==0.0.6",
        "scikit-learn==1.3.2",
        "nltk==3.8.1",
        "pypdf2==3.0.1",
        "numpy==1.26.0",
        "ctransformers==0.2.27",
        "jinja2==3.1.2",
        "requests==2.31.0"
    ]
    
    # Dépendances optionnelles
    optional_dependencies = [
        "sentence-transformers==2.2.2"
    ]
    
    # Installer les dépendances de base
    print("\nInstallation des dépendances de base...")
    for dep in base_dependencies:
        print(f"Installation de {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep])
    
    # Demander à l'utilisateur s'il souhaite installer les dépendances optionnelles
    install_optional = input("\nSouhaitez-vous installer les dépendances optionnelles pour de meilleures performances ? (y/n): ")
    
    if install_optional.lower() == 'y':
        print("\nInstallation des dépendances optionnelles...")
        for dep in optional_dependencies:
            print(f"Installation de {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep])
            except Exception as e:
                print(f"Erreur lors de l'installation de {dep}: {e}")
                print("L'application fonctionnera avec des fonctionnalités réduites.")
    
    print("\nInstallation des dépendances terminée.")
    
    # Télécharger les ressources NLTK nécessaires
    print("\nTéléchargement des ressources NLTK...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Ressources NLTK téléchargées avec succès.")
    except Exception as e:
        print(f"Erreur lors du téléchargement des ressources NLTK: {e}")
    
    print("\nConfiguration terminée. Vous pouvez maintenant démarrer le système RAG avec 'python run_api.py'")

if __name__ == "__main__":
    install_dependencies()