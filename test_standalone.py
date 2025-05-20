import os
import sys
import time

# Ajouter le répertoire du projet au chemin d'importation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.document_processor.manager import DocumentManager
from app.vector_store.manager import VectorStoreManager
from app.rag_pipeline.processor import RAGProcessor

def test_standalone():
    """
    Tester le système RAG sans passer par l'API
    """
    print("=== Test du système RAG en mode standalone ===")
    
    # Initialiser le RAG processor directement
    print("\nInitialisation du processeur RAG...")
    rag = RAGProcessor()
    
    # Afficher les informations du système
    system_info = rag.get_system_info()
    print("\nInformations sur le système:")
    print(f"- Documents: {system_info.get('document_count', 0)}")
    print(f"- Chunks: {system_info.get('chunk_count', 0)}")
    print(f"- Modèle chargé: {system_info.get('model_loaded', False)}")
    if 'model_info' in system_info and system_info['model_info']:
        print(f"- Informations sur le modèle: {system_info['model_info']}")
    
    # Créer un fichier test
    test_content = """Le système RAG (Retrieval-Augmented Generation) est une technologie 
    qui combine la recherche d'informations avec des modèles de langage. 
    Il permet d'améliorer les réponses des LLM en leur fournissant du contexte pertinent.
    
    Les composants principaux d'un système RAG sont:
    1. Un système de gestion de documents
    2. Un système d'indexation vectorielle
    3. Un modèle de langage (LLM)
    4. Un pipeline d'intégration
    
    Les avantages du RAG incluent une meilleure précision des réponses, 
    une réduction des hallucinations et la possibilité d'accéder à des connaissances spécifiques."""
    
    test_filename = "test_document_rag.txt"
    
    # Sauvegarder temporairement le fichier
    print(f"\nCréation du fichier de test: {test_filename}")
    with open(test_filename, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    # Ouvrir le fichier et le convertir en bytes
    with open(test_filename, "rb") as f:
        file_content = f.read()
    
    # Ajouter le document
    print("\nAjout du document au système...")
    result = rag.add_document(file_content, test_filename)
    
    if result["success"]:
        print(f"Document ajouté avec succès. ID: {result['doc_id']}")
    else:
        print(f"Erreur lors de l'ajout du document: {result['message']}")
        return
    
    # Test d'une requête si le modèle est chargé
    if rag.model_loaded:
        question = "Quels sont les composants d'un système RAG?"
        print(f"\nRequête: '{question}'")
        
        result = rag.query(question)
        
        if result["success"]:
            print("\nRéponse:")
            print("-" * 50)
            print(result["answer"])
            print("-" * 50)
            
            print("\nSources:")
            for i, source in enumerate(result["sources"]):
                print(f"{i+1}. Fichier: {source['filename']}, Score: {source['score']:.4f}")
        else:
            print(f"Erreur lors de la requête: {result['message']}")
    else:
        print("\nLe modèle n'est pas chargé, impossible d'effectuer une requête.")
        print("Vous devez télécharger un modèle LLM pour effectuer des requêtes.")
        print("Exécutez download_model.py pour télécharger un modèle approprié.")
    
    # Supprimer le fichier de test
    try:
        os.remove(test_filename)
        print(f"\nFichier de test {test_filename} supprimé.")
    except:
        print(f"\nImpossible de supprimer le fichier de test {test_filename}.")
    
    print("\nTest complet terminé!")

if __name__ == "__main__":
    test_standalone()