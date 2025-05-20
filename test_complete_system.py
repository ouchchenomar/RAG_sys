import requests
import json
import time
import os

# URL de l'API
BASE_URL = "http://127.0.0.1:8000/api"

def test_system_info():
    """Tester l'endpoint d'information sur le système"""
    response = requests.get(f"{BASE_URL}/system/")
    print("Informations sur le système:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_add_document(file_path):
    """Tester l'ajout d'un document"""
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'text/plain')}
        response = requests.post(f"{BASE_URL}/documents/", files=files)
    
    print("\nRésultat de l'ajout du document:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_query(question):
    """Tester une requête"""
    data = {
        "question": question,
        "top_k": 3,
        "max_tokens": 256
    }
    response = requests.post(f"{BASE_URL}/query/", json=data)
    
    print("\nRésultat de la requête:")
    result = response.json()
    
    print(f"Question: {question}")
    print("Réponse:")
    print("-" * 50)
    print(result.get("answer", "Pas de réponse"))
    print("-" * 50)
    
    if "sources" in result and result["sources"]:
        print("\nSources:")
        for i, source in enumerate(result["sources"]):
            print(f"{i+1}. Fichier: {source['filename']}, Score: {source['score']:.4f}")
    
    return result

if __name__ == "__main__":
    # Vérifier l'état du système
    system_info = test_system_info()
    
    # Créer un fichier de test
    test_file = "test_file.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""Le système RAG (Retrieval-Augmented Generation) est une technologie 
        qui combine la recherche d'informations avec des modèles de langage. 
        Il permet d'améliorer les réponses des LLM en leur fournissant du contexte pertinent.
        
        Les composants principaux d'un système RAG sont:
        1. Un système de gestion de documents
        2. Un système d'indexation vectorielle
        3. Un modèle de langage (LLM)
        4. Un pipeline d'intégration
        
        Les avantages du RAG incluent une meilleure précision des réponses, 
        une réduction des hallucinations et la possibilité d'accéder à des connaissances spécifiques.""")
    
    # Ajouter le document
    add_result = test_add_document(test_file)
    
    # Attendre un peu pour s'assurer que le document est indexé
    time.sleep(2)
    
    # Tester une requête
    query_result = test_query("Quels sont les composants d'un système RAG?")
    
    # Tester une autre requête
    query_result = test_query("Pourquoi utiliser un système RAG?")
    
    # Supprimer le fichier de test
    os.remove(test_file)
    
    print("\nTest complet terminé!")