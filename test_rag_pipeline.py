from app.rag_pipeline.processor import RAGProcessor

def test_rag_pipeline():
    # Initialiser le pipeline RAG
    rag = RAGProcessor()
    
    # Vérifier l'état du système
    system_info = rag.get_system_info()
    print("État du système:")
    print(f"- Documents: {system_info['document_count']}")
    print(f"- Chunks: {system_info['chunk_count']}")
    print(f"- Modèle chargé: {system_info['model_loaded']}")
    print(f"- Nom du modèle: {system_info['model_name']}")
    
    # Si le modèle n'est pas chargé, terminer le test
    if not system_info['model_loaded']:
        print("\nVeuillez exécuter download_model.py pour télécharger le modèle.")
        return
    
    # Créer un fichier texte de test
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
    
    # Convertir le contenu en bytes
    file_content = test_content.encode('utf-8')
    
    # Ajouter le document
    print("\nAjout d'un document...")
    result = rag.add_document(file_content, test_filename)
    
    if result["success"]:
        print(f"Document ajouté avec succès: {result['doc_id']}")
    else:
        print(f"Erreur: {result['message']}")
        return
    
    # Tester une requête
    question = "Quels sont les avantages du RAG?"
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
        print(f"Erreur: {result['message']}")

if __name__ == "__main__":
    test_rag_pipeline()