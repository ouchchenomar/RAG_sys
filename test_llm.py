from app.document_processor.manager import DocumentManager
from app.vector_store.manager import VectorStoreManager
from app.llm.model_manager import LLMManager

def test_llm():
    # Initialiser les gestionnaires
    doc_manager = DocumentManager()
    vector_store = VectorStoreManager()
    llm_manager = LLMManager()
    
    # Vérifier si le modèle est disponible
    if not llm_manager.load_model():
        print("Modèle non disponible. Veuillez exécuter download_model.py d'abord.")
        return
    
    print("Modèle chargé avec succès")
    
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
    
    # Traiter le document
    doc_id = doc_manager.process_document(file_content, test_filename)
    print(f"Document traité avec l'ID: {doc_id}")
    
    # Récupérer les chunks
    chunks = doc_manager.get_all_chunks()
    
    # Créer l'index vectoriel
    vector_store.create_index(chunks)
    
    # Tester une requête
    query = "Quels sont les avantages du RAG?"
    print(f"\nRecherche pour: '{query}'")
    
    # Récupérer les chunks pertinents
    relevant_chunks = vector_store.search(query, chunks, top_k=2)
    
    # Créer le prompt avec contexte
    prompt = llm_manager.create_prompt(query, relevant_chunks)
    
    print("\nPrompt créé:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    
    # Générer une réponse
    print("\nGénération de la réponse...")
    response = llm_manager.generate(prompt)
    
    print("\nRéponse générée:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    test_llm()