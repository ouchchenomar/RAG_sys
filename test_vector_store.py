from app.document_processor.manager import DocumentManager
from app.vector_store.manager import VectorStoreManager

def test_vector_store():
    # Initialiser les gestionnaires
    doc_manager = DocumentManager()
    vector_store = VectorStoreManager()
    
    # Créer un fichier texte de test
    test_content = "Ceci est un document sur l'intelligence artificielle.\n" * 20 + \
                   "Le machine learning permet d'améliorer les systèmes.\n" * 20 + \
                   "Les modèles de langage sont très utiles aujourd'hui.\n" * 20
    test_filename = "test_document_ia.txt"
    
    # Convertir le contenu en bytes comme s'il était téléchargé
    file_content = test_content.encode('utf-8')
    
    # Traiter le document
    doc_id = doc_manager.process_document(file_content, test_filename)
    print(f"Document traité avec l'ID: {doc_id}")
    
    # Récupérer les chunks
    chunks = doc_manager.get_all_chunks()
    print(f"Nombre total de chunks: {len(chunks)}")
    
    # Créer l'index vectoriel
    vector_store.create_index(chunks)
    print("Index vectoriel créé")
    
    # Tester la recherche
    query = "intelligence artificielle"
    results = vector_store.search(query, chunks, top_k=2)
    
    print(f"\nRecherche pour '{query}':")
    for i, result in enumerate(results):
        print(f"\nRésultat {i+1} (score: {result['score']:.4f}):")
        print(result["text"][:200] + "...")

if __name__ == "__main__":
    test_vector_store()