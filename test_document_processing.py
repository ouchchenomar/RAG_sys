import os
from app.document_processor.manager import DocumentManager

def test_document_processing():
    # Initialiser le gestionnaire de documents
    doc_manager = DocumentManager()
    
    # Créer un fichier texte de test
    test_content = "Ceci est un document de test pour notre système RAG.\n" * 100
    test_filename = "test_document.txt"
    
    # Convertir le contenu en bytes comme s'il était téléchargé
    file_content = test_content.encode('utf-8')
    
    # Traiter le document
    doc_id = doc_manager.process_document(file_content, test_filename)
    print(f"Document traité avec l'ID: {doc_id}")
    
    # Récupérer les chunks
    chunks = doc_manager.get_chunks(doc_id)
    print(f"Nombre de chunks créés: {len(chunks)}")
    
    # Afficher le premier chunk
    if chunks:
        print("\nPremier chunk:")
        print(chunks[0]["text"][:200] + "...")
        print("\nMétadonnées du premier chunk:")
        print(chunks[0]["metadata"])

if __name__ == "__main__":
    test_document_processing()