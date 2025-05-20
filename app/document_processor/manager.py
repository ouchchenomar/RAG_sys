import os
import json
from typing import List, Dict, Any
from .loader import DocumentLoader
from .chunker import DocumentChunker

class DocumentManager:
    """
    Classe pour gérer les documents et leurs chunks
    """
    
    def __init__(self, storage_dir: str = "./data", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialiser le gestionnaire de documents
        
        Args:
            storage_dir (str): Répertoire de stockage
            chunk_size (int): Taille des chunks en caractères
            chunk_overlap (int): Chevauchement entre les chunks en caractères
        """
        self.docs_dir = os.path.join(storage_dir, "documents")
        self.chunks_dir = os.path.join(storage_dir, "chunks")
        self.loader = DocumentLoader(self.docs_dir)
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        
        # Créer les répertoires nécessaires
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
    
    def process_document(self, file_content: bytes, filename: str) -> str:
        """
        Traiter un document: charger, découper et sauvegarder
        
        Args:
            file_content (bytes): Contenu du fichier
            filename (str): Nom du fichier
            
        Returns:
            str: ID du document
        """
        # Charger et sauvegarder le document
        doc_metadata = self.loader.save_file(file_content, filename)
        
        # Découper le document
        chunks = self.chunker.split_text(doc_metadata["content"], doc_metadata)
        
        # Sauvegarder les chunks
        chunks_path = os.path.join(self.chunks_dir, f"{doc_metadata['id']}.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        return doc_metadata["id"]
    
    def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Récupérer les chunks d'un document
        
        Args:
            doc_id (str): ID du document
            
        Returns:
            List[Dict[str, Any]]: Liste des chunks avec leurs métadonnées
        """
        chunks_path = os.path.join(self.chunks_dir, f"{doc_id}.json")
        
        if not os.path.exists(chunks_path):
            return []
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Récupérer tous les chunks de tous les documents
        
        Returns:
            List[Dict[str, Any]]: Liste de tous les chunks avec leurs métadonnées
        """
        all_chunks = []
        
        for filename in os.listdir(self.chunks_dir):
            if filename.endswith('.json'):
                chunks_path = os.path.join(self.chunks_dir, filename)
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
        
        return all_chunks