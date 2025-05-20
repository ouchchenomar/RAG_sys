from typing import Dict, List, Optional
import logging
from ..vector_store.manager import VectorStoreManager
from ..llm.manager import LLMManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Pipeline RAG pour le traitement des données"""
    
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.llm = LLMManager()
        
    def add_document(self, content: str, metadata: Dict) -> Dict:
        """
        Ajoute un document au système RAG
        
        Args:
            content: Contenu du document
            metadata: Métadonnées du document
            
        Returns:
            Dict: Résultat de l'opération
        """
        try:
            # Ajouter le document au vector store
            self.vector_store.add_document(content, metadata)
            return {"success": True, "message": "Document ajouté avec succès"}
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du document: {e}")
            return {"success": False, "message": str(e)}
    
    def query(self, query: str, top_k: int = 3) -> Dict:
        """
        Interroge le système RAG
        
        Args:
            query: Question à poser
            top_k: Nombre de documents pertinents à récupérer
            
        Returns:
            Dict: Réponse générée
        """
        try:
            # Récupérer les documents pertinents
            relevant_docs = self.vector_store.search(query, top_k)
            
            # Générer la réponse
            response = self.llm.generate_response(query, relevant_docs)
            
            return {
                "success": True,
                "answer": response,
                "sources": relevant_docs
            }
        except Exception as e:
            logger.error(f"Erreur lors de la requête: {e}")
            return {"success": False, "message": str(e)} 