import logging
from typing import Dict, Any, List, Union
from app.document_processor.manager import DocumentManager
from app.vector_store.manager import VectorStoreManager
from app.llm.model_manager import LLMManager

# Configurer le logging
logger = logging.getLogger(__name__)

class RAGProcessor:
    """
    Classe pour le pipeline RAG complet
    """
    def __init__(self, use_sentence_transformers: bool = True):
        """
        Initialiser le pipeline RAG
        Args:
            use_sentence_transformers (bool): Utiliser SentenceTransformers au lieu de TF-IDF
        """
        logger.info("Initialisation du RAGProcessor")
        self.doc_manager = DocumentManager()
        self.vector_store = VectorStoreManager(use_sentence_transformers=use_sentence_transformers)
        self.llm_manager = LLMManager()
        
        # Charger l'index vectoriel
        self.vector_store.load_index()
        
        # Essayer de charger le modèle LLM
        if not self.llm_manager.load_model():
            # Essayer des modèles alternatifs si le modèle par défaut ne se charge pas
            alternative_models = [
                "tinyllama", "llama-2-7b-chat", "mistral-7b-instruct-v0.1", 
                "falcon-7b-instruct", "mixtral-8x7b-instruct-v0.1"
            ]
            for model in alternative_models:
                logger.info(f"Tentative de chargement du modèle alternatif: {model}")
                if self.llm_manager.load_model(model):
                    break
        
        self.model_loaded = self.llm_manager.model is not None
        logger.info(f"Modèle LLM chargé: {self.model_loaded}")

    def add_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Ajouter un document à la base de connaissances
        Args:
            file_content (bytes): Contenu du fichier
            filename (str): Nom du fichier
        Returns:
            Dict[str, Any]: Résultat de l'opération
        """
        try:
            logger.info(f"Ajout du document: {filename}")
            # Traiter le document
            doc_id = self.doc_manager.process_document(file_content, filename)
            
            # Récupérer tous les chunks
            chunks = self.doc_manager.get_all_chunks()
            
            # Recréer l'index vectoriel
            self.vector_store.create_index(chunks)
            
            logger.info(f"Document ajouté avec succès: {filename}, ID: {doc_id}")
            return {
                "success": True,
                "doc_id": doc_id,
                "message": f"Document '{filename}' ajouté avec succès"
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du document {filename}: {e}")
            return {
                "success": False,
                "message": f"Erreur lors de l'ajout du document: {e}"
            }

    def query(self, question: str, top_k: int = 5, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Interroger le système RAG avec paramètres améliorés
        Args:
            question (str): Question de l'utilisateur
            top_k (int): Nombre de chunks à récupérer
            max_tokens (int): Nombre maximum de tokens à générer
            temperature (float): Température pour le sampling
        Returns:
            Dict[str, Any]: Résultat de la requête
        """
        # Vérifier si le modèle est chargé
        if not self.model_loaded:
            logger.warning("Modèle LLM non chargé")
            return {
                "success": False,
                "message": "Modèle LLM non chargé. Veuillez charger un modèle d'abord."
            }
        
        try:
            logger.info(f"Traitement de la requête: {question}")
            # Récupérer tous les chunks
            chunks = self.doc_manager.get_all_chunks()
            
            # Si aucun document n'est disponible
            if not chunks:
                logger.warning("Aucun document disponible")
                return {
                    "success": False,
                    "message": "Aucun document disponible dans la base de connaissances."
                }
            
            # Rechercher les chunks pertinents
            relevant_chunks = self.vector_store.search(question, top_k)
            
            # Si aucun chunk pertinent n'est trouvé
            if not relevant_chunks:
                logger.warning("Aucun chunk pertinent trouvé")
                return {
                    "success": True,
                    "answer": "Je n'ai pas trouvé d'information pertinente pour répondre à votre question dans les documents disponibles.",
                    "sources": []
                }
            
            # Créer le prompt avec contexte
            prompt = self.llm_manager.create_prompt(question, relevant_chunks)
            
            # Générer une réponse avec paramètres améliorés
            answer = self.llm_manager.generate(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95
            )
            
            # Préparer les sources
            sources = [{
                "doc_id": chunk["metadata"]["doc_id"],
                "filename": chunk["metadata"]["filename"],
                "chunk_id": chunk["metadata"]["chunk_id"],
                "score": chunk.get("score", 0)
            } for chunk in relevant_chunks]
            
            logger.info(f"Réponse générée avec succès, longueur: {len(answer)}")
            return {
                "success": True,
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Erreur lors de la requête: {e}")
            return {
                "success": False,
                "message": f"Erreur lors de la requête: {e}"
            }

    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtenir des informations sur l'état du système
        Returns:
            Dict[str, Any]: Informations sur le système
        """
        try:
            # Récupérer tous les chunks
            chunks = self.doc_manager.get_all_chunks()
            
            # Compter le nombre de documents uniques
            doc_ids = set()
            for chunk in chunks:
                doc_ids.add(chunk["metadata"]["doc_id"])
            
            logger.info(f"Info système: {len(doc_ids)} documents, {len(chunks)} chunks")
            return {
                "success": True,
                "document_count": len(doc_ids),
                "chunk_count": len(chunks),
                "model_loaded": self.model_loaded,
                "model_name": self.llm_manager.model_name if self.model_loaded else None,
                "embeddings_type": self.vector_store.embeddings_type
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations système: {e}")
            return {
                "success": False,
                "document_count": 0,
                "chunk_count": 0,
                "model_loaded": False,
                "model_name": None,
                "error": str(e)
            }