# app/vector_store/retriever.py
import logging
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Configurer le logging
logger = logging.getLogger(__name__)

class Retriever:
    """
    Classe pour rechercher des documents par similarité
    """
    def __init__(self, embeddings):
        """
        Initialiser le retriever
        Args:
            embeddings: Gestionnaire d'embeddings (TFIDFEmbeddings ou SentenceTransformerEmbeddings)
        """
        self.embeddings = embeddings
        logger.info(f"Retriever initialisé avec {type(embeddings).__name__}")

    def search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5, threshold: float = -0.1) -> List[Dict[str, Any]]:
        """
        Rechercher les chunks les plus pertinents pour une requête
        Args:
            query (str): Requête de recherche
            chunks (List[Dict[str, Any]]): Liste des chunks
            top_k (int): Nombre de résultats à retourner
            threshold (float): Seuil de similarité minimum (peut être négatif pour être plus inclusif)
        Returns:
            List[Dict[str, Any]]: Liste des chunks les plus pertinents avec leurs scores
        """
        # Vérifier que les chunks ont le bon format
        formatted_chunks = []
        for chunk in chunks:
            # S'assurer que chaque chunk a un champ 'text'
            if 'text' not in chunk and 'content' in chunk:
                chunk_copy = chunk.copy()
                chunk_copy['text'] = chunk_copy['content']
                formatted_chunks.append(chunk_copy)
            else:
                formatted_chunks.append(chunk)
        
        chunks = formatted_chunks
        
        # Vérifier si l'index est vide
        if not hasattr(self.embeddings, "doc_vectors") or self.embeddings.doc_vectors is None:
            logger.warning("doc_vectors non disponible, impossible de rechercher")
            return []
        
        # Vérifier si chunk_ids est disponible
        if not hasattr(self.embeddings, "chunk_ids") or not self.embeddings.chunk_ids:
            logger.warning("chunk_ids non disponible, impossible de rechercher")
            return []
            
        # Transformer la requête en vecteur
        try:
            query_vector = self.embeddings.transform(query)
        except Exception as e:
            logger.error(f"Erreur lors de la transformation de la requête: {e}")
            logger.error(f"Type d'erreur: {type(e)}")
            return []
        
        # Calculer la similarité cosinus entre la requête et tous les documents
        try:
            similarities = cosine_similarity(query_vector, self.embeddings.doc_vectors).flatten()
            logger.info(f"Similarités calculées, forme: {similarities.shape}")
            logger.info(f"Min/Max similarité: {np.min(similarities):.4f}/{np.max(similarities):.4f}")
        except Exception as e:
            logger.error(f"Erreur lors du calcul des similarités: {e}")
            logger.error(f"Type d'erreur: {type(e)}")
            return []
        
        # Filtrer par seuil de similarité
        above_threshold = similarities >= threshold
        if not np.any(above_threshold):
            logger.warning(f"Aucun résultat au-dessus du seuil de similarité {threshold}, essai avec un seuil plus bas")
            threshold = -0.5  # Seuil encore plus bas
            above_threshold = similarities >= threshold
            
            # Si toujours rien, prendre les meilleurs quand même
            if not np.any(above_threshold):
                logger.warning("Utilisation des meilleurs résultats malgré le seuil")
                # Prendre les top_k meilleurs
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                above_threshold = np.zeros_like(similarities, dtype=bool)
                above_threshold[top_indices] = True
        
        # Trier les indices par similarité décroissante
        filtered_indices = np.where(above_threshold)[0]
        filtered_similarities = similarities[above_threshold]
        top_indices_filtered = filtered_indices[np.argsort(filtered_similarities)[-top_k:][::-1]]
        
        logger.info(f"Nombre d'indices sélectionnés: {len(top_indices_filtered)}")
        
        # Récupérer les chunks correspondants
        results = []
        for idx in top_indices_filtered:
            try:
                # Vérifier que l'indice est valide
                if idx >= len(self.embeddings.chunk_ids):
                    logger.warning(f"Indice {idx} hors limites pour chunk_ids (longueur: {len(self.embeddings.chunk_ids)})")
                    continue
                
                # Récupérer l'ID du chunk
                chunk_id = self.embeddings.chunk_ids[idx]
                
                # Extraire doc_id et chunk_id
                try:
                    parts = chunk_id.split('_')
                    if len(parts) == 2:
                        doc_id, chunk_idx_str = parts
                        chunk_idx = int(chunk_idx_str)
                    else:
                        logger.warning(f"Format d'ID de chunk invalide (pas deux parties): {chunk_id}")
                        continue
                except Exception as e:
                    logger.warning(f"Format d'ID de chunk invalide: {chunk_id}, erreur: {e}")
                    continue
                
                # Chercher le chunk correspondant
                found = False
                for chunk in chunks:
                    # Vérifier les métadonnées
                    if "metadata" not in chunk:
                        continue
                        
                    metadata = chunk.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                    
                    if metadata.get("doc_id") == doc_id and metadata.get("chunk_id") == chunk_idx:
                        # Ajouter le score de similarité
                        chunk_with_score = chunk.copy()
                        chunk_with_score["score"] = float(similarities[idx])
                        results.append(chunk_with_score)
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Chunk non trouvé pour doc_id={doc_id}, chunk_id={chunk_idx}")
                    # Approche de secours: prendre le premier chunk avec le bon doc_id
                    for chunk in chunks:
                        if "metadata" in chunk and chunk["metadata"].get("doc_id") == doc_id:
                            chunk_with_score = chunk.copy()
                            chunk_with_score["score"] = float(similarities[idx])
                            results.append(chunk_with_score)
                            logger.info(f"Utilisé chunk de secours pour doc_id={doc_id}")
                            break
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du chunk à l'indice {idx}: {e}")
                continue
        
        logger.info(f"Requête '{query[:50]}...' a retourné {len(results)} résultats")
        for i, result in enumerate(results):
            if i < 3:  # limiter à 3 logs pour éviter de polluer
                logger.info(f"Résultat {i+1}: score={result['score']:.4f}, doc={result['metadata'].get('filename', 'inconnu')}")
        
        return results