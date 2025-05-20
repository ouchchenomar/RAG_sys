import logging
from typing import List, Dict, Any

# Configurer le logging
logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Classe pour découper des documents en chunks avec chevauchement
    """
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        """
        Initialiser le découpeur de documents avec des paramètres améliorés
        Args:
            chunk_size (int): Taille des chunks en caractères
            chunk_overlap (int): Chevauchement entre les chunks en caractères
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentChunker initialisé avec chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Découper un texte en chunks avec chevauchement, avec une méthode améliorée
        Args:
            text (str): Texte à découper
            doc_metadata (Dict[str, Any]): Métadonnées du document
        Returns:
            List[Dict[str, Any]]: Liste des chunks avec leurs métadonnées
        """
        chunks = []
        
        # Vérifier si le texte est plus petit que la taille d'un chunk
        if len(text) <= self.chunk_size:
            chunks.append({
                "text": text,
                "metadata": {
                    "doc_id": doc_metadata["id"],
                    "filename": doc_metadata["filename"],
                    "chunk_id": 0
                }
            })
            return chunks
            
        # Découper le texte en paragraphes d'abord
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        # Regrouper les paragraphes en chunks
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            # Si ajouter ce paragraphe dépasse la taille du chunk et le chunk n'est pas vide
            if current_size + len(paragraph) > self.chunk_size and current_chunk:
                # Sauvegarder le chunk actuel
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_metadata["id"],
                        "filename": doc_metadata["filename"],
                        "chunk_id": chunk_id
                    }
                })
                
                # Calculer combien de paragraphes à conserver pour le chevauchement
                chars_to_keep = 0
                paragraphs_to_keep = []
                
                for p in reversed(current_chunk):
                    if chars_to_keep + len(p) <= self.chunk_overlap:
                        chars_to_keep += len(p) + 1  # +1 pour le \n
                        paragraphs_to_keep.insert(0, p)
                    else:
                        break
                        
                # Commencer un nouveau chunk avec les paragraphes chevauchants
                current_chunk = paragraphs_to_keep
                current_size = sum(len(p) for p in current_chunk) + len(current_chunk)  # Taille + \n
                chunk_id += 1
            
            # Ajouter le paragraphe au chunk actuel
            current_chunk.append(paragraph)
            current_size += len(paragraph) + 1  # +1 pour le \n
        
        # Ajouter le dernier chunk s'il n'est pas vide
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_metadata["id"],
                    "filename": doc_metadata["filename"],
                    "chunk_id": chunk_id
                }
            })
        
        logger.info(f"Document découpé en {len(chunks)} chunks")
        return chunks