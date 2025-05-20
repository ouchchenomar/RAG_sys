from typing import List, Dict, Any
from .embeddings import TFIDFEmbeddings
from .retriever import Retriever
import logging
import importlib.util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configurer le logging
logger = logging.getLogger(__name__)

# Vérifier si sentence_transformers est disponible
SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None

# Importer les modules d'embeddings
from .embeddings import TFIDFEmbeddings
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        from .sentence_embeddings import SentenceTransformerEmbeddings
    except ImportError:
        logger.warning("Module sentence_embeddings non trouvé. Utilisation de TF-IDF.")
        SENTENCE_TRANSFORMERS_AVAILABLE = False

class VectorStoreManager:
    """
    Classe pour gérer l'index vectoriel
    """
    def __init__(self, indices_dir: str = "./data/indices", use_sentence_transformers: bool = True):
        """
        Initialiser le gestionnaire d'index vectoriel
        Args:
            indices_dir (str): Répertoire de stockage des indices
            use_sentence_transformers (bool): Utiliser SentenceTransformers si disponible
        """
        self.indices_dir = indices_dir
        
        # Déterminer le type d'embeddings à utiliser
        self.use_transformers = use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_transformers:
            try:
                logger.info("Utilisation de SentenceTransformerEmbeddings")
                self.embeddings = SentenceTransformerEmbeddings(indices_dir)
                self.embeddings_type = "sentence_transformers"
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de SentenceTransformerEmbeddings: {e}")
                logger.info("Fallback vers TFIDFEmbeddings")
                self.embeddings = TFIDFEmbeddings(indices_dir)
                self.embeddings_type = "tfidf"
        else:
            logger.info("Utilisation de TFIDFEmbeddings")
            self.embeddings = TFIDFEmbeddings(indices_dir)
            self.embeddings_type = "tfidf"
        
        # Initialiser le retriever
        from .retriever import Retriever
        self.retriever = Retriever(self.embeddings)

        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.vectors = None

    def create_index(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Créer un index à partir des chunks
        Args:
            chunks (List[Dict[str, Any]]): Liste des chunks
        Returns:
            bool: True si créé avec succès, False sinon
        """
        try:
            logger.info(f"Création d'un index avec {len(chunks)} chunks")
            # Entraîner le vectoriseur sur les chunks
            self.embeddings.fit(chunks)
            
            # Sauvegarder l'index
            index_path = self.embeddings.save()
            logger.info(f"Index créé et sauvegardé: {index_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'index: {e}")
            return False

    def load_index(self) -> bool:
        """
        Charger l'index
        Returns:
            bool: True si chargé avec succès, False sinon
        """
        try:
            # Essayer de charger l'index approprié selon le type d'embeddings
            if self.use_transformers:
                success = self.embeddings.load("sentence_transformer_index")
                if not success:
                    logger.warning("Index SentenceTransformers non trouvé, tentative avec TF-IDF")
                    success = self.embeddings.load()
            else:
                success = self.embeddings.load()
            
            if success:
                logger.info("Index chargé avec succès")
            else:
                logger.warning("Aucun index trouvé")
            
            return success
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'index: {e}")
            return False

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Recherche les documents les plus pertinents
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            
        Returns:
            List[Dict]: Liste des documents pertinents
        """
        try:
            if not self.documents:
                return []
            
            # Vectoriser la requête
            query_vector = self.vectorizer.transform([query])
            
            # Calculer la similarité
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Obtenir les indices des top_k documents les plus similaires
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Retourner les documents correspondants
            results = []
            for idx in top_indices:
                results.append(self.documents[idx])
            
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []

    def add_document(self, content: str, metadata: Dict) -> None:
        """
        Ajoute un document au vector store
        
        Args:
            content: Contenu du document
            metadata: Métadonnées du document
        """
        try:
            # Ajouter le document
            self.documents.append({
                'content': content,
                'metadata': metadata
            })
            
            # Mettre à jour les vecteurs
            self._update_vectors()
            
            logger.info(f"Document ajouté avec succès: {metadata.get('filename', 'Unknown')}")
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du document: {e}")
            raise

    def _update_vectors(self) -> None:
        """Met à jour les vecteurs TF-IDF"""
        try:
            # Extraire les contenus
            contents = [doc['content'] for doc in self.documents]
            
            # Vectoriser les documents
            self.vectors = self.vectorizer.fit_transform(contents)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des vecteurs: {e}")
            raise