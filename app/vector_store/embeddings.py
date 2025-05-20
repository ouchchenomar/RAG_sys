# app/vector_store/embeddings.py
import os
import pickle
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configurer le logging
logger = logging.getLogger(__name__)

class TFIDFEmbeddings:
    """
    Classe pour créer et gérer des embeddings TF-IDF
    """
    def __init__(self, indices_dir: str = "./data/indices"):
        """
        Initialiser le gestionnaire d'embeddings TF-IDF
        Args:
            indices_dir (str): Répertoire de stockage des indices
        """
        self.indices_dir = indices_dir
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_df=0.85,
            min_df=2,
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.doc_vectors = None
        self.chunk_ids = []
        
        # Créer le répertoire d'indices s'il n'existe pas
        os.makedirs(indices_dir, exist_ok=True)
        
        # Essayer de charger les embeddings s'ils existent
        self._load_embeddings()
        
        logger.info(f"TFIDFEmbeddings initialisé, {len(self.chunk_ids)} embeddings chargés")
    
    def fit(self, chunk_texts: List[str], chunk_ids: List[str]) -> None:
        """
        Entraîner le vectoriseur TF-IDF sur une liste de textes
        Args:
            chunk_texts (List[str]): Liste des textes
            chunk_ids (List[str]): Liste des identifiants de chunks
        """
        try:
            logger.info(f"Entraînement des embeddings TF-IDF sur {len(chunk_texts)} textes")
            self.doc_vectors = self.vectorizer.fit_transform(chunk_texts)
            self.chunk_ids = chunk_ids
            logger.info(f"Forme des doc_vectors: {self.doc_vectors.shape}")
            
            # Sauvegarder les embeddings
            self._save_embeddings()
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des embeddings TF-IDF: {e}")
            raise
    
    def transform(self, text: str) -> np.ndarray:
        """
        Transformer un texte en vecteur TF-IDF
        Args:
            text (str): Texte à transformer
        Returns:
            np.ndarray: Vecteur TF-IDF du texte
        """
        if self.vectorizer is None:
            raise ValueError("Le vectoriseur TF-IDF n'a pas été entraîné")
        
        return self.vectorizer.transform([text])
    
    def _save_embeddings(self) -> None:
        """
        Sauvegarder les embeddings
        """
        try:
            # Chemin du fichier d'embeddings
            embeddings_path = os.path.join(self.indices_dir, "tfidf_embeddings.pkl")
            
            # Sauvegarder le vectoriseur
            vectorizer_path = os.path.join(self.indices_dir, "tfidf_vectorizer.pkl")
            
            # Sauvegarder les vecteurs de documents
            doc_vectors_path = os.path.join(self.indices_dir, "tfidf_doc_vectors.npz")
            
            # Sauvegarder les IDs des chunks
            chunk_ids_path = os.path.join(self.indices_dir, "tfidf_chunk_ids.pkl")
            
            # Sauvegarder le vectoriseur
            with open(vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            
            # Sauvegarder les vecteurs de documents
            if isinstance(self.doc_vectors, np.ndarray):
                np.savez_compressed(doc_vectors_path, vectors=self.doc_vectors)
            else:
                # Pour les matrices sparses de scipy
                import scipy.sparse as sp
                sp.save_npz(doc_vectors_path, self.doc_vectors)
            
            # Sauvegarder les IDs des chunks
            with open(chunk_ids_path, "wb") as f:
                pickle.dump(self.chunk_ids, f)
            
            logger.info(f"Embeddings TF-IDF sauvegardés dans {self.indices_dir}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des embeddings TF-IDF: {e}")
    
    def _load_embeddings(self) -> bool:
        """
        Charger les embeddings
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            # Chemin du fichier d'embeddings
            vectorizer_path = os.path.join(self.indices_dir, "tfidf_vectorizer.pkl")
            doc_vectors_path = os.path.join(self.indices_dir, "tfidf_doc_vectors.npz")
            chunk_ids_path = os.path.join(self.indices_dir, "tfidf_chunk_ids.pkl")
            
            # Vérifier si les fichiers existent
            if not os.path.exists(vectorizer_path) or not os.path.exists(chunk_ids_path):
                logger.info("Fichiers d'embeddings TF-IDF non trouvés")
                return False
            
            # Charger le vectoriseur
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Charger les vecteurs de documents
            try:
                # D'abord essayer de charger comme une matrice sparse
                import scipy.sparse as sp
                self.doc_vectors = sp.load_npz(doc_vectors_path)
            except:
                # Ensuite essayer de charger comme un tableau numpy
                try:
                    data = np.load(doc_vectors_path)
                    self.doc_vectors = data["vectors"]
                except:
                    logger.warning("Impossible de charger les vecteurs de documents")
                    return False
            
            # Charger les IDs des chunks
            with open(chunk_ids_path, "rb") as f:
                self.chunk_ids = pickle.load(f)
            
            logger.info(f"Embeddings TF-IDF chargés: {len(self.chunk_ids)} chunks")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement des embeddings TF-IDF: {e}")
            return False


class SentenceTransformerEmbeddings:
    """
    Classe pour créer et gérer des embeddings avec SentenceTransformers
    """
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", indices_dir: str = "./data/indices"):
        """
        Initialiser le gestionnaire d'embeddings SentenceTransformers
        Args:
            model_name (str): Nom du modèle SentenceTransformer à utiliser
            indices_dir (str): Répertoire de stockage des indices
        """
        self.indices_dir = indices_dir
        self.model_name = model_name
        self.model = None
        self.doc_vectors = None
        self.chunk_ids = []
        
        # Créer le répertoire d'indices s'il n'existe pas
        os.makedirs(indices_dir, exist_ok=True)
        
        # Essayer de charger les embeddings s'ils existent
        loaded = self._load_embeddings()
        
        # Charger le modèle si nécessaire
        if not loaded:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                logger.info(f"Modèle SentenceTransformer {model_name} chargé")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle SentenceTransformer: {e}")
                logger.warning("Fallback vers TF-IDF pour les embeddings")
                # Créer un vectoriseur TF-IDF de secours
                self.fallback_tfidf = TFIDFEmbeddings(indices_dir)
        else:
            logger.info(f"Embeddings SentenceTransformer chargés: {len(self.chunk_ids)} chunks")
    
    def fit(self, chunk_texts: List[str], chunk_ids: List[str]) -> None:
        """
        Générer les embeddings pour une liste de textes
        Args:
            chunk_texts (List[str]): Liste des textes
            chunk_ids (List[str]): Liste des identifiants de chunks
        """
        try:
            if self.model is None:
                logger.warning("Modèle SentenceTransformer non disponible, utilisation de TF-IDF")
                # Utiliser TF-IDF comme fallback
                if hasattr(self, "fallback_tfidf"):
                    self.fallback_tfidf.fit(chunk_texts, chunk_ids)
                    self.doc_vectors = self.fallback_tfidf.doc_vectors
                    self.chunk_ids = self.fallback_tfidf.chunk_ids
                else:
                    raise ValueError("Modèle SentenceTransformer non disponible et fallback TF-IDF non initialisé")
                return
            
            logger.info(f"Génération des embeddings pour {len(chunk_texts)} textes")
            
            # Générer les embeddings
            self.doc_vectors = self.model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
            self.chunk_ids = chunk_ids
            
            logger.info(f"Embeddings générés, forme: {self.doc_vectors.shape}")
            
            # Sauvegarder les embeddings
            self._save_embeddings()
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {e}")
            raise
    
    def transform(self, text: str) -> np.ndarray:
        """
        Transformer un texte en vecteur d'embedding
        Args:
            text (str): Texte à transformer
        Returns:
            np.ndarray: Vecteur d'embedding du texte
        """
        if self.model is None:
            # Utiliser TF-IDF comme fallback
            if hasattr(self, "fallback_tfidf"):
                return self.fallback_tfidf.transform(text)
            else:
                raise ValueError("Modèle SentenceTransformer non disponible et fallback TF-IDF non initialisé")
        
        # Générer l'embedding
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding
    
    def _save_embeddings(self) -> None:
        """
        Sauvegarder les embeddings
        """
        try:
            # Chemin des fichiers
            doc_vectors_path = os.path.join(self.indices_dir, "st_doc_vectors.npy")
            chunk_ids_path = os.path.join(self.indices_dir, "st_chunk_ids.pkl")
            
            # Sauvegarder les vecteurs de documents
            np.save(doc_vectors_path, self.doc_vectors)
            
            # Sauvegarder les IDs des chunks
            with open(chunk_ids_path, "wb") as f:
                pickle.dump(self.chunk_ids, f)
            
            logger.info(f"Embeddings SentenceTransformer sauvegardés dans {self.indices_dir}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des embeddings: {e}")
    
    def _load_embeddings(self) -> bool:
        """
        Charger les embeddings
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            # Chemin des fichiers
            doc_vectors_path = os.path.join(self.indices_dir, "st_doc_vectors.npy")
            chunk_ids_path = os.path.join(self.indices_dir, "st_chunk_ids.pkl")
            
            # Vérifier si les fichiers existent
            if not os.path.exists(doc_vectors_path) or not os.path.exists(chunk_ids_path):
                logger.info("Fichiers d'embeddings SentenceTransformer non trouvés")
                return False
            
            # Charger les vecteurs de documents
            self.doc_vectors = np.load(doc_vectors_path)
            
            # Charger les IDs des chunks
            with open(chunk_ids_path, "rb") as f:
                self.chunk_ids = pickle.load(f)
            
            logger.info(f"Embeddings SentenceTransformer chargés: {len(self.chunk_ids)} chunks")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement des embeddings: {e}")
            return False