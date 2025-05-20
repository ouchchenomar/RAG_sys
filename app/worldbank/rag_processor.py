# app/worldbank/rag_processor.py
import os
import logging
import traceback
from typing import Dict, Any, List, Optional, Union
import re
import uuid

from app.rag_pipeline.processor import RAGProcessor
from .data_collector import WorldBankDataCollector
from .document_generator import WorldBankDocumentGenerator

logger = logging.getLogger(__name__)

class WorldBankRAGProcessor(RAGProcessor):
    """Extension du processeur RAG pour les données de la Banque Mondiale"""
    
    def __init__(self, use_openai=False, openai_api_key=None, language="fr"):
        """
        Initialise le processeur RAG spécialisé pour la Banque Mondiale
        Args:
            use_openai (bool): Utiliser l'API OpenAI au lieu du modèle local
            openai_api_key (str): Clé API OpenAI (si None et use_openai=True, cherche dans les variables d'environnement)
            language (str): Langue des données ('fr' pour français, 'en' pour anglais)
        """
        # Appel au constructeur parent
        try:
            super().__init__()
            logger.info("Initialisation du processeur RAG réussie")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du processeur RAG: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Initialiser le collecteur de données et le générateur de documents
        try:
            self.wb_collector = WorldBankDataCollector(language=language)
            self.wb_document_generator = WorldBankDocumentGenerator(self.wb_collector)
            logger.info("Initialisation du collecteur et générateur de documents réussie")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du collecteur ou générateur de documents: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_worldbank_info(self):
        """
        Récupère les informations spécifiques au RAG pour la Banque Mondiale
        Returns:
            dict: Informations sur le système RAG pour la Banque Mondiale
        """
        try:
            # Tester la connexion à l'API de la Banque Mondiale
            api_status = "OK"
            try:
                countries = self.wb_collector.get_countries()
                if not countries or not isinstance(countries, list) or len(countries) < 2 or not countries[1]:
                    api_status = "Erreur - Impossible de récupérer les pays"
            except Exception as e:
                api_status = "Erreur - " + str(e)
                logger.error(f"Erreur lors de la vérification de l'API: {e}")
            
            # Vérifier l'état des fichiers de cache
            cache_status = "Présent" if os.path.exists(self.wb_collector.cache_dir) and os.listdir(self.wb_collector.cache_dir) else "Absent"
            
            # Vérifier l'état des indices vectoriels
            index_status = "Présents" if hasattr(self.vector_store, "embeddings") and getattr(self.vector_store.embeddings, "doc_vectors", None) is not None else "Absents"
            
            # Vérifier le nombre de documents
            doc_count = 0
            try:
                chunks = self.doc_manager.get_all_chunks()
                doc_ids = set()
                for chunk in chunks:
                    if chunk.get("metadata") and chunk["metadata"].get("doc_id"):
                        doc_ids.add(chunk["metadata"]["doc_id"])
                doc_count = len(doc_ids)
            except Exception as e:
                logger.error(f"Erreur lors du comptage des documents: {e}")
            
            # Vérifier le nombre de chunks
            chunk_count = 0
            try:
                chunks = self.doc_manager.get_all_chunks()
                chunk_count = len(chunks)
            except Exception as e:
                logger.error(f"Erreur lors du comptage des chunks: {e}")
            
            # Vérifier si le modèle est chargé
            model_loaded = hasattr(self, "llm_manager") and hasattr(self.llm_manager, "model") and self.llm_manager.model is not None
            model_name = self.llm_manager.model_name if hasattr(self, "llm_manager") and hasattr(self.llm_manager, "model_name") else None
            
            return {
                "success": True,
                "api_status": api_status,
                "server_status": "OK",  # Le serveur fonctionne si cette fonction est appelée
                "cache_status": cache_status,
                "index_status": index_status,
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "model_loaded": model_loaded,
                "model_name": model_name
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations système: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "message": f"Erreur lors de la récupération des informations système: {e}",
                "api_status": "Inconnu",
                "server_status": "Erreur",
                "cache_status": "Inconnu",
                "index_status": "Inconnu",
                "document_count": 0,
                "chunk_count": 0,
                "model_loaded": False,
                "model_name": None
            }
    
    def update_worldbank_knowledge(self, country_codes=None, indicator_ids=None, include_topics=True):
        """
        Met à jour la base de connaissances avec les données de la Banque Mondiale
        Args:
            country_codes (list): Liste de codes pays à inclure, ou None pour les pays principaux
            indicator_ids (list): Liste d'IDs d'indicateurs à inclure, ou None pour les indicateurs principaux
            include_topics (bool): Inclure les thèmes
        Returns:
            dict: Résultat de la mise à jour
        """
        try:
            documents = []
            
            # Définir les pays principaux si non spécifiés
            if not country_codes:
                country_codes = ["FRA", "USA", "DEU", "GBR", "JPN", "CHN", "IND", "BRA", "RUS", "ZAF", "MAR", "DZA", "TUN", "EGY", "NGA"]
            
            # Définir les indicateurs principaux si non spécifiés
            if not indicator_ids:
                indicator_ids = [
                    "NY.GDP.MKTP.CD",     # PIB
                    "NY.GDP.PCAP.CD",     # PIB par habitant
                    "SP.POP.TOTL",        # Population
                    "SP.DYN.LE00.IN",     # Espérance de vie
                    "SE.ADT.LITR.ZS",     # Taux d'alphabétisation
                    "SI.POV.GINI",        # Indice GINI
                    "EG.ELC.ACCS.ZS",     # Accès à l'électricité
                    "SL.UEM.TOTL.ZS",     # Taux de chômage
                    "NY.GDP.MKTP.KD.ZG",  # Croissance du PIB
                    "FP.CPI.TOTL.ZG"      # Inflation
                ]
            
            # Logs pour diagnostiquer la structure des données
            logger.info(f"Pays à inclure: {country_codes}")
            logger.info(f"Indicateurs à inclure: {indicator_ids}")
            
            # 1. Générer les documents pour les pays
            logger.info("Génération des documents pour les pays")
            try:
                country_docs = self.wb_document_generator.generate_country_documents(country_codes)
                documents.extend(country_docs)
                logger.info(f"{len(country_docs)} documents de pays générés")
            except Exception as e:
                logger.error(f"Erreur lors de la génération des documents pour les pays: {e}")
                logger.error(traceback.format_exc())
            
            # 2. Générer les documents pour les indicateurs
            logger.info("Génération des documents pour les indicateurs")
            try:
                indicator_docs = self.wb_document_generator.generate_indicator_documents(indicator_ids)
                documents.extend(indicator_docs)
                logger.info(f"{len(indicator_docs)} documents d'indicateurs générés")
            except Exception as e:
                logger.error(f"Erreur lors de la génération des documents pour les indicateurs: {e}")
                logger.error(traceback.format_exc())
            
            # 3. Générer les documents de comparaison régionale
            try:
                logger.info("Génération des documents de comparaison régionale")
                region_codes = ["EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF"]
                
                # Préparer les tuples d'indicateurs (id, nom)
                indicator_tuples = []
                for ind_id in indicator_ids:
                    # Chercher le nom de l'indicateur
                    indicator_name = ind_id  # Par défaut, utiliser l'ID comme nom
                    indicator_info = self.wb_collector.get_indicators(ind_id)
                    if isinstance(indicator_info, list) and len(indicator_info) > 1 and indicator_info[1]:
                        for ind in indicator_info[1]:
                            if ind.get('id') == ind_id:
                                indicator_name = ind.get('name', ind_id)
                                break
                    indicator_tuples.append((ind_id, indicator_name))
                
                comparison_docs = self.wb_document_generator.generate_regional_comparison_documents(
                    indicator_tuples,
                    region_codes
                )
                documents.extend(comparison_docs)
                logger.info(f"{len(comparison_docs)} documents de comparaison générés")
            except Exception as e:
                logger.error(f"Erreur lors de la génération des documents de comparaison: {e}")
                logger.error(traceback.format_exc())
            
            # 4. Générer les documents pour les thèmes
            if include_topics:
                try:
                    logger.info("Génération des documents pour les thèmes")
                    topic_docs = self.wb_document_generator.generate_topic_documents()
                    documents.extend(topic_docs)
                    logger.info(f"{len(topic_docs)} documents de thèmes générés")
                except Exception as e:
                    logger.error(f"Erreur lors de la génération des documents pour les thèmes: {e}")
                    logger.error(traceback.format_exc())
            
            # Vérifier que nous avons des documents à ajouter
            if not documents:
                return {
                    "success": False,
                    "message": "Aucun document n'a pu être généré",
                    "document_count": 0,
                    "document_ids": []
                }
            
            # Ajouter les documents à la base de connaissances
            doc_ids = []
            for doc in documents:
                try:
                    # Vérifier que le document est bien formé
                    if not isinstance(doc, dict) or 'content' not in doc or 'metadata' not in doc:
                        logger.warning(f"Document mal formé: {doc}")
                        continue
                    
                    # Convertir le document en format texte et métadonnées
                    content = doc["content"]
                    metadata = doc["metadata"]
                    
                    # S'assurer que le document est au bon format
                    doc_formatted = {
                        "text": content,
                        "metadata": metadata
                    }
                    
                    # Vérifier que les métadonnées contiennent un ID
                    if 'id' not in metadata:
                        # Générer un ID pour le document s'il n'en a pas
                        metadata['id'] = str(uuid.uuid4())
                        logger.info(f"ID généré pour le document: {metadata['id']}")
                    
                    # Créer un nom de fichier valide
                    safe_id = re.sub(r'[^\w\.-]', '_', str(metadata['id']))
                    safe_type = re.sub(r'[^\w\.-]', '_', str(metadata.get('type', 'document')))
                    filename = f"{safe_type}_{safe_id}.md"
                    
                    # Ajouter le document via la méthode standard du RAG
                    doc_id = self.add_document(content.encode('utf-8'), filename)
                    
                    if doc_id:
                        doc_ids.append(doc_id)
                    else:
                        logger.warning(f"Échec de l'ajout du document {filename}")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du document {metadata.get('id', 'inconnu')}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            return {
                "success": True,
                "message": f"{len(doc_ids)} documents ajoutés à la base de connaissances",
                "document_count": len(doc_ids),
                "document_ids": doc_ids
            }
        except Exception as e:
            logger.error(f"Erreur globale lors de la mise à jour de la base de connaissances: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "message": f"Erreur lors de la mise à jour de la base de connaissances: {e}",
                "document_count": 0,
                "document_ids": []
            }
    
    def query(self, question: str, top_k: int = 5, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Interroge le système RAG avec une question
        Args:
            question (str): Question de l'utilisateur
            top_k (int): Nombre de chunks à récupérer
            max_tokens (int): Nombre maximum de tokens à générer
        Returns:
            Dict[str, Any]: Résultat de la requête
        """
        try:
            logger.info(f"Question reçue: {question}")
            
            # Vérifier si le modèle est chargé
            if not hasattr(self, "llm_manager") or not hasattr(self.llm_manager, "model") or self.llm_manager.model is None:
                logger.warning("Modèle de langage non chargé")
                return {
                    "success": True,
                    "answer": "Je ne peux pas répondre à cette question car le modèle de langage n'est pas chargé. Veuillez vérifier l'installation du modèle.",
                    "sources": []
                }
            
            # Récupérer tous les chunks
            chunks = self.doc_manager.get_all_chunks()
            logger.info(f"Nombre de chunks récupérés: {len(chunks)}")
            
            if not chunks:
                logger.warning("Aucun document disponible dans la base de connaissances")
                return {
                    "success": True,
                    "answer": "Aucun document n'est disponible dans la base de connaissances. Veuillez mettre à jour la base de connaissances d'abord.",
                    "sources": []
                }
            
            # Vérifier que les chunks ont le bon format
            for i, chunk in enumerate(chunks[:3]):
                if 'text' not in chunk:
                    # Si le chunk a 'content' mais pas 'text', créer 'text'
                    if 'content' in chunk:
                        chunk['text'] = chunk['content']
                    else:
                        logger.warning(f"Chunk sans texte détecté: {chunk}")
            
            # Rechercher les chunks pertinents avec un seuil de similarité plus bas
            try:
                # Vérifier l'état des embeddings
                if hasattr(self.vector_store.embeddings, "doc_vectors") and self.vector_store.embeddings.doc_vectors is not None:
                    if hasattr(self.vector_store.embeddings.doc_vectors, "shape"):
                        logger.info(f"Dimensions des doc_vectors: {self.vector_store.embeddings.doc_vectors.shape}")
                    else:
                        logger.info("doc_vectors présent mais pas de propriété shape")
                else:
                    logger.warning("doc_vectors non disponible dans les embeddings")
                    
                # Recherche vectorielle
                threshold = -0.2  # Seuil plus bas pour être plus inclusif
                relevant_chunks = self.vector_store.retriever.search(question, chunks, top_k, threshold)
                logger.info(f"Nombre de chunks pertinents trouvés: {len(relevant_chunks)}")
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de chunks pertinents: {e}")
                logger.error(traceback.format_exc())
                relevant_chunks = []
            
            # Si aucun chunk pertinent n'est trouvé, essayer une recherche plus simple
            if not relevant_chunks:
                # Créer une recherche par mots-clés simples comme fallback
                logger.info("Aucun chunk pertinent trouvé via la recherche vectorielle, utilisation d'une recherche par mots-clés")
                
                # Extraire les mots clés de la question
                import re
                keywords = re.findall(r'\b\w{3,}\b', question.lower())
                logger.info(f"Mots clés extraits: {keywords}")
                
                # Recherche simple par mots-clés
                if keywords:
                    for chunk in chunks:
                        if 'text' in chunk:
                            text = chunk['text'].lower()
                            # Calculer un score simple basé sur le nombre de mots clés trouvés
                            score = sum(1 for keyword in keywords if keyword in text) / len(keywords)
                            if score > 0.2:  # Au moins 20% des mots clés présents
                                chunk_copy = chunk.copy()
                                chunk_copy['score'] = float(score)
                                relevant_chunks.append(chunk_copy)
                    
                    # Trier par score
                    relevant_chunks = sorted(relevant_chunks, key=lambda x: x.get('score', 0), reverse=True)
                    # Limiter aux top_k résultats
                    relevant_chunks = relevant_chunks[:top_k]
                    logger.info(f"Recherche par mots-clés: {len(relevant_chunks)} chunks trouvés")
            
            # Si toujours aucun chunk pertinent, répondre par un message générique
            if not relevant_chunks:
                logger.warning("Aucun chunk pertinent trouvé même avec la recherche par mots-clés")
                return {
                    "success": True,
                    "answer": "Je n'ai pas trouvé d'information pertinente pour répondre à votre question dans les documents disponibles. Essayez de reformuler votre question ou de mettre à jour la base de connaissances avec plus de pays et d'indicateurs.",
                    "sources": []
                }
            
            # Log des chunks trouvés
            for i, chunk in enumerate(relevant_chunks):
                logger.info(f"Chunk {i+1}: score={chunk.get('score', 0)}, metadata={chunk.get('metadata', {})}")
                if 'text' in chunk:
                    logger.info(f"Contenu: {chunk['text'][:100]}...")
            
            # Créer le prompt avec contexte
            try:
                prompt = self.llm_manager.create_prompt(question, relevant_chunks)
                logger.debug(f"Prompt: {prompt[:500]}...")
            except Exception as e:
                logger.error(f"Erreur lors de la création du prompt: {e}")
                logger.error(traceback.format_exc())
                
                # Créer un prompt de secours simple
                context = "\n\n".join([f"Document {i+1}: {chunk.get('text', '')[:500]}" for i, chunk in enumerate(relevant_chunks)])
                prompt = f"""Tu es un assistant IA qui répond aux questions en te basant uniquement sur le contexte fourni.
Si tu ne trouves pas la réponse dans le contexte, dis-le clairement mais essaie d'être utile.

Contexte:
{context}

Question: {question}

Réponse:"""
            
            # Générer une réponse
            try:
                logger.info("Génération de la réponse avec le modèle de langage")
                answer = self.llm_manager.generate(prompt, max_tokens=max_tokens)
                logger.info(f"Réponse générée: {answer[:100]}...")
            except Exception as e:
                logger.error(f"Erreur lors de la génération de la réponse: {e}")
                logger.error(traceback.format_exc())
                
                # Créer une réponse de secours
                answer = "Je rencontre des difficultés à générer une réponse complète. Voici les informations les plus pertinentes que j'ai trouvées :\n\n"
                
                for i, chunk in enumerate(relevant_chunks[:3]):
                    if 'text' in chunk:
                        answer += f"- Source {i+1}: {chunk.get('metadata', {}).get('filename', 'Document')}\n"
                        answer += f"{chunk['text'][:200]}...\n\n"
            
            # Préparer les sources
            sources = []
            for chunk in relevant_chunks:
                metadata = chunk.get('metadata', {})
                sources.append({
                    "filename": metadata.get('filename', 'Document inconnu'),
                    "doc_id": metadata.get('doc_id', 'unknown'),
                    "chunk_id": metadata.get('chunk_id', 0),
                    "score": float(chunk.get('score', 0.0))
                })
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Erreur lors de la requête: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": True,  # Pour éviter une erreur dans l'interface
                "answer": f"Désolé, une erreur est survenue lors du traitement de votre question: {str(e)}",
                "sources": []
            }