import requests
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class WorldBankDataCollector:
    """Collecteur de données pour l'API de la Banque Mondiale"""
    
    def __init__(self, language="fr", cache_dir="./data/cache"):
        self.base_url = "https://api.worldbank.org/v2"
        self.language = language
        self.cache_dir = cache_dir
        self.cache = {}
        self._ensure_cache_dir()
        
        # Configurer la session HTTP
        self.session = requests.Session()
        
    def _ensure_cache_dir(self):
        """Assure que le répertoire de cache existe"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_cache(self):
        """Charge le cache depuis le disque"""
        cache_file = os.path.join(self.cache_dir, f"worldbank_cache_{self.language}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                    logger.info(f"Cache chargé avec {len(self.cache)} entrées")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du cache: {e}")
                self.cache = {}
        else:
            logger.info(f"Aucun cache trouvé à {cache_file}")
    
    def _save_cache(self):
        """Sauvegarde le cache sur le disque"""
        cache_file = os.path.join(self.cache_dir, f"worldbank_cache_{self.language}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache sauvegardé avec {len(self.cache)} entrées")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du cache: {e}")
    
    def get_countries(self):
        """Récupère la liste des pays"""
        try:
            url = f"{self.base_url}/countries"
            params = {
                "format": "json",
                "per_page": 300,
                "language": self.language
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des pays: {e}")
            return [{"page": 1, "pages": 1, "per_page": 0, "total": 0}, []]
    
    def get_indicators(self, search_query=None):
        """
        Récupère la liste des indicateurs, avec option de recherche
        Args:
            search_query (str): Texte de recherche pour filtrer les indicateurs
        """
        try:
            url = f"{self.base_url}/indicators"
            params = {
                "format": "json",
                "per_page": 500,
                "language": self.language
            }
            
            if search_query:
                params["search"] = search_query
                
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des indicateurs: {e}")
            return [{"page": 1, "pages": 1, "per_page": 0, "total": 0}, []]
    
    def get_indicator_data(self, indicator_id, country_code="all", start_year=2000, end_year=2023):
        """
        Récupère les données d'un indicateur pour un pays/région sur une période
        Args:
            indicator_id (str): Code de l'indicateur (ex: 'NY.GDP.MKTP.CD' pour le PIB)
            country_code (str): Code ISO du pays ou 'all' pour tous les pays
            start_year (int): Année de début
            end_year (int): Année de fin
        """
        try:
            url = f"{self.base_url}/countries/{country_code}/indicators/{indicator_id}"
            params = {
                "format": "json",
                "date": f"{start_year}:{end_year}",
                "per_page": 1000,
                "language": self.language
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour l'indicateur {indicator_id}: {e}")
            return [{"page": 1, "pages": 1, "per_page": 0, "total": 0}, []]
    
    def get_topics(self):
        """Récupère la liste des thèmes"""
        try:
            url = f"{self.base_url}/topics"
            params = {
                "format": "json",
                "language": self.language
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des thèmes: {e}")
            return [{"page": 1, "pages": 1, "per_page": 0, "total": 0}, []]
    
    def get_topic_data(self, topic_id):
        """Récupère les données pour un thème spécifique"""
        try:
            url = f"{self.base_url}/topic/{topic_id}/indicator"
            params = {
                "format": "json",
                "language": self.language
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour le thème {topic_id}: {e}")
            return [{"page": 1, "pages": 1, "per_page": 0, "total": 0}, []]
    
    def get_country_info(self, country_code):
        """
        Récupère les informations détaillées pour un pays
        Args:
            country_code (str): Code ISO du pays
        """
        try:
            countries_data = self.get_countries()
            if isinstance(countries_data, list) and len(countries_data) > 1:
                for country in countries_data[1]:
                    if country.get("id") == country_code:
                        return country
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations pour le pays {country_code}: {e}")
            return None