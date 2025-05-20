import requests
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class WorldBankAPI:
    """Classe pour interagir avec l'API de la Banque Mondiale"""
    
    BASE_URL = "http://api.worldbank.org/v2"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_country_data(self, country_code: str, indicator: str, year: Optional[int] = None) -> Dict:
        """
        Récupère les données d'un indicateur pour un pays spécifique
        
        Args:
            country_code: Code pays ISO (ex: 'FR' pour France)
            indicator: Code de l'indicateur (ex: 'NY.GDP.MKTP.CD' pour PIB)
            year: Année spécifique (optionnel)
            
        Returns:
            Dict contenant les données
        """
        url = f"{self.BASE_URL}/country/{country_code}/indicator/{indicator}"
        if year:
            url += f"?date={year}"
            
        try:
            response = self.session.get(url, params={'format': 'json'})
            response.raise_for_status()
            logger.info(f"Réponse API World Bank: {response.text[:300]}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            return {}
    
    def get_indicators_list(self) -> List[Dict]:
        """Récupère la liste des indicateurs disponibles"""
        try:
            response = self.session.get(f"{self.BASE_URL}/indicators", params={'format': 'json'})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération des indicateurs: {e}")
            return []
    
    def get_countries_list(self) -> List[Dict]:
        """Récupère la liste des pays disponibles"""
        try:
            response = self.session.get(f"{self.BASE_URL}/countries", params={'format': 'json'})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération des pays: {e}")
            return []
    
    def format_data_for_rag(self, data: Dict) -> str:
        """
        Formate les données pour l'utilisation dans le système RAG
        
        Args:
            data: Données brutes de l'API
            
        Returns:
            str: Texte formaté pour l'indexation
        """
        if not data or 'data' not in data:
            logger.warning(f"Format inattendu ou vide pour les données World Bank: {data}")
            return ""
            
        formatted_text = []
        for item in data['data']:
            if item.get('value') is not None:
                text = f"En {item.get('date')}, {item.get('country', {}).get('value')} "
                text += f"a un {item.get('indicator', {}).get('value')} de {item.get('value')} "
                text += f"{item.get('unit', '')}"
                formatted_text.append(text)
                
        return "\n".join(formatted_text) 