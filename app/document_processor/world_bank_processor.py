from typing import List, Dict
from ..world_bank_api import WorldBankAPI
import logging
import uuid

logger = logging.getLogger(__name__)

class WorldBankProcessor:
    """Processeur pour les données de la Banque Mondiale"""
    
    def __init__(self):
        self.api = WorldBankAPI()
        
    def process_indicators(self, indicators: List[str], countries: List[str] = None, years: List[int] = None) -> List[Dict]:
        """
        Traite les indicateurs spécifiés pour les pays et années donnés
        
        Args:
            indicators: Liste des codes d'indicateurs à traiter
            countries: Liste des codes pays (optionnel)
            years: Liste des années (optionnel)
            
        Returns:
            List[Dict]: Liste des documents traités
        """
        if not countries:
            countries = [country['id'] for country in self.api.get_countries_list()]
        logger.info(f"Traitement des indicateurs: {indicators} pour les pays: {countries}")
        documents = []
        for indicator in indicators:
            for country in countries:
                logger.info(f"Récupération des données pour {country} - {indicator}")
                data = self.api.get_country_data(country, indicator)
                logger.info(f"Données brutes récupérées: {str(data)[:300]}")
                if data:
                    formatted_text = self.api.format_data_for_rag(data)
                    logger.info(f"Texte formaté: {formatted_text[:200]}")
                    if formatted_text:
                        doc_id = str(uuid.uuid4())
                        documents.append({
                            'id': doc_id,
                            'content': formatted_text,
                            'metadata': {
                                'id': doc_id,
                                'source': 'World Bank API',
                                'indicator': indicator,
                                'country': country,
                                'type': 'world_bank_data'
                            }
                        })
                    else:
                        logger.warning(f"Aucun texte formaté pour {country} - {indicator}")
                else:
                    logger.warning(f"Aucune donnée récupérée pour {country} - {indicator}")
        logger.info(f"Nombre total de documents indexés: {len(documents)}")
        return documents
    
    def get_available_indicators(self) -> List[Dict]:
        """Récupère la liste des indicateurs disponibles avec leurs descriptions"""
        return self.api.get_indicators_list()
    
    def get_available_countries(self) -> List[Dict]:
        """Récupère la liste des pays disponibles"""
        return self.api.get_countries_list() 