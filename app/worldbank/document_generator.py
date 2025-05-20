import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class WorldBankDocumentGenerator:
    """Générateur de documents à partir des données de la Banque Mondiale"""
    
    def __init__(self, collector):
        """
        Initialise le générateur de documents
        Args:
            collector: Collecteur de données de la Banque Mondiale
        """
        self.collector = collector
    
    def generate_country_documents(self, country_codes=None):
        """
        Génère des documents pour les pays
        Args:
            country_codes (list): Liste de codes pays, ou None pour tous les pays
        Returns:
            list: Liste de documents formatés pour le RAG
        """
        countries_data = self.collector.get_countries()
        
        # Vérifier que les données sont au format attendu
        if not isinstance(countries_data, list) or len(countries_data) < 2:
            logger.warning("Format de données pays inattendu")
            return []
        
        countries_data = countries_data[1]  # Prendre la deuxième partie (la première contient des métadonnées)
        
        # Filtrer les pays si une liste est fournie
        if country_codes:
            filtered_countries = []
            for country in countries_data:
                if country.get('id') in country_codes:
                    filtered_countries.append(country)
            countries_data = filtered_countries
        
        documents = []
        for country in countries_data:
            try:
                # Vérifier si c'est un pays réel
                if not country.get('name'):
                    continue
                
                # Créer le contenu du document
                content = f"# {country['name']}\n\n"
                content += f"**Code ISO**: {country.get('id', 'Non disponible')}\n\n"
                
                if country.get('region'):
                    content += f"**Région**: {country['region'].get('value', 'Non spécifiée')}\n\n"
                
                if country.get('capitalCity'):
                    content += f"**Capitale**: {country['capitalCity']}\n\n"
                
                if country.get('incomeLevel'):
                    content += f"**Revenu**: {country['incomeLevel'].get('value', 'Non spécifié')}\n\n"
                
                # Récupérer quelques indicateurs clés pour ce pays
                try:
                    # PIB
                    gdp_data = self.collector.get_indicator_data('NY.GDP.MKTP.CD', country['id'])
                    if isinstance(gdp_data, list) and len(gdp_data) > 1 and gdp_data[1]:
                        # Tri par date (la plus récente en premier)
                        sorted_gdp = sorted(gdp_data[1], key=lambda x: x.get('date', ''), reverse=True)
                        for data_point in sorted_gdp:
                            if data_point.get('value'):
                                content += f"**PIB (année {data_point['date']})**: {data_point['value']:,.2f} USD\n\n"
                                break
                    
                    # Population
                    pop_data = self.collector.get_indicator_data('SP.POP.TOTL', country['id'])
                    if isinstance(pop_data, list) and len(pop_data) > 1 and pop_data[1]:
                        # Tri par date (la plus récente en premier)
                        sorted_pop = sorted(pop_data[1], key=lambda x: x.get('date', ''), reverse=True)
                        for data_point in sorted_pop:
                            if data_point.get('value'):
                                content += f"**Population (année {data_point['date']})**: {int(data_point['value']):,} habitants\n\n"
                                break
                    
                    # Espérance de vie
                    life_data = self.collector.get_indicator_data('SP.DYN.LE00.IN', country['id'])
                    if isinstance(life_data, list) and len(life_data) > 1 and life_data[1]:
                        # Tri par date (la plus récente en premier)
                        sorted_life = sorted(life_data[1], key=lambda x: x.get('date', ''), reverse=True)
                        for data_point in sorted_life:
                            if data_point.get('value'):
                                content += f"**Espérance de vie (année {data_point['date']})**: {data_point['value']:.1f} ans\n\n"
                                break
                except Exception as e:
                    logger.warning(f"Erreur lors de la récupération des indicateurs pour {country['name']}: {e}")
                
                # Créer les métadonnées
                metadata = {
                    'type': 'country',
                    'id': country['id'],
                    'name': country['name'],
                    'region': country.get('region', {}).get('value', ''),
                    'source': 'World Bank API'
                }
                
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
            except Exception as e:
                logger.warning(f"Erreur lors de la génération du document pour le pays {country.get('id', 'inconnu')}: {e}")
        
        return documents
    
    def generate_indicator_documents(self, indicator_ids=None):
        """
        Génère des documents pour les indicateurs
        Args:
            indicator_ids (list): Liste de codes d'indicateurs
        Returns:
            list: Liste de documents formatés pour le RAG
        """
        if not indicator_ids:
            return []
            
        documents = []
        for indicator_id in indicator_ids:
            try:
                # Récupérer les informations sur l'indicateur
                indicator_info = self.collector.get_indicators(indicator_id)
                
                if not isinstance(indicator_info, list) or len(indicator_info) < 2 or not indicator_info[1]:
                    logger.warning(f"Informations manquantes pour l'indicateur {indicator_id}")
                    continue
                
                indicator = None
                for ind in indicator_info[1]:
                    if ind.get('id') == indicator_id:
                        indicator = ind
                        break
                
                if not indicator:
                    logger.warning(f"Indicateur {indicator_id} non trouvé")
                    continue
                
                # Créer le contenu du document
                content = f"# Indicateur: {indicator['name']}\n\n"
                content += f"**Code**: {indicator['id']}\n\n"
                
                if indicator.get('sourceNote'):
                    content += f"**Description**: {indicator['sourceNote']}\n\n"
                
                if indicator.get('sourceOrganization'):
                    content += f"**Organisation source**: {indicator['sourceOrganization']}\n\n"
                
                # Récupérer les données mondiales pour cet indicateur
                try:
                    global_data = self.collector.get_indicator_data(indicator['id'], 'WLD')
                    if isinstance(global_data, list) and len(global_data) > 1 and global_data[1]:
                        # Tri par date (la plus récente en premier)
                        sorted_data = sorted(global_data[1], key=lambda x: x.get('date', ''), reverse=True)
                        
                        content += "## Données mondiales récentes\n\n"
                        content += "| Année | Valeur |\n|-------|--------|\n"
                        
                        # Limiter à 10 dernières années
                        for data_point in sorted_data[:10]:
                            if data_point.get('value') is not None:
                                content += f"| {data_point['date']} | {data_point['value']} |\n"
                except Exception as e:
                    logger.warning(f"Erreur lors de la récupération des données mondiales pour {indicator['id']}: {e}")
                
                # Créer les métadonnées
                metadata = {
                    'type': 'indicator',
                    'id': indicator['id'],
                    'name': indicator['name'],
                    'source': 'World Bank API'
                }
                
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
            except Exception as e:
                logger.warning(f"Erreur lors de la génération du document pour l'indicateur {indicator_id}: {e}")
        
        return documents
    
    def generate_topic_documents(self):
        """
        Génère des documents pour les thèmes de la Banque Mondiale
        Returns:
            list: Liste de documents formatés pour le RAG
        """
        topics_data = self.collector.get_topics()
        
        # Vérifier que les données sont au format attendu
        if not isinstance(topics_data, list) or len(topics_data) < 2:
            logger.warning("Format de données thèmes inattendu")
            return []
        
        topics = topics_data[1]
        documents = []
        
        for topic in topics:
            try:
                # Créer le contenu du document
                content = f"# Thème: {topic['value']}\n\n"
                content += f"**ID**: {topic['id']}\n\n"
                
                # Récupérer les indicateurs liés à ce thème
                indicators_data = self.collector.get_topic_data(topic['id'])
                
                if isinstance(indicators_data, list) and len(indicators_data) > 1 and indicators_data[1]:
                    indicators = indicators_data[1]
                    
                    content += f"## Indicateurs liés au thème ({len(indicators)})\n\n"
                    # Limiter à 20 indicateurs pour éviter des documents trop longs
                    for indicator in indicators[:20]:
                        content += f"- **{indicator['name']}** (Code: {indicator['id']})\n"
                    
                    if len(indicators) > 20:
                        content += f"\n*Note: {len(indicators) - 20} autres indicateurs non affichés.*\n"
                
                # Créer les métadonnées
                metadata = {
                    'type': 'topic',
                    'id': topic['id'],
                    'name': topic['value'],
                    'source': 'World Bank API'
                }
                
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
            except Exception as e:
                logger.warning(f"Erreur lors de la génération du document pour le thème {topic.get('id', 'inconnu')}: {e}")
        
        return documents
    
    def generate_regional_comparison_documents(self, indicators, regions=None):
        """
        Génère des documents de comparaison régionale pour certains indicateurs
        Args:
            indicators (list): Liste de tuples (id, nom) d'indicateurs
            regions (list): Liste de codes région, ou None pour les régions principales
        Returns:
            list: Liste de documents formatés pour le RAG
        """
        if not regions:
            regions = ["EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF"]
        
        documents = []
        
        for indicator_id, indicator_name in indicators:
            try:
                # Créer le contenu du document
                content = f"# Comparaison régionale: {indicator_name}\n\n"
                content += f"**Indicateur**: {indicator_id}\n\n"
                
                content += "## Données par région (dernière année disponible)\n\n"
                content += "| Région | Année | Valeur |\n|--------|-------|--------|\n"
                
                for region_code in regions:
                    try:
                        # Récupérer les infos de la région
                        region_name = region_code
                        countries_data = self.collector.get_countries()
                        
                        if isinstance(countries_data, list) and len(countries_data) > 1:
                            for country in countries_data[1]:
                                if country.get('id') == region_code:
                                    region_name = country.get('name', region_code)
                                    break
                        
                        # Récupérer les données pour cette région
                        region_data = self.collector.get_indicator_data(indicator_id, region_code)
                        
                        if isinstance(region_data, list) and len(region_data) > 1 and region_data[1]:
                            # Tri par date (la plus récente en premier)
                            sorted_data = sorted(region_data[1], key=lambda x: x.get('date', ''), reverse=True)
                            
                            # Prendre la donnée la plus récente
                            for data_point in sorted_data:
                                if data_point.get('value') is not None:
                                    content += f"| {region_name} | {data_point['date']} | {data_point['value']} |\n"
                                    break
                    except Exception as e:
                        logger.warning(f"Erreur lors de la récupération des données pour la région {region_code}: {e}")
                
                # Créer les métadonnées
                metadata = {
                    'type': 'regional_comparison',
                    'id': f"comparison_{indicator_id}",
                    'name': f"Comparaison régionale: {indicator_name}",
                    'source': 'World Bank API'
                }
                
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
            except Exception as e:
                logger.warning(f"Erreur lors de la génération du document de comparaison pour {indicator_name}: {e}")
        
        return documents