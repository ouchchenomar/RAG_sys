import os
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logging(log_dir="./logs", level=logging.INFO):
    """
    Configurer le système de logging
    Args:
        log_dir (str): Répertoire pour les fichiers de log
        level: Niveau de logging
    """
    # Créer le répertoire de logs s'il n'existe pas
    os.makedirs(log_dir, exist_ok=True)
    
    # Créer un formateur pour les logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configurer le handler pour le fichier de log
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "rag_system.log"),
        maxBytes=10*1024*1024,  # 10 Mo
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Configurer le handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configurer le logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Supprimer les handlers existants
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Ajouter les nouveaux handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Créer des loggers spécifiques pour différentes parties de l'application
    loggers = {
        "app": logging.getLogger("app"),
        "app.document_processor": logging.getLogger("app.document_processor"),
        "app.vector_store": logging.getLogger("app.vector_store"),
        "app.llm": logging.getLogger("app.llm"),
        "app.rag_pipeline": logging.getLogger("app.rag_pipeline"),
        "app.api": logging.getLogger("app.api")
    }
    
    # Configurer les niveaux de log spécifiques si nécessaire
    loggers["app.llm"].setLevel(logging.INFO)
    
    return loggers