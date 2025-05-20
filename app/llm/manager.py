from typing import Dict, List
import logging
import os
from ctransformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

class LLMManager:
    """Gestionnaire des modèles de langage"""
    
    def __init__(self):
        self.model = None
        self.model_path = os.path.join("models", "tinyllama.gguf")
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle de langage"""
        try:
            if os.path.exists(self.model_path):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    model_type="llama",
                    max_new_tokens=512,
                    context_length=2048
                )
                logger.info("Modèle chargé avec succès")
            else:
                logger.warning(f"Modèle non trouvé à {self.model_path}. Utilisation du mode sans modèle.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        """
        Génère une réponse basée sur le contexte et la question
        
        Args:
            query: Question posée
            context: Liste des documents pertinents
            
        Returns:
            str: Réponse générée
        """
        try:
            if not self.model:
                return "Le modèle n'est pas disponible. Veuillez vérifier l'installation."
            
            # Construire le prompt
            prompt = self._build_prompt(query, context)
            
            # Générer la réponse
            response = self.model(prompt)
            
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            return f"Erreur lors de la génération de la réponse: {str(e)}"
    
    def _build_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Construit le prompt pour le modèle
        
        Args:
            query: Question posée
            context: Liste des documents pertinents
            
        Returns:
            str: Prompt formaté
        """
        # Extraire les textes du contexte
        context_texts = [doc.get('content', '') for doc in context]
        
        # Construire le prompt
        prompt = "Contexte:\n"
        for i, text in enumerate(context_texts, 1):
            prompt += f"{i}. {text}\n"
        
        prompt += f"\nQuestion: {query}\n"
        prompt += "Réponse: "
        
        return prompt 