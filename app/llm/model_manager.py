import os
import logging
from typing import Dict, Any, List
from ctransformers import AutoModelForCausalLM

# Configurer le logging
logger = logging.getLogger(__name__)

class LLMManager:
    """
    Classe pour gérer les modèles de langage
    """
    def __init__(self, models_dir: str = "./models"):
        """
        Initialiser le gestionnaire de modèles
        Args:
            models_dir (str): Répertoire de stockage des modèles
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.model = None
        self.model_name = None
        self.model_type = "llama"  # Type par défaut

    def load_model(self, model_name: str = "tinyllama", model_type: str = None) -> bool:
        """
        Charger un modèle local
        Args:
            model_name (str): Nom du modèle
            model_type (str): Type du modèle (llama, mistral, falcon, etc.)
        Returns:
            bool: True si chargé avec succès, False sinon
        """
        # Vérifier si le modèle est déjà chargé
        if self.model is not None and self.model_name == model_name:
            return True

        # Déterminer le type de modèle basé sur le nom
        if model_type is None:
            if "llama" in model_name.lower():
                model_type = "llama"
            elif "mistral" in model_name.lower():
                model_type = "mistral"
            elif "falcon" in model_name.lower():
                model_type = "falcon"
            else:
                model_type = "llama"  # Par défaut
            
        self.model_type = model_type

        # Chemins possibles pour le modèle
        model_paths = [
            os.path.join(self.models_dir, f"{model_name}.gguf"),  # Format GGUF
            os.path.join(self.models_dir, model_name)  # Dossier contenant le modèle
        ]
        
        # Chercher également dans des sous-dossiers spécifiques
        model_subdirs = ["gguf", "models", model_name]
        for subdir in model_subdirs:
            path = os.path.join(self.models_dir, subdir, f"{model_name}.gguf")
            model_paths.append(path)
            
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            logger.error(f"Modèle '{model_name}' non trouvé. Veuillez le télécharger et le placer dans {self.models_dir}")
            return False

        try:
            # Charger le modèle avec plus d'options pour améliorer la qualité
            logger.info(f"Chargement du modèle {model_name} depuis {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                context_length=2048,  # Augmenter le contexte pour de meilleures réponses
                gpu_layers=0  # Mettre à un nombre plus élevé si vous avez un GPU
            )
            self.model_name = model_name
            logger.info(f"Modèle {model_name} chargé avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95) -> str:
        """
        Générer une réponse à partir d'un prompt
        Args:
            prompt (str): Prompt pour le modèle
            max_tokens (int): Nombre maximum de tokens à générer
            temperature (float): Température pour le sampling (plus bas = plus déterministe)
            top_p (float): Paramètre pour le nucleus sampling
        Returns:
            str: Texte généré
        """
        if self.model is None:
            return "Modèle non chargé. Veuillez charger un modèle d'abord."

        try:
            logger.info(f"Génération de réponse avec {self.model_name}, max_tokens={max_tokens}, temp={temperature}")
            # Générer une réponse avec plus de paramètres pour contrôler la qualité
            response = self.model(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,  # Pénaliser les répétitions
                top_k=40,  # Limiter le nombre de tokens considérés
                stop=["Question:", "Human:", "User:"]  # Arrêter à ces tokens
            )
            
            # Post-traitement pour nettoyer la réponse
            response = self._clean_response(response)
            
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            return f"Erreur lors de la génération: {e}"

    def _clean_response(self, text: str) -> str:
        """
        Nettoie la réponse générée
        """
        # Supprimer les lignes multiples vides
        text = '\n'.join([line for line in text.split('\n') if line.strip()])
        
        # Supprimer les textes instables à la fin
        stop_phrases = ["Question:", "Human:", "User:", "Assistant:", "AI:"]
        for phrase in stop_phrases:
            if phrase in text:
                text = text.split(phrase)[0]
                
        return text.strip()

    def create_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Créer un prompt amélioré pour le modèle en incluant le contexte
        Args:
            query (str): Requête de l'utilisateur
            context_chunks (List[Dict[str, Any]]): Chunks de contexte
        Returns:
            str: Prompt pour le modèle
        """
        # Trier les chunks par score si disponible
        if all('score' in chunk for chunk in context_chunks):
            context_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        # Construire le contexte à partir des chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_part = f"[Document {i+1}: {chunk['metadata']['filename']}]\n{chunk['text']}"
            context_parts.append(context_part)
            
        context = "\n\n".join(context_parts)
        
        # Construire un prompt amélioré et structuré selon le modèle utilisé
        if "mistral" in self.model_type.lower():
            # Format pour les modèles Mistral
            prompt = f"""<s>[INST] Tu es un assistant IA qui répond aux questions en te basant uniquement sur le contexte fourni.
Si tu ne trouves pas la réponse dans le contexte, dis-le clairement mais essaie d'être utile.

Contexte:
{context}

Question: {query} [/INST]

"""
        elif "llama" in self.model_type.lower():
            # Format pour les modèles LLaMA
            prompt = f"""<s>[INST]<<SYS>>
Tu es un assistant IA qui répond aux questions en te basant uniquement sur le contexte fourni.
Si tu ne trouves pas la réponse dans le contexte, dis-le clairement mais essaie d'être utile.
<</SYS>>

Voici le contexte pertinent extrait des documents :
---
{context}
---

Pour répondre, suis ces étapes :
1. Comprends bien la question : "{query}"
2. Analyse attentivement le contexte fourni
3. Formule une réponse claire, factuelle et concise basée uniquement sur ce contexte
4. Structure ta réponse en paragraphes cohérents si nécessaire

Question: {query}[/INST]

"""
        else:
            # Format générique pour les autres modèles
            prompt = f"""Tu es un assistant IA qui répond aux questions en te basant uniquement sur le contexte fourni.
Si tu ne trouves pas la réponse dans le contexte, dis-le clairement mais essaie d'être utile.

Voici le contexte pertinent extrait des documents :
---
{context}
---

Question: {query}

Réponse:"""
        
        logger.debug(f"Prompt créé: {prompt[:200]}...")
        return prompt