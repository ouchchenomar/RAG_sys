# Système RAG (Retrieval-Augmented Generation)

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)

Un système de Retrieval-Augmented Generation (RAG) complet qui améliore les modèles de langage en les connectant à des sources de données externes. Ce projet combine une indexation vectorielle avancée avec un LLM local ou l'API OpenAI pour fournir des réponses précises et contextuelles.

## ✨ Fonctionnalités

- 📄 **Gestion intelligente des documents** - Supporte les formats TXT et PDF avec chunking optimisé
- 🔎 **Recherche sémantique** - Indexation vectorielle avec TF-IDF ou Sentence Transformers
- 🧠 **Modèles de langage flexibles** - Fonctionne avec des modèles GGUF locaux (TinyLlama, Llama, Mistral) ou l'API OpenAI
- 🔄 **API REST complète** - Endpoints pour toutes les fonctionnalités via FastAPI
- 🖥️ **Interface utilisateur web** - Interface intuitive pour interagir avec le système

## 🏗️ Architecture

Le système est construit avec une architecture modulaire :

```
RAGSystem/
├── app/
│   ├── static/              # Fichiers statiques (CSS, JS)
│   ├── templates/           # Templates HTML
│   ├── document_processor/  # Traitement des documents
│   ├── vector_store/        # Indexation et recherche vectorielle
│   ├── llm/                 # Interface avec les modèles de langage
│   ├── rag_pipeline/        # Pipeline RAG complet
│   ├── api/                 # Endpoints API
│   └── main.py              # Point d'entrée principal
├── data/                    # Données et index
├── models/                  # Modèles de langage
├── logs/                    # Fichiers de log
└── run_api.py               # Script de démarrage
```

## 📦 Fichiers volumineux non inclus

⚠️ **Attention** : Certains fichiers volumineux, notamment les modèles comme `models/tinyllama.gguf`, **ne sont pas inclus dans ce dépôt** car ils dépassent la limite de taille imposée par GitHub (100 Mo par fichier).

Si vous souhaitez utiliser ce projet, veuillez :

1. Télécharger le fichier `tinyllama.gguf` depuis [Hugging Face](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
2. Le placer manuellement dans le répertoire suivant du projet :
   ```
   models/tinyllama.gguf
   ```

❗ Assurez-vous que ce fichier est présent pour que le système fonctionne correctement avec des modèles locaux.

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- Pip (gestionnaire de paquets Python)
- 4 Go RAM minimum (8 Go recommandé)

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-nom/RAGSystem.git
   cd RAGSystem
   ```

2. **Installer les dépendances**
   ```bash
   python install_dependencies.py
   ```

3. **Configuration pour l'API OpenAI (Optionnel)**
   
   Créez un fichier `.env` à la racine du projet :
   ```
   OPENAI_API_KEY=votre-clé-api
   ```

4. **Télécharger un modèle LLM local (pour l'utilisation sans OpenAI)**
   ```bash
   python download_model.py
   ```

## 💻 Utilisation

### Démarrer le serveur

```bash
python run_api.py
```

Accédez à l'interface via votre navigateur à l'adresse : http://127.0.0.1:8000

### Utilisation du système RAG

1. **Télécharger des documents** : Utilisez l'interface web pour ajouter des documents à votre base de connaissances.
2. **Poser des questions** : Entrez vos questions dans la zone de chat pour obtenir des réponses basées sur vos documents.
3. **Explorer les sources** : Chaque réponse affiche les sources utilisées pour la génération, avec des scores de pertinence.

## 🔍 Caractéristiques détaillées

### Module Document Processor

- Chargement et traitement de fichiers TXT et PDF
- Découpage intelligent en chunks avec chevauchement
- Gestion des métadonnées et organisation hiérarchique

### Module Vector Store

- Support pour TF-IDF et Sentence Transformers
- Indexation vectorielle optimisée
- Recherche par similarité cosinus avec seuil de confiance

### Module LLM

- Support pour les modèles GGUF (Llama, TinyLlama, Mistral, etc.)
- Intégration avec l'API OpenAI
- Génération de prompts optimisés selon le type de modèle

## 🌐 Technologies utilisées

- **FastAPI** : Framework API web rapide et moderne
- **scikit-learn / FAISS** : Bibliothèques d'indexation vectorielle
- **ctransformers** : Interface pour les modèles de langage locaux
- **OpenAI API** : Pour l'accès à des modèles de langage avancés
- **Bootstrap** : Framework CSS pour l'interface utilisateur

## 📜 Utilisation et Attribution

Ce code est libre d'utilisation. Toutefois, si vous utilisez ou adaptez ce projet, merci de mentionner l'auteur original:

**Développé par Omar Ouchchen**

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request pour améliorer ce projet.

## 📞 Contact

Pour toute question ou suggestion, veuillez me contacter à [ouchcheno@gmail.com](mailto:ouchcheno@gmail.com).
