#!/bin/bash

# Variables
PROJECT_ID="votre-projet-gcp"
IMAGE_NAME="rag-system"
REGION="europe-west1"
SERVICE_NAME="rag-api"

# Construire l'image Docker
echo "Construction de l'image Docker..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Pousser l'image vers Google Container Registry
echo "Envoi de l'image vers Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Déployer sur Google Cloud Run
echo "Déploiement sur Google Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi

echo "Déploiement terminé!"