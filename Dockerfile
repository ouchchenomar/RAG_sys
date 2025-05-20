FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p /app/data/documents
RUN mkdir -p /app/data/chunks
RUN mkdir -p /app/data/indices
RUN mkdir -p /app/models

# Exposer le port
EXPOSE 8000

# Commande à exécuter
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]