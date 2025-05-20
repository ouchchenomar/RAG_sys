import uvicorn
from app.logging_config import setup_logging

# Configurer le logging
loggers = setup_logging()
logger = loggers["app"]

if __name__ == "__main__":
    logger.info("Démarrage du serveur RAG")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)