from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from app.api.endpoints import router as api_router
from app.api.worldbank_routes import worldbank_router
from app.logging_config import setup_logging

# Configurer le logging
loggers = setup_logging()
logger = loggers["app"]

# Créer les répertoires nécessaires
os.makedirs("./data/documents", exist_ok=True)
os.makedirs("./data/chunks", exist_ok=True)
os.makedirs("./data/indices", exist_ok=True)
os.makedirs("./data/cache", exist_ok=True)
os.makedirs("./models", exist_ok=True)

app = FastAPI(
    title="Système RAG",
    description="API pour un système de Retrieval-Augmented Generation",
    version="0.1.0"
)

# Configurer les CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configurer Jinja2Templates
templates = Jinja2Templates(directory="app/templates")

# Inclure les routes de l'API
app.include_router(api_router, prefix="/api")

# Inclure les routes pour la Banque Mondiale
app.include_router(worldbank_router, prefix="/api")

@app.get("/")
async def root(request: Request):
    logger.info("Page d'accueil demandée")
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    logger.info("Application démarrée")
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application arrêtée")

if __name__ == "__main__":
    logger.info("Démarrage du serveur Uvicorn")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)