# app/api/worldbank_routes.py
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any, Optional
from ..worldbank.rag_processor import WorldBankRAGProcessor
import logging

logger = logging.getLogger(__name__)

worldbank_router = APIRouter(prefix="/worldbank", tags=["Banque Mondiale"])

# Instance partagée du processeur RAG pour la Banque Mondiale
worldbank_processor = None

def get_wb_processor():
    """
    Obtenir une instance du processeur RAG pour la Banque Mondiale
    """
    global worldbank_processor
    if worldbank_processor is None:
        try:
            worldbank_processor = WorldBankRAGProcessor()
        except Exception as e:
            logger.error(f"Erreur lors de la création du processeur RAG pour la Banque Mondiale: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de l'initialisation du processeur RAG: {e}"
            )
    return worldbank_processor

@worldbank_router.get("/system-info/")
async def get_system_info(wb_processor: WorldBankRAGProcessor = Depends(get_wb_processor)):
    """
    Obtenir des informations sur l'état du système RAG pour la Banque Mondiale
    """
    try:
        info = wb_processor.get_worldbank_info()
        return info
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations système: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des informations système: {e}"
        )

@worldbank_router.post("/update-knowledge/")
async def update_knowledge(
    request: Request,
    wb_processor: WorldBankRAGProcessor = Depends(get_wb_processor)
):
    """
    Mettre à jour la base de connaissances avec les données de la Banque Mondiale
    """
    try:
        # Récupérer le corps de la requête
        body = await request.json()
        
        # Extraire les paramètres de la requête
        country_codes = body.get("countries", [])
        indicator_ids = body.get("indicators", [])
        include_topics = body.get("include_topics", True)
        
        logger.info(f"Mise à jour de la base de connaissances avec {len(country_codes)} pays et {len(indicator_ids)} indicateurs")
        
        # Effectuer la mise à jour
        result = wb_processor.update_worldbank_knowledge(
            country_codes=country_codes,
            indicator_ids=indicator_ids,
            include_topics=include_topics
        )
        
        if not result["success"]:
            logger.error(f"Erreur lors de la mise à jour de la base de connaissances: {result['message']}")
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la base de connaissances: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la mise à jour de la base de connaissances: {e}"
        )

@worldbank_router.post("/query/")
async def query(
    request: Request,
    wb_processor: WorldBankRAGProcessor = Depends(get_wb_processor)
):
    """
    Interroger le système RAG de la Banque Mondiale
    """
    try:
        # Récupérer le corps de la requête
        body = await request.json()
        
        # Extraire les paramètres
        question = body.get("question", "")
        top_k = body.get("top_k", 3)
        max_tokens = body.get("max_tokens", 512)
        
        if not question:
            raise HTTPException(
                status_code=400,
                detail="La question ne peut pas être vide"
            )
        
        logger.info(f"Requête au système RAG: '{question}'")
        
        # Effectuer la requête
        result = wb_processor.query(question, top_k, max_tokens)
        
        if not result["success"]:
            logger.error(f"Erreur lors de la requête: {result.get('message', 'Erreur inconnue')}")
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Erreur lors de la requête")
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la requête: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la requête: {e}"
        )