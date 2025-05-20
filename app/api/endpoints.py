from fastapi import APIRouter, HTTPException
from typing import List, Optional
from ..document_processor.world_bank_processor import WorldBankProcessor

router = APIRouter()
world_bank_processor = WorldBankProcessor()

@router.get("/world-bank/indicators")
async def get_indicators():
    try:
        indicators = world_bank_processor.get_available_indicators()
        return {"indicators": indicators}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/world-bank/countries")
async def get_countries():
    try:
        countries = world_bank_processor.get_available_countries()
        return {"countries": countries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/world-bank/process")
async def process_world_bank_data(indicators: List[str], countries: Optional[List[str]] = None):
    try:
        documents = world_bank_processor.process_indicators(indicators, countries)
        return {"message": f"{len(documents)} documents traités avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))