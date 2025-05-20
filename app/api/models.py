from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Modèle pour les requêtes de l'utilisateur"""
    question: str
    top_k: int = 3
    max_tokens: int = 256

class QueryResponse(BaseModel):
    """Modèle pour les réponses aux requêtes"""
    success: bool
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

class SystemInfo(BaseModel):
    """Modèle pour les informations sur le système"""
    success: bool
    document_count: int
    chunk_count: int
    model_loaded: bool
    model_name: Optional[str] = None

class DocumentResponse(BaseModel):
    """Modèle pour les réponses à l'ajout de documents"""
    success: bool
    doc_id: Optional[str] = None
    message: str