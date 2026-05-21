# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import uvicorn

from .services import (
    train_peer_tutoring_model,
    get_tutor_recommendations,
    get_best_matches,
    predict_pair_score,
    get_model_status,
    clear_ia_engine,
    get_all_engines
)

# ==================== MODELOS PYDANTIC ====================

class AlumnoNotas(BaseModel):
    """Modelo de alumno con sus notas"""
    c_usua: str
    nombre_completo: str
    email: Optional[str] = None
    notas: Dict[str, float]


class TrainRequest(BaseModel):
    """Request para entrenar modelo"""
    curso_id: str
    alumnos: List[AlumnoNotas]
    iterations: Optional[int] = 200


class PairScoreRequest(BaseModel):
    """Request para predecir score de un par"""
    curso_id: str
    tutor_id: str
    alumno_id: str


# ==================== CREAR APLICACIÓN FASTAPI ====================

app = FastAPI(
    title="KoEduko AI Backend",
    description="API para recomendación de peer tutoring usando redes neuronales ligeras",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "name": "KoEduko AI Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /api/ia/train",
            "GET /api/ia/recommendations/{curso_id}",
            "GET /api/ia/matches/{curso_id}",
            "POST /api/ia/predict-score",
            "GET /api/ia/status/{curso_id}",
            "DELETE /api/ia/clear/{curso_id}"
        ]
    }


@app.post("/api/ia/train")
async def train_model(request: TrainRequest):
    """Entrena modelo de IA para un curso específico"""
    try:
        alumnos_data = [alumno.dict() for alumno in request.alumnos]
        
        result = await train_peer_tutoring_model(
            curso_id=request.curso_id,
            alumnos_data=alumnos_data,
            iterations=request.iterations or 200
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ia/recommendations/{curso_id}")
async def recommendations(
    curso_id: str, 
    student_id: Optional[str] = None, 
    top_n: int = 10
):
    """Obtiene recomendaciones de tutores para un curso"""
    try:
        result = await get_tutor_recommendations(
            curso_id=curso_id,
            student_id=student_id,
            top_n=top_n
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ia/matches/{curso_id}")
async def best_matches(curso_id: str):
    """Obtiene los mejores emparejamientos para toda la clase"""
    try:
        result = await get_best_matches(curso_id=curso_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ia/predict-score")
async def predict_score(request: PairScoreRequest):
    """Predice score de compatibilidad entre tutor y alumno"""
    try:
        result = await predict_pair_score(
            curso_id=request.curso_id,
            tutor_id=request.tutor_id,
            alumno_id=request.alumno_id
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ia/status/{curso_id}")
async def model_status(curso_id: str):
    """Obtiene el estado del modelo para un curso"""
    try:
        result = await get_model_status(curso_id=curso_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/ia/clear/{curso_id}")
async def clear_model(curso_id: str):
    """Limpia el modelo entrenado para un curso"""
    try:
        clear_ia_engine(curso_id=curso_id)
        return {
            "status": "success",
            "curso_id": curso_id,
            "message": f"Modelo eliminado para curso {curso_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    engines = get_all_engines()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(engines),
        "cursos_activos": list(engines.keys())
    }


# ==================== EJECUCIÓN DIRECTA ====================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )