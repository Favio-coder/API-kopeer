# app/services.py - Versión completa con todas las funciones

"""Servicios de IA para peer tutoring"""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

# Importar librería kopeer
from kopeer import PeerTutoringEngine, KopeerConfig, extract_notas_from_alumnos

# Singleton para el motor de IA por curso - DEFINIDO A NIVEL GLOBAL
_ia_engines: Dict[str, PeerTutoringEngine] = {}


def get_ia_engine(curso_id: str) -> PeerTutoringEngine:
    """Obtiene o crea instancia del motor de IA para un curso específico"""
    global _ia_engines
    if curso_id not in _ia_engines:
        # Configurar según los atributos activos (3 notas básicas)
        config = KopeerConfig(
            atributos=['conocimiento', 'actitud', 'participacion'],
            alpha=0.7,
            learning_rate=0.01,
            epochs=100,
            hidden_size=16
        )
        _ia_engines[curso_id] = PeerTutoringEngine(config=config)
        print(f"✅ Nuevo motor IA creado para curso {curso_id}")
    return _ia_engines[curso_id]


def clear_ia_engine(curso_id: str):
    """Limpia el motor de IA para un curso"""
    global _ia_engines
    if curso_id in _ia_engines:
        del _ia_engines[curso_id]
        print(f"🗑️ Motor IA eliminado para curso {curso_id}")


def get_all_engines() -> Dict[str, PeerTutoringEngine]:
    """Retorna todos los motores IA activos"""
    global _ia_engines
    return _ia_engines  # ← ¡ATENCIÓN! Aquí hay un typo: _ia_engines


async def train_peer_tutoring_model(
    curso_id: str, 
    alumnos_data: List[Dict[str, Any]],
    iterations: int = 200
) -> Dict[str, Any]:
    """Entrena el modelo de peer tutoring para un curso específico"""
    try:
        engine = get_ia_engine(curso_id)
        
        df = extract_notas_from_alumnos(alumnos_data)
        
        if df.empty:
            return {
                "status": "error",
                "message": "No hay datos de alumnos para entrenar",
                "curso_id": curso_id
            }
        
        engine.load_students(df, id_col='c_usua', nombre_col='nombre_completo')
        engine.train(iterations=iterations)
        
        return {
            "status": "success",
            "curso_id": curso_id,
            "n_estudiantes": len(alumnos_data),
            "n_atributos": len(engine.config.atributos),
            "iterations": iterations,
            "message": f"Modelo entrenado con {len(alumnos_data)} estudiantes"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "curso_id": curso_id
        }


async def get_tutor_recommendations(
    curso_id: str, 
    student_id: Optional[str] = None, 
    top_n: int = 10
) -> Dict[str, Any]:
    """Obtiene recomendaciones de tutores"""
    try:
        engine = get_ia_engine(curso_id)
        
        if student_id:
            recommendations = engine.get_recommendations(
                student_id=student_id, 
                top_n=top_n
            )
        else:
            recommendations = engine.get_recommendations(top_n=top_n)
        
        return {
            "status": "success",
            "curso_id": curso_id,
            "student_id": student_id,
            "total_recomendaciones": len(recommendations),
            "recomendaciones": recommendations
        }
        
    except ValueError as e:
        return {
            "status": "error",
            "message": "Modelo no entrenado. Ejecuta /train primero",
            "curso_id": curso_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "curso_id": curso_id
        }


async def get_best_matches(curso_id: str) -> Dict[str, Any]:
    """Obtiene los mejores emparejamientos para toda la clase"""
    try:
        engine = get_ia_engine(curso_id)
        matches = engine.get_best_match_for_class()
        
        return {
            "status": "success",
            "curso_id": curso_id,
            "total_parejas": len(matches),
            "emparejamientos": matches
        }
        
    except ValueError as e:
        return {
            "status": "error",
            "message": "Modelo no entrenado. Ejecuta /train primero",
            "curso_id": curso_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "curso_id": curso_id
        }


async def predict_pair_score(
    curso_id: str,
    tutor_id: str,
    alumno_id: str
) -> Dict[str, Any]:
    """Predice el score de compatibilidad entre un tutor y un alumno específicos"""
    try:
        engine = get_ia_engine(curso_id)
        
        tutor_profile = None
        alumno_profile = None
        
        for student in engine.students:
            if student.id == tutor_id:
                tutor_profile = student
            if student.id == alumno_id:
                alumno_profile = student
        
        if not tutor_profile or not alumno_profile:
            return {
                "status": "error",
                "message": "No se encontraron los perfiles de los estudiantes",
                "curso_id": curso_id
            }
        
        score = engine.model.predict_score(tutor_profile, alumno_profile)
        
        return {
            "status": "success",
            "curso_id": curso_id,
            "tutor_id": tutor_id,
            "alumno_id": alumno_id,
            "score": score,
            "compatibilidad": "Alta" if score > 0.7 else "Media" if score > 0.4 else "Baja"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "curso_id": curso_id
        }


async def get_model_status(curso_id: str) -> Dict[str, Any]:
    """Obtiene el estado del modelo para un curso"""
    global _ia_engines
    
    if curso_id not in _ia_engines:
        return {
            "status": "not_trained",
            "curso_id": curso_id,
            "message": "Modelo no entrenado para este curso"
        }
    
    engine = _ia_engines[curso_id]
    
    return {
        "status": "trained",
        "curso_id": curso_id,
        "n_estudiantes": len(engine.students),
        "atributos": engine.config.atributos,
        "modelo_listo": engine.model is not None and hasattr(engine.model, '_is_trained') and engine.model._is_trained
    }