"""
API de Análisis de Churn - Xumaa
Arquitectura modular para análisis predictivo de abandono de usuarios
"""

__version__ = "1.0.0"
__author__ = "Xumaa Analytics Team"
__description__ = "API modular para análisis predictivo de churn con datos reales"

# Importaciones principales para facilitar el uso
from .churn_service import ChurnAnalysisService
from .database import DatabaseManager
from .data_processor import DataProcessor
from .models import ChurnModel, FuturePredictionModel
from .visualizations import VisualizationGenerator
from .recommendations import RecommendationEngine

__all__ = [
    'ChurnAnalysisService',
    'DatabaseManager', 
    'DataProcessor',
    'ChurnModel',
    'FuturePredictionModel',
    'VisualizationGenerator',
    'RecommendationEngine'
]