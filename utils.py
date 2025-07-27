"""
Utilidades para la API de análisis de churn
"""
import numpy as np
import pandas as pd
from datetime import datetime, date


def convert_to_json_serializable(obj):
    """
    Convertir objetos de NumPy/Pandas a tipos serializables en JSON
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj


def safe_json_response(data):
    """
    Preparar datos para respuesta JSON segura
    """
    return convert_to_json_serializable(data)


def format_user_recommendations(recommendations):
    """
    Formatear recomendaciones para salida consistente
    """
    formatted = []
    for rec in recommendations:
        formatted_rec = {
            'user_id': str(rec['user_id']),
            'current_churned_status': int(rec['current_churned_status']),
            'avg_churn_risk': float(rec['avg_churn_risk']),
            'max_churn_risk': float(rec['max_churn_risk']),
            'days_high_risk': int(rec['days_high_risk']),
            'avg_engagement': float(rec['avg_engagement']),
            'engagement_trend': float(rec['engagement_trend']),
            'recommendations': list(rec['recommendations'])
        }
        formatted.append(formatted_rec)
    return formatted


def format_model_metrics(metrics):
    """
    Formatear métricas del modelo para JSON
    """
    if not metrics:
        return {}
    
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.float64, np.float32)):
            formatted[key] = float(value)
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            formatted[key] = int(value)
        elif isinstance(value, list):
            formatted[key] = [convert_to_json_serializable(item) for item in value]
        else:
            formatted[key] = convert_to_json_serializable(value)
    
    return formatted


def format_data_statistics(stats):
    """
    Formatear estadísticas de datos para JSON
    """
    if not stats:
        return {}
    
    formatted = {}
    for key, value in stats.items():
        if key == 'churn_distribution' and isinstance(value, dict):
            # Convertir claves y valores a tipos nativos
            formatted[key] = {str(k): int(v) for k, v in value.items()}
        elif key == 'null_counts' and isinstance(value, dict):
            formatted[key] = {str(k): int(v) for k, v in value.items()}
        elif key == 'numeric_stats' and isinstance(value, dict):
            formatted[key] = convert_to_json_serializable(value)
        else:
            formatted[key] = convert_to_json_serializable(value)
    
    return formatted