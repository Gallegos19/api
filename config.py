"""
Configuración de la aplicación
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
DB_CONFIG = {
    'dbname': os.environ.get('DB_NAME', 'postgres'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'sslmode': 'require'  # Supabase requiere SSL
}

# Configuración Flask
FLASK_CONFIG = {
    'debug': os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
    'secret_key': os.environ.get('SECRET_KEY', 'dev-secret-key'),
    'host': os.environ.get('HOST', '0.0.0.0'),
    'port': int(os.environ.get('PORT', '8000'))
}

# Configuración del modelo
MODEL_CONFIG = {
    'n_estimators': int(os.environ.get('MODEL_N_ESTIMATORS', '50')),
    'max_depth': int(os.environ.get('MODEL_MAX_DEPTH', '5')),
    'min_samples_split': int(os.environ.get('MODEL_MIN_SAMPLES_SPLIT', '20')),
    'min_samples_leaf': int(os.environ.get('MODEL_MIN_SAMPLES_LEAF', '10')),
    'max_features': float(os.environ.get('MODEL_MAX_FEATURES', '0.5')),
    'random_state': int(os.environ.get('MODEL_RANDOM_STATE', '42'))
}

# Configuración de predicciones
PREDICTION_CONFIG = {
    'prediction_days': int(os.environ.get('PREDICTION_DAYS', '15')),
    'sequence_length': int(os.environ.get('SEQUENCE_LENGTH', '30')),
    'noise_level': float(os.environ.get('NOISE_LEVEL', '0.1'))
}

# Configuración de visualizaciones
VISUALIZATION_CONFIG = {
    'figure_size': (16, 12),
    'dpi': 300,
    'style': 'whitegrid',
    'font_size': 12
}