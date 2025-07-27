#!/usr/bin/env python3
"""
Script para probar que las variables de entorno se están leyendo correctamente
"""
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def test_environment_variables():
    """Probar todas las variables de entorno necesarias"""
    print("🔧 Probando variables de entorno...")
    print("=" * 50)
    
    # Variables de Railway que vimos en la imagen
    railway_vars = {
        'FLASK_DEBUG': os.environ.get('FLASK_DEBUG'),
        'SECRET_KEY': os.environ.get('SECRET_KEY'),
        'DB_HOST': os.environ.get('DB_HOST'),
        'DB_PORT': os.environ.get('DB_PORT'),
        'DB_NAME': os.environ.get('DB_NAME'),
        'DB_USER': os.environ.get('DB_USER'),
        'DB_PASSWORD': os.environ.get('DB_PASSWORD'),
        'MODEL_RETRAIN_INTERVAL_HOURS': os.environ.get('MODEL_RETRAIN_INTERVAL_HOURS'),
        'PORT': os.environ.get('PORT')
    }
    
    print("Variables de Railway:")
    for var, value in railway_vars.items():
        status = "✅" if value else "❌"
        display_value = "***HIDDEN***" if 'PASSWORD' in var or 'SECRET' in var else value
        print(f"  {status} {var}: {display_value}")
    
    print("\n" + "=" * 50)
    
    # Probar configuración de base de datos
    try:
        from config import DB_CONFIG, FLASK_CONFIG
        print("📊 Configuración de base de datos:")
        for key, value in DB_CONFIG.items():
            display_value = "***HIDDEN***" if key == 'password' else value
            print(f"  - {key}: {display_value}")
        
        print("\n🌐 Configuración de Flask:")
        for key, value in FLASK_CONFIG.items():
            display_value = "***HIDDEN***" if 'secret' in key.lower() else value
            print(f"  - {key}: {display_value}")
            
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
    
    print("\n" + "=" * 50)
    
    # Probar conexión a base de datos
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        
        print("🔌 Probando conexión a base de datos...")
        if db.test_connection():
            print("✅ Conexión exitosa!")
        else:
            print("❌ Error de conexión")
            
    except Exception as e:
        print(f"❌ Error probando conexión: {e}")

if __name__ == '__main__':
    test_environment_variables()