#!/usr/bin/env python3
"""
Script de inicio optimizado para Railway
"""
import os
import sys
import logging
from app import app

# Configurar logging para producción
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Función principal para Railway"""
    try:
        # Configuración para Railway
        port = int(os.environ.get('PORT', 5001))
        host = '0.0.0.0'
        
        logger.info("🚀 Iniciando API de Análisis de Churn en Railway")
        logger.info(f"🌐 Puerto: {port}")
        logger.info(f"🔧 Host: {host}")
        
        # Mostrar variables de entorno para debug (sin mostrar passwords)
        env_vars = {
            'PORT': os.environ.get('PORT'),
            'DB_HOST': os.environ.get('DB_HOST'),
            'DB_NAME': os.environ.get('DB_NAME'),
            'DB_USER': os.environ.get('DB_USER'),
            'DB_PORT': os.environ.get('DB_PORT'),
            'SECRET_KEY': '***' if os.environ.get('SECRET_KEY') else None,
            'FLASK_DEBUG': os.environ.get('FLASK_DEBUG'),
            'MODEL_RETRAIN_INTERVAL_HOURS': os.environ.get('MODEL_RETRAIN_INTERVAL_HOURS')
        }
        
        logger.info("🔧 Variables de entorno detectadas:")
        for key, value in env_vars.items():
            status = "✅" if value else "❌"
            logger.info(f"   {status} {key}: {value}")
        
        # Verificar variables de entorno críticas
        required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error(f"❌ Variables de entorno faltantes: {missing_vars}")
            sys.exit(1)
        
        logger.info("✅ Variables de entorno configuradas correctamente")
        
        # Probar conexión a base de datos
        try:
            from database import DatabaseManager
            db = DatabaseManager()
            if db.test_connection():
                logger.info("✅ Conexión a base de datos exitosa")
            else:
                logger.error("❌ Error de conexión a base de datos")
                sys.exit(1)
        except Exception as db_error:
            logger.error(f"❌ Error probando base de datos: {db_error}")
            sys.exit(1)
        
        # Iniciar aplicación
        logger.info("🎯 Iniciando servidor Flask...")
        app.run(
            debug=False,
            host=host,
            port=port,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Error iniciando aplicación: {e}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()