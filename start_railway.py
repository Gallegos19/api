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
        
        # Verificar variables de entorno críticas
        required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error(f"❌ Variables de entorno faltantes: {missing_vars}")
            sys.exit(1)
        
        logger.info("✅ Variables de entorno configuradas")
        
        # Iniciar aplicación
        app.run(
            debug=False,
            host=host,
            port=port,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Error iniciando aplicación: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()