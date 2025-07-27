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
            'DATABASE_URL': '***' if os.environ.get('DATABASE_URL') else None,
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
        
        # Verificar si está en modo demo (sin base de datos)
        demo_mode = os.environ.get('DEMO_MODE', 'false').lower() == 'true'
        
        if demo_mode:
            logger.info("🎭 Iniciando en MODO DEMO (sin base de datos)")
        else:
            # Verificar variables de entorno críticas
            required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST']
            missing_vars = [var for var in required_vars if not os.environ.get(var) and not os.environ.get('DATABASE_URL')]
            
            if missing_vars and not os.environ.get('DATABASE_URL'):
                logger.error(f"❌ Variables de entorno faltantes: {missing_vars}")
                logger.info("💡 Tip: Agrega DEMO_MODE=true para ejecutar sin base de datos")
                sys.exit(1)
            
            logger.info("✅ Variables de entorno configuradas correctamente")
            
            # Probar conexión a base de datos con diagnóstico detallado
            try:
                from database import DatabaseManager
                from config import DB_CONFIG
                
                logger.info("🔌 Intentando conectar a base de datos...")
                logger.info(f"   Host: {os.environ.get('DB_HOST')}")
                logger.info(f"   Puerto: {os.environ.get('DB_PORT')}")
                logger.info(f"   Base de datos: {os.environ.get('DB_NAME')}")
                logger.info(f"   Usuario: {os.environ.get('DB_USER')}")
                
                # Mostrar el tipo de configuración que se está usando
                if isinstance(DB_CONFIG, str):
                    logger.info(f"   Tipo de conexión: DATABASE_URL string")
                    # Mostrar la URL sin la contraseña
                    safe_url = DB_CONFIG.replace(os.environ.get('DB_PASSWORD', ''), '***')
                    logger.info(f"   URL (segura): {safe_url}")
                else:
                    logger.info(f"   Tipo de conexión: Parámetros individuales")
                    logger.info(f"   SSL Mode: {DB_CONFIG.get('sslmode', 'no especificado')}")
                
                db = DatabaseManager()
                
                # Intentar conexión con más detalles del error
                try:
                    conn = db.connect()
                    if conn:
                        # Probar una consulta simple
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1;")
                        result = cursor.fetchone()
                        cursor.close()
                        conn.close()
                        
                        if result and result[0] == 1:
                            logger.info("✅ Conexión a base de datos exitosa")
                        else:
                            logger.warning("⚠️ Conexión establecida pero consulta falló")
                            logger.warning("🔄 Continuando en modo degradado...")
                    else:
                        logger.warning("⚠️ No se pudo establecer conexión")
                        logger.warning("🔄 Continuando en modo degradado...")
                        
                except Exception as conn_error:
                    logger.error(f"❌ Error específico de conexión: {conn_error}")
                    logger.error(f"   Tipo de error: {type(conn_error).__name__}")
                    
                    # Diagnóstico específico para errores comunes
                    error_str = str(conn_error).lower()
                    if 'timeout' in error_str:
                        logger.error("💡 Diagnóstico: Timeout de conexión - posible problema de red o IPv6")
                    elif 'connection refused' in error_str:
                        logger.error("💡 Diagnóstico: Conexión rechazada - verificar host y puerto")
                    elif 'authentication failed' in error_str:
                        logger.error("💡 Diagnóstico: Fallo de autenticación - verificar usuario/contraseña")
                    elif 'ssl' in error_str:
                        logger.error("💡 Diagnóstico: Problema SSL - verificar configuración SSL")
                    elif 'name resolution' in error_str or 'getaddrinfo' in error_str:
                        logger.error("💡 Diagnóstico: Problema de resolución DNS - posible problema IPv6")
                    
                    logger.warning("🔄 Continuando en modo degradado sin base de datos...")
                    
            except Exception as db_error:
                logger.error(f"❌ Error general probando base de datos: {db_error}")
                logger.error(f"   Tipo de error: {type(db_error).__name__}")
                logger.warning("🔄 Continuando en modo degradado sin base de datos...")
                # No salir, continuar sin base de datos
        
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