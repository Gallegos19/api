"""
Configuración específica para Railway con soporte IPv6 optimizado
"""
import os
import psycopg2
from urllib.parse import urlparse

def get_railway_db_config():
    """
    Obtener configuración de base de datos optimizada para Railway + Supabase IPv6
    """
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        # Fallback a parámetros individuales
        return {
            'host': os.environ.get('DB_HOST'),
            'port': int(os.environ.get('DB_PORT', 6543)),
            'database': os.environ.get('DB_NAME'),
            'user': os.environ.get('DB_USER'),
            'password': os.environ.get('DB_PASSWORD'),
            'sslmode': 'require',
            'connect_timeout': 60,
            'application_name': 'xumaa_analytics_railway'
        }
    
    # Parsear la DATABASE_URL
    parsed = urlparse(database_url)
    
    # Asegurar que usa el puerto 6543 (pooler)
    port = 6543 if parsed.port == 5432 else (parsed.port or 6543)
    
    # Construir string de conexión optimizado para IPv6 (solo parámetros válidos para psycopg2)
    optimized_url = (
        f"postgresql://{parsed.username}:{parsed.password}@"
        f"{parsed.hostname}:{port}{parsed.path}"
        f"?sslmode=require"
        f"&connect_timeout=60"
        f"&application_name=xumaa_analytics_railway"
    )
    
    # Usar parámetros individuales (más compatible)
    return {
        'host': parsed.hostname,
        'port': port,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password,
        'sslmode': 'require',
        'connect_timeout': 60,
        'application_name': 'xumaa_analytics_railway'
    }
    
    # return optimized_url

def test_railway_connection():
    """
    Probar conexión específicamente optimizada para Railway
    """
    config = get_railway_db_config()
    
    try:
        if isinstance(config, str):
            conn = psycopg2.connect(config)
        else:
            conn = psycopg2.connect(**config)
        
        # Probar consulta simple
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return True, f"Conexión exitosa: {version[0][:50]}..."
        
    except Exception as e:
        return False, str(e)

if __name__ == '__main__':
    # Test directo
    success, message = test_railway_connection()
    print(f"{'✅' if success else '❌'} {message}")