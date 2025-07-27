#!/usr/bin/env python3
"""
Script para probar la conexión IPv6 con Supabase
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def test_connection_methods():
    """Probar diferentes métodos de conexión"""
    
    # Método 1: DATABASE_URL con pooler (puerto 6543)
    database_url = os.environ.get('DATABASE_URL')
    if database_url and ':6543/' in database_url:
        database_url_pooled = database_url.replace(':5432/', ':6543/')
    else:
        database_url_pooled = database_url
    
    print("🔍 Probando conexiones a Supabase...")
    print(f"Host: {os.environ.get('DB_HOST')}")
    print(f"Usuario: {os.environ.get('DB_USER')}")
    print(f"Base de datos: {os.environ.get('DB_NAME')}")
    print()
    
    # Test 1: Pooler con parámetros IPv6
    print("1️⃣ Probando conexión con pooler (puerto 6543) + parámetros IPv6...")
    try:
        conn_string = f"{database_url_pooled}?sslmode=require&connect_timeout=60&statement_timeout=300000&idle_in_transaction_session_timeout=300000"
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ ÉXITO: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Conexión directa (puerto 5432)
    print("\n2️⃣ Probando conexión directa (puerto 5432)...")
    try:
        conn_string = f"{database_url}?sslmode=require&connect_timeout=60"
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ ÉXITO: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Parámetros individuales con pooler
    print("\n3️⃣ Probando con parámetros individuales (pooler)...")
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST'),
            port=6543,
            database=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD'),
            sslmode='require',
            connect_timeout=60,
            application_name='test_ipv6_connection'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ ÉXITO: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Parámetros individuales directo
    print("\n4️⃣ Probando con parámetros individuales (directo)...")
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST'),
            port=6543,
            database=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD'),
            sslmode='require',
            connect_timeout=60,
            application_name='test_ipv6_connection'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ ÉXITO: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

def main():
    print("🧪 Test de Conexión IPv6 para Supabase en Railway")
    print("=" * 50)
    
    # Verificar variables de entorno
    required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Variables de entorno faltantes: {missing_vars}")
        sys.exit(1)
    
    success = test_connection_methods()
    
    if success:
        print("\n🎉 Al menos una conexión fue exitosa!")
        print("💡 Usa el método que funcionó en tu aplicación.")
    else:
        print("\n😞 Ninguna conexión fue exitosa.")
        print("💡 Verifica tus credenciales y la conectividad de red.")
        sys.exit(1)

if __name__ == '__main__':
    main()