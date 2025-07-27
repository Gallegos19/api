#!/usr/bin/env python3
"""
Script para probar la conexión directa a Supabase
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def test_supabase_connection():
    """Probar diferentes métodos de conexión a Supabase"""
    
    # Método 1: DATABASE_URL completa
    database_url = "postgresql://postgres:2axRmYdLPUdsA7AO@db.qcysdkgbewyrdrtdjekc.supabase.co:5432/postgres"
    
    print("🔧 Probando conexión a Supabase...")
    print("=" * 50)
    
    print("📡 Método 1: DATABASE_URL completa")
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Conexión exitosa!")
        print(f"   PostgreSQL version: {version[0][:50]}...")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📡 Método 2: Parámetros individuales")
    try:
        conn = psycopg2.connect(
            host="db.qcysdkgbewyrdrtdjekc.supabase.co",
            port=6543,
            database="postgres",
            user="postgres",
            password="2axRmYdLPUdsA7AO",
            sslmode="require"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT current_database();")
        db_name = cursor.fetchone()
        print(f"✅ Conexión exitosa!")
        print(f"   Base de datos: {db_name[0]}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📡 Método 3: Sin SSL")
    try:
        conn = psycopg2.connect(
            host="db.qcysdkgbewyrdrtdjekc.supabase.co",
            port=5432,
            database="postgres",
            user="postgres",
            password="2axRmYdLPUdsA7AO"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        print(f"✅ Conexión sin SSL exitosa!")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_supabase_connection()