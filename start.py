#!/usr/bin/env python3
"""
Script de inicio para la API de Análisis de Churn
"""
import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Verificar que el entorno esté configurado correctamente"""
    print("🔍 Verificando configuración del entorno...")
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Variables requeridas
    required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Variables de entorno faltantes: {', '.join(missing_vars)}")
        print("💡 Asegúrate de tener un archivo .env con las variables necesarias")
        return False
    
    print("✅ Variables de entorno configuradas correctamente")
    return True

def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    print("📦 Verificando dependencias...")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'psycopg2', 'scikit-learn', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Paquetes faltantes: {', '.join(missing_packages)}")
        print("💡 Instala las dependencias con: pip install -r requirements.txt")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def test_database_connection():
    """Probar conexión a la base de datos"""
    print("🗄️ Probando conexión a la base de datos...")
    
    try:
        from database import DatabaseManager
        db_manager = DatabaseManager()
        
        if db_manager.test_connection():
            print("✅ Conexión a la base de datos exitosa")
            return True
        else:
            print("❌ No se pudo conectar a la base de datos")
            return False
    except Exception as e:
        print(f"❌ Error probando conexión: {e}")
        return False

def start_api():
    """Iniciar la API"""
    print("🚀 Iniciando API de Análisis de Churn...")
    
    try:
        from app import app
        
        print("📡 Endpoints disponibles:")
        print("   GET  /api/churn-analysis/health")
        print("   POST /api/churn-analysis/full-analysis")
        print("   GET  /api/churn-analysis/predictions")
        print("   GET  /api/churn-analysis/visualizations")
        print("   GET  /api/churn-analysis/recommendations")
        print("   GET  /api/churn-analysis/status")
        print("🌐 Servidor ejecutándose en: http://localhost:5001")
        print("🧪 Prueba la API con: python test_api.py")
        print("-" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except Exception as e:
        print(f"❌ Error iniciando la API: {e}")
        sys.exit(1)

def main():
    """Función principal"""
    print("=" * 60)
    print("🎯 API DE ANÁLISIS DE CHURN - XUMAA")
    print("   Arquitectura Modular v1.0.0")
    print("=" * 60)
    
    # Verificaciones previas
    if not check_environment():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    if not test_database_connection():
        print("⚠️ Continuando sin conexión a BD (algunos endpoints fallarán)")
    
    # Iniciar API
    start_api()

if __name__ == "__main__":
    main()