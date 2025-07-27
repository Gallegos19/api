"""
Punto de entrada para Vercel - Versión simplificada
"""
import sys
import os

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar la aplicación Flask simplificada
from app_simple import app

# Vercel necesita que la app esté disponible como 'app'
if __name__ == "__main__":
    app.run()