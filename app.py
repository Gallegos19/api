"""
API Principal de Análisis de Churn - Xumaa
"""
from flask import Flask, jsonify, request
from config import FLASK_CONFIG
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Crear aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_CONFIG['secret_key']

# Importar el servicio de churn con logging detallado
print("🔄 Iniciando carga de servicios de churn...")

try:
    print("🔄 Intentando cargar ChurnAnalysisService completo...")
    from churn_service import ChurnAnalysisService
    print("✅ Módulo churn_service importado")
    
    print("🔄 Inicializando ChurnAnalysisService...")
    churn_service = ChurnAnalysisService()
    print("✅ ChurnAnalysisService inicializado")
    
    SERVICE_AVAILABLE = True
    SERVICE_TYPE = "full"
    print("✅ ChurnAnalysisService completo cargado exitosamente")
    
except Exception as e:
    print(f"⚠️ Warning: ChurnAnalysisService completo falló: {e}")
    print(f"📋 Tipo de error: {type(e).__name__}")
    import traceback
    print(f"📋 Traceback: {traceback.format_exc()}")
    
    try:
        print("🔄 Intentando cargar ChurnAnalysisServiceSimple...")
        from churn_service_simple import ChurnAnalysisServiceSimple
        print("✅ Módulo churn_service_simple importado")
        
        print("🔄 Inicializando ChurnAnalysisServiceSimple...")
        churn_service = ChurnAnalysisServiceSimple()
        print("✅ ChurnAnalysisServiceSimple inicializado")
        
        SERVICE_AVAILABLE = True
        SERVICE_TYPE = "demo"
        print("✅ ChurnAnalysisService demo cargado exitosamente")
        
    except Exception as e2:
        print(f"❌ Error: ChurnAnalysisServiceSimple también falló: {e2}")
        print(f"📋 Tipo de error: {type(e2).__name__}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        
        churn_service = None
        SERVICE_AVAILABLE = False
        SERVICE_TYPE = "none"
        print("❌ Ningún servicio de churn disponible")

print(f"🎯 Estado final: SERVICE_AVAILABLE={SERVICE_AVAILABLE}, SERVICE_TYPE={SERVICE_TYPE}")


@app.route('/api/churn-analysis/health', methods=['GET'])
def health_check():
    """Verificar el estado de la API"""
    try:
        return jsonify({
            'success': True,
            'status': 'healthy',
            'service': 'Analytics API - Xumaa',
            'version': '1.0.0',
            'timestamp': pd.Timestamp.now().isoformat(),
            'churn_service_available': SERVICE_AVAILABLE,
            'service_type': SERVICE_TYPE,
            'database_connected': SERVICE_AVAILABLE and SERVICE_TYPE == "full",
            'demo_mode': SERVICE_TYPE == "demo",
            'message': f'API funcionando correctamente ({SERVICE_TYPE} service)'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error en verificación de salud'
        }), 500


@app.route('/api/churn-analysis/full-analysis', methods=['POST'])
def full_churn_analysis():
    """
    Endpoint principal que ejecuta el análisis completo de churn
    Retorna: JSON con resultados + imágenes en base64
    """
    print("🔄 [ENDPOINT] full-analysis: Request recibida")
    print(f"🔄 [ENDPOINT] SERVICE_AVAILABLE: {SERVICE_AVAILABLE}")
    print(f"🔄 [ENDPOINT] SERVICE_TYPE: {SERVICE_TYPE}")
    
    if not SERVICE_AVAILABLE:
        print("❌ [ENDPOINT] Servicio no disponible")
        return jsonify({
            'success': False,
            'error': 'Servicio no disponible',
            'message': 'El servicio de análisis de churn no está disponible. Verifique la conexión a la base de datos.'
        }), 503
    
    try:
        print("🔄 [ENDPOINT] Iniciando análisis completo...")
        # Agregar timeout y manejo de errores más robusto
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Análisis tardó demasiado tiempo")
        
        # Configurar timeout de 30 segundos
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            results = churn_service.run_complete_analysis()
            signal.alarm(0)  # Cancelar timeout
            
            if results and results.get('success'):
                return jsonify({
                    'success': True,
                    'data': results['data'],
                    'service_type': SERVICE_TYPE,
                    'message': f'Análisis de churn completado exitosamente ({SERVICE_TYPE} service)'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': results.get('error', 'Error desconocido') if results else 'No se obtuvieron resultados',
                    'message': results.get('message', 'Error durante el análisis') if results else 'Error durante el análisis'
                }), 500
                
        except TimeoutError:
            signal.alarm(0)
            return jsonify({
                'success': False,
                'error': 'Timeout',
                'message': 'El análisis tardó demasiado tiempo. Intente nuevamente.'
            }), 504
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'service_type': SERVICE_TYPE,
            'message': f'Error durante el análisis de churn: {str(e)}'
        }), 500


@app.route('/api/churn-analysis/predictions', methods=['GET'])
def get_predictions():
    """Obtener predicciones de usuarios de alto riesgo"""
    print("🔄 [ENDPOINT] predictions: Request recibida")
    print(f"🔄 [ENDPOINT] SERVICE_AVAILABLE: {SERVICE_AVAILABLE}")
    
    if not SERVICE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Servicio no disponible',
            'message': 'El servicio de análisis de churn no está disponible. Verifique la conexión a la base de datos.'
        }), 503
    
    try:
        predictions_data = churn_service.get_high_risk_predictions()
        
        return jsonify({
            'success': True,
            'data': predictions_data,
            'message': 'Predicciones obtenidas exitosamente'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error obteniendo predicciones'
        }), 500


@app.route('/api/churn-analysis/visualizations', methods=['GET'])
def get_visualizations():
    """Obtener visualizaciones como imágenes en base64"""
    print("🔄 [ENDPOINT] visualizations: Request recibida")
    print(f"🔄 [ENDPOINT] SERVICE_AVAILABLE: {SERVICE_AVAILABLE}")
    
    if not SERVICE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Servicio no disponible',
            'message': 'El servicio de análisis de churn no está disponible. Verifique la conexión a la base de datos.'
        }), 503
    
    try:
        visualizations = churn_service.get_visualizations_only()
        
        return jsonify({
            'success': True,
            'data': {
                'images': visualizations
            },
            'message': 'Visualizaciones generadas exitosamente'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error generando visualizaciones'
        }), 500


@app.route('/api/churn-analysis/recommendations', methods=['GET'])
def get_recommendations():
    """Obtener recomendaciones de intervención"""
    if not SERVICE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Servicio no disponible',
            'message': 'El servicio de análisis de churn no está disponible. Verifique la conexión a la base de datos.'
        }), 503
    
    try:
        recommendations_data = churn_service.get_recommendations_only()
        
        return jsonify({
            'success': True,
            'data': recommendations_data,
            'message': 'Recomendaciones generadas exitosamente'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error generando recomendaciones'
        }), 500


@app.route('/api/churn-analysis/detailed-report', methods=['GET'])
def get_detailed_report():
    """
    Obtener reporte detallado con los 5 usuarios de la gráfica y resumen completo
    """
    if not SERVICE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Servicio no disponible',
            'message': 'El servicio de análisis de churn no está disponible. Verifique la conexión a la base de datos.'
        }), 503
    
    try:
        detailed_data = churn_service.get_detailed_report()
        
        return jsonify({
            'success': True,
            'data': detailed_data,
            'message': 'Reporte detallado generado exitosamente'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error generando reporte detallado'
        }), 500


@app.route('/api/churn-analysis/status', methods=['GET'])
def get_status():
    """Obtener estado detallado del servicio"""
    try:
        if SERVICE_AVAILABLE:
            status = churn_service.health_check()
        else:
            status = {
                'status': 'degraded',
                'database': 'disconnected',
                'model_trained': False,
                'future_model_trained': False
            }
        
        # Información adicional sobre el estado
        status_info = {
            **status,
            'service_available': SERVICE_AVAILABLE,
            'endpoints': {
                'health': '/api/churn-analysis/health',
                'full_analysis': '/api/churn-analysis/full-analysis',
                'predictions': '/api/churn-analysis/predictions',
                'visualizations': '/api/churn-analysis/visualizations',
                'recommendations': '/api/churn-analysis/recommendations',
                'detailed_report': '/api/churn-analysis/detailed-report'
            },
            'version': '1.0.0',
            'description': 'API de Análisis Predictivo de Churn - Xumaa'
        }
        
        return jsonify({
            'success': True,
            'data': status_info,
            'message': 'Estado del servicio obtenido exitosamente'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error obteniendo estado del servicio'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Manejar rutas no encontradas"""
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado',
        'message': 'La ruta solicitada no existe',
        'available_endpoints': [
            '/api/churn-analysis/health',
            '/api/churn-analysis/full-analysis',
            '/api/churn-analysis/predictions',
            '/api/churn-analysis/visualizations',
            '/api/churn-analysis/recommendations',
            '/api/churn-analysis/status'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Manejar errores internos del servidor"""
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor',
        'message': 'Ha ocurrido un error inesperado'
    }), 500


if __name__ == '__main__':
    # Usar configuración de Railway
    debug_mode = FLASK_CONFIG['debug']
    host = FLASK_CONFIG['host']
    port = FLASK_CONFIG['port']
    
    if debug_mode:
        print("🚀 Iniciando API de Análisis de Churn - Xumaa")
        print("📡 Endpoints disponibles:")
        print("   GET  /api/churn-analysis/health")
        print("   POST /api/churn-analysis/full-analysis")
        print("   GET  /api/churn-analysis/predictions")
        print("   GET  /api/churn-analysis/visualizations")
        print("   GET  /api/churn-analysis/recommendations")
        print("   GET  /api/churn-analysis/detailed-report")
        print("   GET  /api/churn-analysis/status")
        print(f"🌐 Servidor ejecutándose en: {host}:{port}")
    
    app.run(debug=debug_mode, host=host, port=port)