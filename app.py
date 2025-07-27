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

# Importar el servicio de churn
try:
    from churn_service import ChurnAnalysisService
    churn_service = ChurnAnalysisService()
    SERVICE_AVAILABLE = True
    SERVICE_TYPE = "full"
    print("✅ ChurnAnalysisService completo cargado")
except Exception as e:
    print(f"⚠️ Warning: ChurnAnalysisService completo no disponible: {e}")
    try:
        from churn_service_simple import ChurnAnalysisServiceSimple
        churn_service = ChurnAnalysisServiceSimple()
        SERVICE_AVAILABLE = True
        SERVICE_TYPE = "demo"
        print("✅ ChurnAnalysisService demo cargado")
    except Exception as e2:
        print(f"❌ Error: Ningún servicio de churn disponible: {e2}")
        churn_service = None
        SERVICE_AVAILABLE = False
        SERVICE_TYPE = "none"


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
    if not SERVICE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Servicio no disponible',
            'message': 'El servicio de análisis de churn no está disponible. Verifique la conexión a la base de datos.'
        }), 503
    
    try:
        results = churn_service.run_complete_analysis()
        
        if results['success']:
            return jsonify({
                'success': True,
                'data': results['data'],
                'message': 'Análisis de churn completado exitosamente'
            })
        else:
            return jsonify({
                'success': False,
                'error': results['error'],
                'message': results['message']
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error durante el análisis de churn'
        }), 500


@app.route('/api/churn-analysis/predictions', methods=['GET'])
def get_predictions():
    """Obtener predicciones de usuarios de alto riesgo"""
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