"""
API Principal de Análisis de Churn - Xumaa
Arquitectura modular y limpia
"""
from flask import Flask, jsonify, request
from churn_service import ChurnAnalysisService
import warnings
warnings.filterwarnings('ignore')

# Crear aplicación Flask
app = Flask(__name__)

# Instancia global del servicio
churn_service = ChurnAnalysisService()


@app.route('/api/churn-analysis/health', methods=['GET'])
def health_check():
    """Verificar el estado de la API"""
    try:
        health_data = churn_service.health_check()
        return jsonify({
            'success': True,
            'data': health_data,
            'message': 'API funcionando correctamente'
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
        status = churn_service.health_check()
        
        # Información adicional sobre el estado
        status_info = {
            **status,
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
    import os
    
    # Configuración para producción vs desarrollo
    is_production = os.environ.get('FLASK_ENV') == 'production'
    port = int(os.environ.get('PORT', 5001))
    
    if not is_production:
        print("🚀 Iniciando API de Análisis de Churn - Xumaa")
        print("📡 Endpoints disponibles:")
        print("   GET  /api/churn-analysis/health")
        print("   POST /api/churn-analysis/full-analysis")
        print("   GET  /api/churn-analysis/predictions")
        print("   GET  /api/churn-analysis/visualizations")
        print("   GET  /api/churn-analysis/recommendations")
        print("   GET  /api/churn-analysis/detailed-report")
        print("   GET  /api/churn-analysis/status")
        print(f"🌐 Servidor ejecutándose en: http://localhost:{port}")
    
    app.run(debug=not is_production, host='0.0.0.0', port=port)