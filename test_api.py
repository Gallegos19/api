#!/usr/bin/env python3
"""
Script de prueba para la API de análisis de churn
Arquitectura modular
"""

import requests
import json
import base64
from datetime import datetime

# Configuración de la API
API_BASE_URL = "http://localhost:5001/api/churn-analysis"

def test_health_check():
    """Probar el endpoint de salud"""
    print("🔍 Probando health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_full_analysis():
    """Probar el análisis completo"""
    print("\n🚀 Ejecutando análisis completo...")
    try:
        response = requests.post(f"{API_BASE_URL}/full-analysis")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Análisis completado exitosamente!")
            
            # Mostrar resumen
            if 'data' in data and 'summary' in data['data']:
                summary = data['data']['summary']
                print(f"\n📊 RESUMEN:")
                print(f"   • Usuarios analizados: {summary.get('total_users_analyzed', 0)}")
                print(f"   • Usuarios activos: {summary.get('active_users', 0)}")
                print(f"   • Usuarios con churn: {summary.get('churned_users', 0)}")
                print(f"   • Días de predicción: {summary.get('prediction_days', 0)}")
                print(f"   • Total predicciones: {summary.get('total_predictions', 0)}")
                
                # Mostrar distribución de riesgo
                if 'risk_distribution' in summary:
                    print(f"\n📈 DISTRIBUCIÓN DE RIESGO:")
                    for level, stats in summary['risk_distribution'].items():
                        emoji = "🟢" if level == "Bajo" else "🟡" if level == "Medio" else "🔴"
                        print(f"   {emoji} {level}: {stats['count']} usuarios ({stats['percentage']:.1f}%)")
            
            # Mostrar algunas recomendaciones
            if 'data' in data and 'recommendations' in data['data']:
                recommendations = data['data']['recommendations'][:3]  # Primeras 3
                print(f"\n💡 RECOMENDACIONES (primeras 3):")
                for i, rec in enumerate(recommendations, 1):
                    status_emoji = "🔄" if rec['current_churned_status'] == 1 else "👤"
                    status_text = "ABANDONÓ" if rec['current_churned_status'] == 1 else "ACTIVO"
                    print(f"\n   {i}. {status_emoji} Usuario {rec['user_id']} ({status_text}):")
                    print(f"      📊 Riesgo promedio: {rec['avg_churn_risk']:.3f}")
                    print(f"      📈 Riesgo máximo: {rec['max_churn_risk']:.3f}")
                    print(f"      💡 Engagement promedio: {rec['avg_engagement']:.1f}")
                    print(f"      🎯 Recomendaciones: {len(rec['recommendations'])}")
                    for recommendation in rec['recommendations'][:2]:  # Primeras 2
                        print(f"         • {recommendation}")
            
            # Verificar si hay visualizaciones
            if 'data' in data and 'visualizations' in data['data']:
                visualizations = data['data']['visualizations']
                print(f"\n📊 VISUALIZACIONES:")
                for viz_name, viz_data in visualizations.items():
                    if viz_data:
                        print(f"   ✅ {viz_name}: {len(viz_data)} caracteres (base64)")
                        
                        # Guardar imagen como ejemplo
                        try:
                            image_data = base64.b64decode(viz_data)
                            filename = f"test_{viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            with open(filename, 'wb') as f:
                                f.write(image_data)
                            print(f"      💾 Guardada como: {filename}")
                        except Exception as e:
                            print(f"      ❌ Error guardando imagen: {e}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predictions():
    """Probar el endpoint de predicciones"""
    print("\n🔮 Probando predicciones...")
    try:
        response = requests.get(f"{API_BASE_URL}/predictions")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Predicciones obtenidas!")
            
            if 'data' in data:
                pred_data = data['data']
                print(f"   • Usuarios de alto riesgo: {pred_data.get('total_high_risk', 0)}")
                print(f"   • Predicciones obtenidas: {len(pred_data.get('predictions', []))}")
                
                # Mostrar algunos usuarios de alto riesgo
                if 'high_risk_users' in pred_data:
                    high_risk = pred_data['high_risk_users']
                    print(f"\n🔴 TOP USUARIOS DE ALTO RIESGO:")
                    for i, (user_id, risk) in enumerate(list(high_risk.items())[:5], 1):
                        print(f"   {i}. Usuario {user_id}: {risk:.3f}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_visualizations():
    """Probar el endpoint de visualizaciones"""
    print("\n📊 Probando visualizaciones...")
    try:
        response = requests.get(f"{API_BASE_URL}/visualizations")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Visualizaciones generadas!")
            
            if 'data' in data and 'images' in data['data']:
                images = data['data']['images']
                print(f"   • Imágenes generadas: {len(images)}")
                
                for img_name, img_data in images.items():
                    if img_data:
                        print(f"   ✅ {img_name}: {len(img_data)} caracteres")
                        
                        # Guardar imagen
                        try:
                            image_data = base64.b64decode(img_data)
                            filename = f"viz_{img_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            with open(filename, 'wb') as f:
                                f.write(image_data)
                            print(f"      💾 Guardada como: {filename}")
                        except Exception as e:
                            print(f"      ❌ Error guardando: {e}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_recommendations():
    """Probar el endpoint de recomendaciones"""
    print("\n💡 Probando recomendaciones...")
    try:
        response = requests.get(f"{API_BASE_URL}/recommendations")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Recomendaciones generadas!")
            
            if 'data' in data:
                rec_data = data['data']
                recommendations = rec_data.get('recommendations', [])
                print(f"   • Total usuarios: {rec_data.get('total_users', 0)}")
                print(f"   • Recomendaciones obtenidas: {len(recommendations)}")
                
                # Mostrar algunas recomendaciones
                print(f"\n🎯 TOP RECOMENDACIONES:")
                for i, rec in enumerate(recommendations[:3], 1):
                    status_emoji = "🔄" if rec['current_churned_status'] == 1 else "👤"
                    status_text = "ABANDONÓ" if rec['current_churned_status'] == 1 else "ACTIVO"
                    print(f"\n   {i}. {status_emoji} Usuario {rec['user_id']} ({status_text}):")
                    print(f"      📊 Riesgo promedio: {rec['avg_churn_risk']:.3f}")
                    print(f"      📈 Engagement promedio: {rec['avg_engagement']:.1f}")
                    print(f"      🎯 Recomendaciones:")
                    for recommendation in rec['recommendations'][:3]:
                        print(f"         • {recommendation}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("🧪 INICIANDO PRUEBAS DE LA API DE ANÁLISIS DE CHURN")
    print("=" * 60)
    
    # Probar health check
    if not test_health_check():
        print("❌ Health check falló. Verifica que la API esté ejecutándose.")
        return
    
    # Probar análisis completo (esto puede tomar tiempo)
    print("\n⏳ El análisis completo puede tomar varios minutos...")
    if not test_full_analysis():
        print("❌ Análisis completo falló.")
        return
    
    # Probar otros endpoints (estos deberían ser más rápidos)
    test_predictions()
    test_visualizations()
    test_recommendations()
    
    print("\n" + "=" * 60)
    print("✅ PRUEBAS COMPLETADAS")
    print("📁 Revisa las imágenes generadas en el directorio actual")

if __name__ == "__main__":
    main()