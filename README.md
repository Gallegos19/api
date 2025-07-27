# API de Análisis de Churn - Xumaa

Esta API replica completamente la funcionalidad del notebook de análisis de churn, proporcionando endpoints para ejecutar análisis predictivo de abandono de usuarios y generar visualizaciones.

## 🏗️ Arquitectura Modular

La API está dividida en módulos especializados para mantener un código limpio y mantenible:

```
analytics-service/api/
├── app.py                 # API principal (Flask)
├── config.py             # Configuración centralizada
├── database.py           # Gestión de base de datos
├── data_processor.py     # Procesamiento de datos
├── models.py             # Modelos de ML
├── visualizations.py     # Generación de gráficos
├── recommendations.py    # Motor de recomendaciones
├── churn_service.py      # Servicio principal (orquestador)
├── test_api.py          # Script de pruebas
├── requirements.txt      # Dependencias
└── README.md            # Documentación
```

## 🚀 Características

- **Análisis completo de churn** con datos reales del data warehouse
- **Predicciones futuras** de comportamiento de usuarios (15 días)
- **Recomendaciones de intervención** personalizadas
- **Visualizaciones** en formato base64
- **Modelos de machine learning** entrenados con datos reales
- **Arquitectura modular** y escalable
- **Separación de responsabilidades** clara

## 📋 Requisitos

- Python 3.8+
- PostgreSQL con data warehouse configurado
- Variables de entorno configuradas

## 🛠️ Instalación

1. **Instalar dependencias:**
```bash
cd analytics-service/api
pip install -r requirements.txt
```

2. **Configurar variables de entorno:**
Asegúrate de que tu archivo `.env` contenga:
```env
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=tu_password
DB_HOST=localhost
DB_PORT=5432

# Configuración opcional del modelo
MODEL_N_ESTIMATORS=50
MODEL_MAX_DEPTH=5
PREDICTION_DAYS=15
```

3. **Ejecutar la API:**
```bash
python app.py
```

La API estará disponible en `http://localhost:5001`

## 📡 Endpoints

### 1. Health Check
```http
GET /api/churn-analysis/health
```
Verifica el estado de la API y la conexión a la base de datos.

**Respuesta:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "database": "connected",
    "model_trained": false,
    "future_model_trained": false
  },
  "message": "API funcionando correctamente"
}
```

### 2. Análisis Completo
```http
POST /api/churn-analysis/full-analysis
```
Ejecuta el análisis completo de churn (puede tomar varios minutos).

**Respuesta incluye:**
- Estadísticas de extracción de datos
- Métricas del modelo
- Recomendaciones de intervención
- Visualizaciones en base64
- Resumen de riesgo por usuarios

### 3. Predicciones
```http
GET /api/churn-analysis/predictions
```
Obtiene predicciones de usuarios de alto riesgo.

**Respuesta:**
```json
{
  "success": true,
  "data": {
    "high_risk_users": {
      "user_id_1": 0.85,
      "user_id_2": 0.78
    },
    "total_high_risk": 25,
    "predictions": [...]
  }
}
```

### 4. Visualizaciones
```http
GET /api/churn-analysis/visualizations
```
Genera y retorna visualizaciones como imágenes en base64.

**Respuesta:**
```json
{
  "success": true,
  "data": {
    "images": {
      "future_predictions": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  }
}
```

### 5. Recomendaciones
```http
GET /api/churn-analysis/recommendations
```
Obtiene recomendaciones de intervención para usuarios.

**Respuesta:**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "user_id": "user_123",
        "current_churned_status": 0,
        "avg_churn_risk": 0.75,
        "max_churn_risk": 0.85,
        "avg_engagement": 35.2,
        "recommendations": [
          "🚨 CRÍTICO: Contacto inmediato requerido",
          "📞 Llamada telefónica de retención"
        ]
      }
    ],
    "total_users": 1077
  }
}
```

## 🧪 Pruebas

Ejecuta el script de pruebas para verificar todos los endpoints:

```bash
python test_api.py
```

Este script:
- Verifica la conectividad
- Ejecuta el análisis completo
- Prueba todos los endpoints
- Guarda las imágenes generadas

## 📊 Salida del Análisis

El análisis completo retorna exactamente los mismos resultados que el notebook:

### Recomendaciones de Intervención
```
👤 Usuario 42ac9d23-7bba-4665-8d6b-4c8b91ef228e (ACTIVO):
   📊 Riesgo promedio: 0.320
   📈 Riesgo máximo: 0.500
   ⚠️  Días de alto riesgo: 0
   💡 Engagement promedio: 50.0
   📉 Tendencia engagement: +3.0
   🎯 Recomendaciones:
      📱 RIESGO MODERADO: Recordatorios de actividad
      🏆 Mostrar progreso y logros
```

### Resumen de Riesgo
```
📊 Usuarios analizados:
   👤 Usuarios activos: 493
   🔄 Usuarios ya con churn: 584
   📈 Total usuarios: 1077

📊 Distribución de riesgo futuro:
🟢 Bajo riesgo: 170 usuarios (15.8%)
🟡 Medio riesgo: 907 usuarios (84.2%)
🔴 Alto riesgo: 0 usuarios (0.0%)
```

## 🔧 Configuración Avanzada

### Variables de Entorno Adicionales
```env
# Configuración del modelo
MODEL_N_ESTIMATORS=50
MODEL_MAX_DEPTH=5
MODEL_MIN_SAMPLES_SPLIT=20

# Configuración de predicciones
PREDICTION_DAYS=15
SEQUENCE_LENGTH=30
```

### Personalización de Visualizaciones
La API genera automáticamente las mismas visualizaciones que el notebook:
- Predicción de riesgo de abandono (15 días)
- Predicción de engagement (15 días)
- Predicción de sesiones diarias (15 días)
- Predicción de intentos de quiz (15 días)

## 🚨 Manejo de Errores

La API maneja errores de forma robusta:

```json
{
  "success": false,
  "error": "Error conectando a la base de datos: connection refused",
  "message": "Error durante el análisis de churn"
}
```

## 📈 Rendimiento

- **Análisis completo**: 2-5 minutos (dependiendo del tamaño de datos)
- **Predicciones**: 10-30 segundos
- **Visualizaciones**: 5-15 segundos
- **Recomendaciones**: 5-10 segundos

## 🔒 Seguridad

- Validación de entrada en todos los endpoints
- Manejo seguro de conexiones a base de datos
- Logs de errores sin exposición de datos sensibles

## 📝 Logs

La API genera logs detallados para debugging:
- Conexiones a base de datos
- Tiempo de ejecución de modelos
- Errores y excepciones
- Estadísticas de procesamiento

## 🤝 Integración

### Ejemplo de uso con JavaScript:
```javascript
// Ejecutar análisis completo
const response = await fetch('http://localhost:5001/api/churn-analysis/full-analysis', {
  method: 'POST'
});
const data = await response.json();

// Mostrar visualizaciones
const imageBase64 = data.data.visualizations.future_predictions;
const img = document.createElement('img');
img.src = `data:image/png;base64,${imageBase64}`;
document.body.appendChild(img);
```

### Ejemplo de uso con Python:
```python
import requests
import base64

# Obtener recomendaciones
response = requests.get('http://localhost:5001/api/churn-analysis/recommendations')
recommendations = response.json()['data']['recommendations']

for rec in recommendations[:5]:
    print(f"Usuario {rec['user_id']}: Riesgo {rec['avg_churn_risk']:.3f}")
```

## 🐛 Troubleshooting

### Error de conexión a base de datos
- Verifica las variables de entorno
- Asegúrate de que PostgreSQL esté ejecutándose
- Verifica que el data warehouse tenga datos

### Modelo no entrenado
- Ejecuta primero `POST /api/churn-analysis/full-analysis`
- Verifica que haya suficientes datos en el data warehouse

### Visualizaciones no se generan
- Verifica que matplotlib esté instalado correctamente
- Asegúrate de que hay datos suficientes para generar gráficos

## 📞 Soporte

Para problemas o preguntas sobre la API, revisa:
1. Los logs de la aplicación
2. El endpoint `/health` para verificar el estado
3. El script `test_api.py` para ejemplos de uso