# API de Análisis de Churn - Despliegue en Railway

## 🚀 Despliegue Automático

Esta API está configurada para desplegarse automáticamente en Railway.

## 📋 Variables de Entorno Requeridas

Configura estas variables en Railway:

```env
# Base de datos (REQUERIDAS)
DB_NAME=tu_db_name
DB_USER=tu_db_user  
DB_PASSWORD=tu_db_password
DB_HOST=tu_db_host
DB_PORT=6543

# Aplicación
FLASK_ENV=production
PORT=5001

# Modelo (OPCIONALES)
MODEL_N_ESTIMATORS=50
MODEL_MAX_DEPTH=5
PREDICTION_DAYS=15
```

## 🔗 Endpoints Disponibles

Una vez desplegado, los endpoints estarán disponibles en:

- `GET /api/churn-analysis/health` - Health check
- `POST /api/churn-analysis/full-analysis` - Análisis completo
- `GET /api/churn-analysis/predictions` - Predicciones de alto riesgo
- `GET /api/churn-analysis/visualizations` - Gráficos en base64
- `GET /api/churn-analysis/recommendations` - Recomendaciones
- `GET /api/churn-analysis/detailed-report` - Reporte detallado
- `GET /api/churn-analysis/status` - Estado del servicio

## 🏥 Health Check

Railway usará `/api/churn-analysis/health` para verificar que el servicio esté funcionando.

## ⚙️ Configuración

- **Puerto**: Configurado automáticamente por Railway
- **Timeout**: 300 segundos para análisis largos
- **Workers**: 2 workers de Gunicorn
- **Memoria**: Optimizado para Railway