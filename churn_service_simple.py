"""
Servicio simplificado de análisis de churn para demostración
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class ChurnAnalysisServiceSimple:
    """Servicio simplificado que funciona con datos sintéticos"""
    
    def __init__(self):
        self.df = None
        self.model = None
        self.model_trained = False
        self.future_model_trained = True  # Simular que está entrenado
        
    def _generate_demo_data(self, n_users=1000):
        """Generar datos de demostración"""
        np.random.seed(42)
        
        # Generar usuarios sintéticos
        users = []
        for i in range(n_users):
            # Características base
            age = np.random.randint(13, 65)
            days_since_registration = np.random.randint(1, 365)
            
            # Métricas de engagement (correlacionadas con churn)
            base_engagement = np.random.normal(50, 20)
            engagement_score = max(0, min(100, base_engagement))
            
            # Actividad (correlacionada con engagement)
            sessions_count = max(0, int(np.random.poisson(engagement_score / 10)))
            avg_session_duration = max(0, np.random.normal(engagement_score * 2, 30))
            
            # Quiz y contenido
            quiz_attempts = max(0, int(np.random.poisson(sessions_count * 0.3)))
            quiz_completions = max(0, int(quiz_attempts * np.random.beta(2, 3)))
            content_views = max(0, int(np.random.poisson(sessions_count * 1.5)))
            
            # Días desde último login (factor importante para churn)
            days_since_last_login = max(0, int(np.random.exponential(5)))
            
            # Calcular churn basado en lógica realista
            churn_probability = 0.1  # Base rate
            
            # Factores que aumentan churn
            if days_since_last_login > 14:
                churn_probability += 0.4
            elif days_since_last_login > 7:
                churn_probability += 0.2
                
            if engagement_score < 30:
                churn_probability += 0.3
            elif engagement_score < 50:
                churn_probability += 0.1
                
            if sessions_count < 3:
                churn_probability += 0.2
                
            # Generar churn
            churned = 1 if np.random.random() < churn_probability else 0
            
            user = {
                'user_id': f'user_{i:04d}',
                'age': age,
                'days_since_registration': days_since_registration,
                'engagement_score': engagement_score,
                'sessions_count': sessions_count,
                'avg_session_duration': avg_session_duration,
                'quiz_attempts': quiz_attempts,
                'quiz_completions': quiz_completions,
                'content_views': content_views,
                'days_since_last_login': days_since_last_login,
                'churned': churned
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def _create_plot_base64(self, fig):
        """Convertir matplotlib figure a base64"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo con datos sintéticos"""
        try:
            # Generar datos de demostración
            self.df = self._generate_demo_data()
            
            # Entrenar modelo simple
            features = ['age', 'days_since_registration', 'engagement_score', 
                       'sessions_count', 'avg_session_duration', 'quiz_attempts',
                       'quiz_completions', 'content_views', 'days_since_last_login']
            
            X = self.df[features]
            y = self.df['churned']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X_train, y_train)
            
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            self.model_trained = True
            
            # Generar predicciones futuras sintéticas
            future_predictions = self._generate_future_predictions()
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(future_predictions)
            
            # Generar visualizaciones
            visualizations = self._generate_visualizations()
            
            return {
                'success': True,
                'data': {
                    'model_metrics': {
                        'roc_auc': float(auc_score),
                        'total_users': len(self.df),
                        'churn_rate': float(self.df['churned'].mean()),
                        'features_used': features
                    },
                    'predictions': future_predictions[:100],  # Primeras 100
                    'recommendations': recommendations[:20],  # Top 20
                    'visualizations': visualizations,
                    'summary': {
                        'total_users_analyzed': len(self.df),
                        'high_risk_users': len([r for r in recommendations if r['avg_churn_risk'] > 0.6]),
                        'model_accuracy': f"{auc_score:.3f}"
                    }
                },
                'message': 'Análisis completo ejecutado exitosamente (datos de demostración)'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Error durante el análisis completo'
            }
    
    def _generate_future_predictions(self):
        """Generar predicciones futuras sintéticas"""
        predictions = []
        
        # Seleccionar algunos usuarios para predicciones detalladas
        sample_users = self.df.sample(min(50, len(self.df)))
        
        for _, user in sample_users.iterrows():
            # Generar predicciones para los próximos 15 días
            for day in range(1, 16):
                # Simular degradación del engagement
                base_risk = user['churned'] if user['churned'] == 1 else 0.1
                
                # Factores que afectan el riesgo futuro
                if user['days_since_last_login'] > 7:
                    base_risk += 0.3
                if user['engagement_score'] < 40:
                    base_risk += 0.2
                if user['sessions_count'] < 5:
                    base_risk += 0.1
                
                # Añadir variación temporal
                daily_variation = np.random.normal(0, 0.05)
                predicted_risk = min(1.0, max(0.0, base_risk + daily_variation))
                
                # Simular engagement futuro
                engagement_trend = -0.5 if predicted_risk > 0.5 else 0.2
                predicted_engagement = max(0, user['engagement_score'] + engagement_trend * day + np.random.normal(0, 2))
                
                prediction = {
                    'user_id': user['user_id'],
                    'day': day,
                    'predicted_churn_risk': float(predicted_risk),
                    'predicted_engagement_score': float(predicted_engagement),
                    'current_churned_status': int(user['churned'])
                }
                predictions.append(prediction)
        
        return predictions
    
    def _generate_recommendations(self, predictions):
        """Generar recomendaciones basadas en predicciones"""
        # Agrupar por usuario
        user_predictions = {}
        for pred in predictions:
            user_id = pred['user_id']
            if user_id not in user_predictions:
                user_predictions[user_id] = []
            user_predictions[user_id].append(pred)
        
        recommendations = []
        for user_id, user_preds in user_predictions.items():
            avg_risk = np.mean([p['predicted_churn_risk'] for p in user_preds])
            max_risk = max([p['predicted_churn_risk'] for p in user_preds])
            avg_engagement = np.mean([p['predicted_engagement_score'] for p in user_preds])
            current_status = user_preds[0]['current_churned_status']
            
            # Generar recomendaciones basadas en riesgo
            user_recommendations = []
            if avg_risk > 0.7:
                user_recommendations.extend([
                    "🚨 CRÍTICO: Contacto inmediato requerido",
                    "📞 Llamada telefónica de retención",
                    "🎁 Ofrecer incentivo personalizado"
                ])
            elif avg_risk > 0.5:
                user_recommendations.extend([
                    "⚠️ ALTO RIESGO: Intervención necesaria",
                    "📧 Email personalizado de re-engagement",
                    "🎯 Contenido recomendado personalizado"
                ])
            elif avg_risk > 0.3:
                user_recommendations.extend([
                    "📊 Monitoreo cercano recomendado",
                    "💡 Sugerir nuevas funcionalidades",
                    "🔔 Notificaciones de contenido relevante"
                ])
            else:
                user_recommendations.extend([
                    "✅ Usuario estable",
                    "🌟 Candidato para programa de referidos",
                    "📈 Oportunidad de upselling"
                ])
            
            recommendation = {
                'user_id': user_id,
                'current_churned_status': current_status,
                'avg_churn_risk': float(avg_risk),
                'max_churn_risk': float(max_risk),
                'avg_engagement': float(avg_engagement),
                'days_high_risk': len([p for p in user_preds if p['predicted_churn_risk'] > 0.6]),
                'engagement_trend': float(np.random.normal(-0.5 if avg_risk > 0.5 else 0.2, 0.3)),
                'recommendations': user_recommendations
            }
            recommendations.append(recommendation)
        
        # Ordenar por riesgo
        recommendations.sort(key=lambda x: x['avg_churn_risk'], reverse=True)
        return recommendations
    
    def _generate_visualizations(self):
        """Generar visualizaciones básicas"""
        visualizations = {}
        
        try:
            # 1. Distribución de riesgo
            fig, ax = plt.subplots(figsize=(10, 6))
            risk_bins = [0, 0.3, 0.6, 1.0]
            risk_labels = ['Bajo', 'Medio', 'Alto']
            
            # Simular distribución de riesgo
            risks = np.random.beta(2, 5, 1000)  # Sesgado hacia riesgo bajo
            risk_categories = pd.cut(risks, bins=risk_bins, labels=risk_labels)
            
            counts = risk_categories.value_counts()
            colors = ['green', 'orange', 'red']
            
            bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.7)
            ax.set_title('Distribución de Riesgo de Churn', fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de Usuarios')
            ax.set_xlabel('Nivel de Riesgo')
            
            # Añadir valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            visualizations['risk_distribution'] = self._create_plot_base64(fig)
            
            # 2. Predicciones futuras (ejemplo)
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Simular datos de predicción temporal
            days = list(range(1, 16))
            high_risk_user = [0.3 + 0.04*d + np.random.normal(0, 0.02) for d in days]
            medium_risk_user = [0.2 + 0.01*d + np.random.normal(0, 0.02) for d in days]
            low_risk_user = [0.1 + 0.005*d + np.random.normal(0, 0.01) for d in days]
            
            ax.plot(days, high_risk_user, 'r-', linewidth=2, label='Usuario Alto Riesgo', marker='o')
            ax.plot(days, medium_risk_user, 'orange', linewidth=2, label='Usuario Riesgo Medio', marker='s')
            ax.plot(days, low_risk_user, 'g-', linewidth=2, label='Usuario Bajo Riesgo', marker='^')
            
            ax.set_title('Predicciones de Riesgo de Churn (Próximos 15 días)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Días')
            ax.set_ylabel('Probabilidad de Churn')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            visualizations['future_predictions'] = self._create_plot_base64(fig)
            
        except Exception as e:
            print(f"Error generando visualizaciones: {e}")
        
        return visualizations
    
    def get_high_risk_predictions(self):
        """Obtener predicciones de usuarios de alto riesgo"""
        if self.df is None:
            self.df = self._generate_demo_data()
        
        # Simular usuarios de alto riesgo
        high_risk_users = self.df[self.df['days_since_last_login'] > 7].copy()
        high_risk_users['risk_score'] = (
            0.3 + 
            (high_risk_users['days_since_last_login'] / 30) * 0.4 +
            ((100 - high_risk_users['engagement_score']) / 100) * 0.3
        )
        
        high_risk_dict = high_risk_users.set_index('user_id')['risk_score'].to_dict()
        
        return {
            'high_risk_users': high_risk_dict,
            'total_high_risk': len(high_risk_users),
            'predictions': high_risk_users.to_dict('records')[:50]
        }
    
    def get_visualizations_only(self):
        """Generar solo visualizaciones"""
        return self._generate_visualizations()
    
    def get_recommendations_only(self):
        """Generar solo recomendaciones"""
        if self.df is None:
            self.df = self._generate_demo_data()
        
        future_predictions = self._generate_future_predictions()
        recommendations = self._generate_recommendations(future_predictions)
        
        return {
            'recommendations': recommendations[:20],
            'total_users': len(recommendations)
        }
    
    def get_detailed_report(self):
        """Generar reporte detallado"""
        if self.df is None:
            self.df = self._generate_demo_data()
        
        future_predictions = self._generate_future_predictions()
        recommendations = self._generate_recommendations(future_predictions)
        
        # Top 5 usuarios para el reporte
        top_5_users = recommendations[:5]
        
        detailed_users = []
        for rec in top_5_users:
            status_emoji = "🔄" if rec['current_churned_status'] == 1 else "👤"
            status_text = "ABANDONÓ" if rec['current_churned_status'] == 1 else "ACTIVO"
            
            user_detail = {
                'user_id': rec['user_id'],
                'status': {
                    'emoji': status_emoji,
                    'text': status_text,
                    'code': rec['current_churned_status']
                },
                'metrics': {
                    'avg_churn_risk': rec['avg_churn_risk'],
                    'max_churn_risk': rec['max_churn_risk'],
                    'days_high_risk': rec['days_high_risk'],
                    'avg_engagement': rec['avg_engagement'],
                    'engagement_trend': rec['engagement_trend'],
                    'engagement_trend_formatted': f"{rec['engagement_trend']:+.1f}"
                },
                'recommendations': rec['recommendations']
            }
            detailed_users.append(user_detail)
        
        # Distribución de riesgo
        risk_distribution = {
            'Bajo': {'emoji': '🟢', 'count': 650, 'percentage': 65.0},
            'Medio': {'emoji': '🟡', 'count': 250, 'percentage': 25.0},
            'Alto': {'emoji': '🔴', 'count': 100, 'percentage': 10.0}
        }
        
        return {
            'intervention_recommendations': {
                'title': 'RECOMENDACIONES DE INTERVENCIÓN PARA LOS PRÓXIMOS 15 DÍAS',
                'users': detailed_users
            },
            'risk_summary': {
                'title': 'RESUMEN DE RIESGO FUTURO (próximos 15 días)',
                'users_analyzed': {
                    'active_users': 850,
                    'churned_users': 150,
                    'total_users': 1000
                },
                'risk_distribution': risk_distribution
            },
            'model_statistics': {
                'title': 'ESTADÍSTICAS DEL MODELO',
                'stats': {
                    'users_analyzed': 1000,
                    'prediction_days': 15,
                    'metrics_per_day': 8,
                    'total_predictions': 750,
                    'predictions_per_user': 15
                }
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_users_in_report': len(detailed_users),
                'report_type': 'detailed_intervention_report_demo'
            }
        }
    
    def health_check(self):
        """Verificar el estado del servicio"""
        return {
            'status': 'healthy',
            'database': 'demo_mode',
            'model_trained': True,
            'future_model_trained': True,
            'data_loaded': True
        }