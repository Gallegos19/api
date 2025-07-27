"""
Módulo de generación de recomendaciones de intervención
"""
import pandas as pd


class RecommendationEngine:
    """Motor de recomendaciones para intervención de churn"""
    
    def __init__(self):
        pass
    
    def generate_intervention_recommendations(self, future_predictions):
        """Generar recomendaciones de intervención basadas en predicciones"""
        if len(future_predictions) == 0:
            return []
        
        recommendations = []
        
        # Convertir a DataFrame para facilitar el análisis
        df_predictions = pd.DataFrame(future_predictions)
        
        # Agrupar por usuario
        for user_id in df_predictions['user_id'].unique():
            user_data = df_predictions[df_predictions['user_id'] == user_id]
            
            # Calcular métricas de riesgo
            avg_churn_risk = user_data['predicted_churn_risk'].mean()
            max_churn_risk = user_data['predicted_churn_risk'].max()
            days_high_risk = len(user_data[user_data['predicted_churn_risk'] > 0.7])
            avg_engagement = user_data['predicted_engagement_score'].mean()
            trend_engagement = user_data['predicted_engagement_score'].iloc[-1] - user_data['predicted_engagement_score'].iloc[0]
            current_status = user_data['current_churned_status'].iloc[0]
            
            # Generar recomendaciones
            user_recommendations = {
                'user_id': user_id,
                'current_churned_status': int(current_status),
                'avg_churn_risk': float(avg_churn_risk),
                'max_churn_risk': float(max_churn_risk),
                'days_high_risk': int(days_high_risk),
                'avg_engagement': float(avg_engagement),
                'engagement_trend': float(trend_engagement),
                'recommendations': []
            }
            
            # Generar recomendaciones específicas
            user_recommendations['recommendations'] = self._generate_user_recommendations(
                current_status, avg_churn_risk, avg_engagement, trend_engagement, user_data
            )
            
            recommendations.append(user_recommendations)
        
        # Ordenar por riesgo (mayor a menor)
        recommendations.sort(key=lambda x: x['avg_churn_risk'], reverse=True)
        
        return recommendations
    
    def _generate_user_recommendations(self, current_status, avg_churn_risk, avg_engagement, trend_engagement, user_data):
        """Generar recomendaciones específicas para un usuario"""
        recommendations = []
        
        # Recomendaciones especiales para usuarios ya con churn
        if current_status == 1:
            recommendations.append("🔄 REACTIVACIÓN: Usuario ya abandonó - campaña de reactivación")
            recommendations.append("🎁 Ofrecer incentivos especiales de regreso")
            recommendations.append("📞 Contacto directo para entender motivos de abandono")
            recommendations.append("📧 Email personalizado con contenido relevante")
            return recommendations
        
        # Recomendaciones basadas en riesgo futuro
        if avg_churn_risk > 0.7:
            recommendations.extend([
                "🚨 CRÍTICO: Contacto inmediato requerido",
                "📞 Llamada telefónica de retención",
                "🎁 Ofrecer incentivo especial",
                "👨‍🏫 Asignar tutor personal",
                "📱 Activar notificaciones push urgentes"
            ])
        elif avg_churn_risk > 0.5:
            recommendations.extend([
                "⚠️ ALTO RIESGO: Enviar contenido motivacional",
                "👥 Invitar a desafíos grupales",
                "📧 Email personalizado del tutor",
                "🏆 Destacar logros y progreso",
                "📚 Recomendar contenido personalizado"
            ])
        elif avg_churn_risk > 0.3:
            recommendations.extend([
                "📱 RIESGO MODERADO: Recordatorios de actividad",
                "🏆 Mostrar progreso y logros",
                "🎮 Sugerir actividades interactivas",
                "📊 Compartir estadísticas de progreso"
            ])
        else:
            recommendations.extend([
                "✅ USUARIO ESTABLE: Mantener engagement actual",
                "🌟 Ofrecer contenido avanzado",
                "🏅 Reconocer como usuario ejemplar"
            ])
        
        # Recomendaciones basadas en engagement
        if avg_engagement < 30:
            recommendations.extend([
                "📚 Revisar dificultad del contenido",
                "🎮 Ofrecer contenido más interactivo",
                "🔄 Cambiar metodología de enseñanza",
                "👥 Conectar con otros estudiantes"
            ])
        elif avg_engagement < 50:
            recommendations.extend([
                "📈 Incrementar variedad de contenido",
                "🎯 Personalizar experiencia de aprendizaje"
            ])
        
        # Recomendaciones basadas en tendencia de engagement
        if trend_engagement < -10:
            recommendations.extend([
                "📉 Engagement decreciente: Intervención urgente",
                "🔄 Cambiar estrategia de contenido",
                "💬 Solicitar feedback directo",
                "🎯 Revisar objetivos de aprendizaje"
            ])
        elif trend_engagement > 10:
            recommendations.extend([
                "📈 Engagement creciente: Aprovechar momentum",
                "🚀 Ofrecer desafíos adicionales"
            ])
        
        # Recomendaciones basadas en actividad predicha
        low_activity_days = len(user_data[user_data['predicted_sessions_count'] == 0])
        if low_activity_days > 7:
            recommendations.extend([
                "🔔 Implementar notificaciones push",
                "💰 Ofrecer incentivos de reactivación",
                "📅 Crear calendario de estudio personalizado"
            ])
        
        # Recomendaciones basadas en patrones de quiz
        low_quiz_days = len(user_data[user_data['predicted_quiz_attempts'] == 0])
        if low_quiz_days > 5:
            recommendations.extend([
                "❓ Simplificar formato de quizzes",
                "🎲 Gamificar evaluaciones",
                "⏰ Recordatorios de práctica"
            ])
        
        return recommendations
    
    def get_high_risk_users(self, recommendations, risk_threshold=0.6):
        """Obtener usuarios de alto riesgo"""
        high_risk_users = [
            rec for rec in recommendations 
            if rec['avg_churn_risk'] > risk_threshold
        ]
        return high_risk_users
    
    def get_users_by_status(self, recommendations):
        """Agrupar usuarios por estado actual"""
        active_users = [rec for rec in recommendations if rec['current_churned_status'] == 0]
        churned_users = [rec for rec in recommendations if rec['current_churned_status'] == 1]
        
        return {
            'active': active_users,
            'churned': churned_users,
            'total_active': len(active_users),
            'total_churned': len(churned_users)
        }
    
    def generate_summary_report(self, recommendations):
        """Generar reporte resumen de recomendaciones"""
        if not recommendations:
            return {}
        
        # Estadísticas generales
        total_users = len(recommendations)
        avg_risk = sum(rec['avg_churn_risk'] for rec in recommendations) / total_users
        
        # Distribución por nivel de riesgo
        risk_distribution = {
            'bajo': len([r for r in recommendations if r['avg_churn_risk'] <= 0.3]),
            'medio': len([r for r in recommendations if 0.3 < r['avg_churn_risk'] <= 0.6]),
            'alto': len([r for r in recommendations if r['avg_churn_risk'] > 0.6])
        }
        
        # Usuarios por estado
        status_distribution = self.get_users_by_status(recommendations)
        
        # Top recomendaciones más comunes
        all_recommendations = []
        for rec in recommendations:
            all_recommendations.extend(rec['recommendations'])
        
        recommendation_counts = pd.Series(all_recommendations).value_counts().head(10)
        
        return {
            'total_users': total_users,
            'average_risk': avg_risk,
            'risk_distribution': risk_distribution,
            'status_distribution': {
                'active_users': status_distribution['total_active'],
                'churned_users': status_distribution['total_churned']
            },
            'top_recommendations': recommendation_counts.to_dict(),
            'high_risk_users': len(self.get_high_risk_users(recommendations))
        }
    
    def export_recommendations_to_csv(self, recommendations, filename='recommendations.csv'):
        """Exportar recomendaciones a CSV"""
        if not recommendations:
            return None
        
        # Preparar datos para CSV
        csv_data = []
        for rec in recommendations:
            base_data = {
                'user_id': rec['user_id'],
                'current_churned_status': rec['current_churned_status'],
                'avg_churn_risk': rec['avg_churn_risk'],
                'max_churn_risk': rec['max_churn_risk'],
                'days_high_risk': rec['days_high_risk'],
                'avg_engagement': rec['avg_engagement'],
                'engagement_trend': rec['engagement_trend'],
                'total_recommendations': len(rec['recommendations'])
            }
            
            # Añadir recomendaciones como columnas separadas
            for i, recommendation in enumerate(rec['recommendations'][:5]):  # Top 5
                base_data[f'recommendation_{i+1}'] = recommendation
            
            csv_data.append(base_data)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        return filename