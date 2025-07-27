"""
Servicio principal de análisis de churn
"""
from database import DatabaseManager
from data_processor import DataProcessor
from models import ChurnModel, FuturePredictionModel
from visualizations import VisualizationGenerator
from recommendations import RecommendationEngine
from utils import safe_json_response, format_user_recommendations, format_model_metrics, format_data_statistics
import pandas as pd


class ChurnAnalysisService:
    """Servicio principal que orquesta todo el análisis de churn"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_processor = DataProcessor()
        self.churn_model = ChurnModel()
        self.future_model = FuturePredictionModel()
        self.viz_generator = VisualizationGenerator()
        self.recommendation_engine = RecommendationEngine()
        
        # Estado del servicio
        self.df = None
        self.selected_features = None
        self.model_trained = False
        self.future_model_trained = False
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo de churn"""
        try:
            results = {}
            
            # 1. Extraer datos
            results['step_1'] = self._extract_data()
            
            # 2. Procesar datos
            results['step_2'] = self._process_data()
            
            # 3. Entrenar modelo de churn
            results['step_3'] = self._train_churn_model()
            
            # 4. Entrenar modelo de predicción futura
            results['step_4'] = self._train_future_model()
            
            # 5. Generar predicciones
            results['step_5'] = self._generate_predictions()
            
            # 6. Generar recomendaciones
            results['step_6'] = self._generate_recommendations(results['step_5']['predictions'])
            
            # 7. Generar visualizaciones
            results['step_7'] = self._generate_visualizations(results)
            
            # 8. Crear resumen final
            results['summary'] = self._create_summary(results)
            
            return {
                'success': True,
                'data': results,
                'message': 'Análisis completo ejecutado exitosamente'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Error durante el análisis completo'
            }
    
    def _extract_data(self):
        """Paso 1: Extraer datos del data warehouse"""
        # Extraer datos principales
        self.df = self.db_manager.get_user_data()
        
        # Añadir características derivadas
        self.df = self.data_processor.add_derived_features(self.df)
        
        # Obtener estadísticas
        stats = self.data_processor.get_data_statistics(self.df)
        
        return safe_json_response({
            'total_records': len(self.df),
            'extraction_stats': format_data_statistics(stats),
            'message': 'Datos extraídos exitosamente'
        })
    
    def _process_data(self):
        """Paso 2: Procesar y preparar datos"""
        # Intentar definición realista de churn
        try:
            df_realistic = self.db_manager.get_realistic_churn_data()
            if len(df_realistic) > 0 and df_realistic['churned_realistic'].sum() >= 10:
                self.df = df_realistic.copy()
                self.df['churned'] = self.df['churned_realistic']
                self.df = self.df.drop('churned_realistic', axis=1)
                approach = "Nueva definición de churn"
                processing_stats = {
                    'churn_rate': self.df['churned'].mean() * 100,
                    'churn_distribution': self.df['churned'].value_counts().to_dict()
                }
            else:
                raise Exception("Insuficientes casos de churn realista")
        except:
            # Fallback: añadir ruido
            self.df, n_flips = self.data_processor.add_realistic_noise(self.df)
            approach = "Datos con ruido añadido"
            processing_stats = {
                'flipped_labels': n_flips,
                'churn_rate': self.df['churned'].mean() * 100,
                'churn_distribution': self.df['churned'].value_counts().to_dict()
            }
        
        # Preparar características
        df_prepared, self.selected_features, correlations = self.data_processor.prepare_features_ultra_conservative(self.df)
        self.df = df_prepared
        
        return safe_json_response({
            'approach_used': approach,
            'processing_stats': format_data_statistics(processing_stats),
            'selected_features': self.selected_features,
            'correlations': correlations.head(10).to_dict(),
            'message': 'Datos procesados exitosamente'
        })
    
    def _train_churn_model(self):
        """Paso 3: Entrenar modelo de churn"""
        # Dividir y escalar datos
        X_train, X_test, y_train, y_test, scaler = self.data_processor.split_and_scale_data(
            self.df, self.selected_features
        )
        
        # Entrenar modelo
        model, y_pred, y_pred_proba, metrics = self.churn_model.train_ultra_conservative_model(
            X_train, y_train, X_test, y_test, self.selected_features
        )
        
        # Diagnóstico del modelo
        diagnosis = self.churn_model.diagnose_model(X_test, y_test, y_pred_proba)
        
        self.model_trained = True
        
        return safe_json_response({
            'model_metrics': format_model_metrics(metrics),
            'model_diagnosis': diagnosis,
            'test_data': {
                'y_test': y_test.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            },
            'message': 'Modelo de churn entrenado exitosamente'
        })
    
    def _train_future_model(self):
        """Paso 4: Entrenar modelo de predicción futura"""
        # Crear secuencias de usuario
        user_sequences, user_info = self.future_model.create_user_sequences(self.df)
        
        # Entrenar modelo
        metrics = self.future_model.train_model(user_sequences)
        
        # Guardar información para uso posterior
        self.user_sequences = user_sequences
        self.user_info = user_info
        self.future_model_trained = True
        
        return safe_json_response({
            'future_model_metrics': format_model_metrics(metrics),
            'users_processed': len(user_info),
            'message': 'Modelo de predicción futura entrenado exitosamente'
        })
    
    def _generate_predictions(self):
        """Paso 5: Generar predicciones futuras"""
        if not self.future_model_trained:
            raise Exception("Modelo de predicción futura no entrenado")
        
        # Generar predicciones
        predictions = self.future_model.predict_future_behavior(
            self.user_sequences, self.user_info
        )
        
        return safe_json_response({
            'predictions': predictions,
            'total_predictions': len(predictions),
            'users_predicted': len(self.user_info),
            'message': 'Predicciones generadas exitosamente'
        })
    
    def _generate_recommendations(self, predictions):
        """Paso 6: Generar recomendaciones"""
        recommendations = self.recommendation_engine.generate_intervention_recommendations(predictions)
        summary_report = self.recommendation_engine.generate_summary_report(recommendations)
        
        return safe_json_response({
            'recommendations': format_user_recommendations(recommendations[:20]),  # Top 20
            'summary_report': summary_report,
            'total_recommendations': len(recommendations),
            'message': 'Recomendaciones generadas exitosamente'
        })
    
    def _generate_visualizations(self, results):
        """Paso 7: Generar visualizaciones"""
        # Preparar datos para visualizaciones
        viz_data = {
            'df': self.df,
            'future_predictions': results['step_5']['predictions'],
            'correlations': pd.Series(results['step_2']['correlations']),
            'feature_importances': results['step_3']['model_metrics']['feature_importances'],
            'y_test': results['step_3']['test_data']['y_test'],
            'y_pred_proba': results['step_3']['test_data']['y_pred_proba']
        }
        
        # Generar todas las visualizaciones
        visualizations = self.viz_generator.generate_all_visualizations(viz_data)
        
        return safe_json_response({
            'visualizations': visualizations,
            'total_visualizations': len(visualizations),
            'message': 'Visualizaciones generadas exitosamente'
        })
    
    def _create_summary(self, results):
        """Paso 8: Crear resumen final"""
        # Calcular distribución de riesgo
        predictions_df = pd.DataFrame(results['step_5']['predictions'])
        risk_summary = predictions_df.groupby('user_id')['predicted_churn_risk'].mean()
        risk_levels = pd.cut(risk_summary, bins=[0, 0.3, 0.6, 1.0], labels=['Bajo', 'Medio', 'Alto'])
        
        risk_distribution = {}
        for level in ['Bajo', 'Medio', 'Alto']:
            count = (risk_levels == level).sum()
            percentage = count / len(risk_levels) * 100 if len(risk_levels) > 0 else 0
            risk_distribution[level] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # Resumen por estado actual
        current_status_summary = predictions_df.groupby(['user_id', 'current_churned_status']).first().reset_index()
        active_users_count = len(current_status_summary[current_status_summary['current_churned_status'] == 0])
        churned_users_count = len(current_status_summary[current_status_summary['current_churned_status'] == 1])
        
        return safe_json_response({
            'total_users_analyzed': len(self.user_info),
            'active_users': active_users_count,
            'churned_users': churned_users_count,
            'risk_distribution': risk_distribution,
            'prediction_days': self.future_model.prediction_days,
            'total_predictions': len(results['step_5']['predictions']),
            'model_performance': {
                'churn_model_auc': results['step_3']['model_metrics']['roc_auc'],
                'future_model_mae': results['step_4']['future_model_metrics'].get('mae', 0)
            }
        })
    
    def _quick_model_setup(self):
        """Configuración rápida del modelo si no está entrenado"""
        if not self.future_model_trained:
            # Extraer datos si no están cargados
            if self.df is None:
                self._extract_data()
                self._process_data()
            
            # Entrenar modelo de predicción futura
            user_sequences, user_info = self.future_model.create_user_sequences(self.df)
            self.future_model.train_model(user_sequences)
            
            # Guardar información para uso posterior
            self.user_sequences = user_sequences
            self.user_info = user_info
            self.future_model_trained = True

    def get_high_risk_predictions(self):
        """Obtener predicciones de usuarios de alto riesgo"""
        if not self.future_model_trained:
            # Si no hay modelo entrenado, ejecutar análisis rápido
            self._quick_model_setup()
        
        # Generar predicciones
        predictions = self.future_model.predict_future_behavior(
            self.user_sequences, self.user_info
        )
        
        # Filtrar usuarios de alto riesgo
        df_pred = pd.DataFrame(predictions)
        high_risk_users = df_pred.groupby('user_id')['predicted_churn_risk'].mean()
        high_risk_users = high_risk_users[high_risk_users > 0.6].sort_values(ascending=False)
        
        return safe_json_response({
            'high_risk_users': high_risk_users.to_dict(),
            'total_high_risk': len(high_risk_users),
            'predictions': predictions[:100]  # Primeras 100 predicciones
        })
    
    def get_visualizations_only(self):
        """Generar solo visualizaciones"""
        if not self.future_model_trained:
            # Si no hay modelo entrenado, ejecutar análisis rápido
            self._quick_model_setup()
        
        # Generar predicciones
        predictions = self.future_model.predict_future_behavior(
            self.user_sequences, self.user_info
        )
        
        # Generar visualizaciones
        viz_data = {
            'future_predictions': predictions
        }
        
        return self.viz_generator.generate_all_visualizations(viz_data)
    
    def get_recommendations_only(self):
        """Generar solo recomendaciones"""
        if not self.future_model_trained:
            # Si no hay modelo entrenado, ejecutar análisis rápido
            self._quick_model_setup()
        
        # Generar predicciones
        predictions = self.future_model.predict_future_behavior(
            self.user_sequences, self.user_info
        )
        
        # Generar recomendaciones
        recommendations = self.recommendation_engine.generate_intervention_recommendations(predictions)
        
        # Ordenar por riesgo
        recommendations.sort(key=lambda x: x['avg_churn_risk'], reverse=True)
        
        return safe_json_response({
            'recommendations': format_user_recommendations(recommendations[:20]),  # Top 20
            'total_users': len(recommendations)
        })
    
    def get_detailed_report(self):
        """Generar reporte detallado con los 5 usuarios de la gráfica y resumen completo"""
        if not self.future_model_trained:
            # Si no hay modelo entrenado, ejecutar análisis rápido
            self._quick_model_setup()
        
        # Generar predicciones
        predictions = self.future_model.predict_future_behavior(
            self.user_sequences, self.user_info
        )
        
        # Generar recomendaciones completas
        recommendations = self.recommendation_engine.generate_intervention_recommendations(predictions)
        
        # Seleccionar los primeros 5 usuarios (los que aparecen en la gráfica)
        top_5_users = recommendations[:5]
        
        # Formatear los usuarios detallados
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
        
        # Crear resumen de riesgo
        df_pred = pd.DataFrame(predictions)
        risk_summary = df_pred.groupby('user_id')['predicted_churn_risk'].mean()
        risk_levels = pd.cut(risk_summary, bins=[0, 0.3, 0.6, 1.0], labels=['Bajo', 'Medio', 'Alto'])
        
        current_status_summary = df_pred.groupby(['user_id', 'current_churned_status']).first().reset_index()
        active_users_count = len(current_status_summary[current_status_summary['current_churned_status'] == 0])
        churned_users_count = len(current_status_summary[current_status_summary['current_churned_status'] == 1])
        
        risk_distribution = {}
        for level in ['Bajo', 'Medio', 'Alto']:
            count = (risk_levels == level).sum()
            percentage = count / len(risk_levels) * 100 if len(risk_levels) > 0 else 0
            emoji = "🟢" if level == "Bajo" else "🟡" if level == "Medio" else "🔴"
            risk_distribution[level] = {
                'emoji': emoji,
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # Estadísticas del modelo
        model_stats = {
            'users_analyzed': len(self.user_info),
            'prediction_days': self.future_model.prediction_days,
            'metrics_per_day': len(self.future_model.feature_names),
            'total_predictions': len(predictions),
            'predictions_per_user': len(predictions) // len(self.user_info) if len(self.user_info) > 0 else 0
        }
        
        return safe_json_response({
            'intervention_recommendations': {
                'title': 'RECOMENDACIONES DE INTERVENCIÓN PARA LOS PRÓXIMOS 15 DÍAS',
                'users': detailed_users
            },
            'risk_summary': {
                'title': 'RESUMEN DE RIESGO FUTURO (próximos 15 días)',
                'users_analyzed': {
                    'active_users': active_users_count,
                    'churned_users': churned_users_count,
                    'total_users': len(risk_summary)
                },
                'risk_distribution': risk_distribution
            },
            'model_statistics': {
                'title': 'ESTADÍSTICAS DEL MODELO',
                'stats': model_stats
            },
            'metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'total_users_in_report': len(detailed_users),
                'report_type': 'detailed_intervention_report'
            }
        })

    def health_check(self):
        """Verificar el estado del servicio"""
        db_status = self.db_manager.test_connection()
        
        return {
            'status': 'healthy',
            'database': 'connected' if db_status else 'disconnected',
            'model_trained': self.model_trained,
            'future_model_trained': self.future_model_trained,
            'data_loaded': self.df is not None
        }