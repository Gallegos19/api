"""
Módulo de generación de visualizaciones
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from config import VISUALIZATION_CONFIG


class VisualizationGenerator:
    """Generador de visualizaciones para análisis de churn"""
    
    def __init__(self):
        self.config = VISUALIZATION_CONFIG
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configurar matplotlib"""
        plt.ioff()  # No mostrar ventanas
        sns.set_style(self.config['style'])
        plt.rcParams['figure.figsize'] = self.config['figure_size']
        plt.rcParams['font.size'] = self.config['font_size']
    
    def _fig_to_base64(self, fig):
        """Convertir figura de matplotlib a base64"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.config['dpi'], bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64
    
    def generate_future_predictions_chart(self, future_predictions, user_ids_to_show=5):
        """Generar gráfico de predicciones futuras"""
        if len(future_predictions) == 0:
            return None
        
        df_predictions = pd.DataFrame(future_predictions)
        selected_users = df_predictions['user_id'].unique()[:user_ids_to_show]
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        
        # 1. Riesgo de abandono a lo largo del tiempo
        for user_id in selected_users:
            user_data = df_predictions[df_predictions['user_id'] == user_id]
            axes[0, 0].plot(user_data['prediction_day'], user_data['predicted_churn_risk'], 
                           marker='o', label=f'Usuario {user_id}', linewidth=2)
        
        axes[0, 0].set_title('Predicción de Riesgo de Abandono (15 días)', fontsize=14)
        axes[0, 0].set_xlabel('Día de Predicción', fontsize=12)
        axes[0, 0].set_ylabel('Riesgo de Abandono', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Engagement predicho
        for user_id in selected_users:
            user_data = df_predictions[df_predictions['user_id'] == user_id]
            axes[0, 1].plot(user_data['prediction_day'], user_data['predicted_engagement_score'], 
                           marker='s', label=f'Usuario {user_id}', linewidth=2)
        
        axes[0, 1].set_title('Predicción de Engagement (15 días)', fontsize=14)
        axes[0, 1].set_xlabel('Día de Predicción', fontsize=12)
        axes[0, 1].set_ylabel('Score de Engagement', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Actividad de sesiones predicha
        for user_id in selected_users:
            user_data = df_predictions[df_predictions['user_id'] == user_id]
            axes[1, 0].plot(user_data['prediction_day'], user_data['predicted_sessions_count'], 
                           marker='^', label=f'Usuario {user_id}', linewidth=2)
        
        axes[1, 0].set_title('Predicción de Sesiones Diarias (15 días)', fontsize=14)
        axes[1, 0].set_xlabel('Día de Predicción', fontsize=12)
        axes[1, 0].set_ylabel('Número de Sesiones', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Actividad de quizzes predicha
        for user_id in selected_users:
            user_data = df_predictions[df_predictions['user_id'] == user_id]
            axes[1, 1].plot(user_data['prediction_day'], user_data['predicted_quiz_attempts'], 
                           marker='d', label=f'Usuario {user_id}', linewidth=2)
        
        axes[1, 1].set_title('Predicción de Intentos de Quiz (15 días)', fontsize=14)
        axes[1, 1].set_xlabel('Día de Predicción', fontsize=12)
        axes[1, 1].set_ylabel('Intentos de Quiz', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_churn_distribution_chart(self, df):
        """Generar gráfico de distribución de churn"""
        if 'churned' not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        churn_counts = df['churned'].value_counts()
        ax = sns.countplot(x='churned', data=df, ax=ax)
        ax.set_title('Distribución de Abandono - Datos Reales del Data Warehouse', fontsize=16)
        ax.set_xlabel('Abandono (1 = Sí, 0 = No)', fontsize=14)
        ax.set_ylabel('Cantidad de Usuarios', fontsize=14)
        
        # Añadir etiquetas con porcentajes
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                    f'{int(height)} ({height/total*100:.1f}%)',
                    ha="center", fontsize=12)
        
        return self._fig_to_base64(fig)
    
    def generate_correlation_chart(self, correlations):
        """Generar gráfico de correlaciones"""
        if correlations is None or len(correlations) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Tomar las top 15 correlaciones
        top_correlations = correlations.head(15)
        
        sns.barplot(x=top_correlations.values, y=top_correlations.index, ax=ax)
        ax.set_title('Correlación con Abandono - Datos Reales', fontsize=16)
        ax.set_xlabel('Coeficiente de Correlación', fontsize=14)
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_feature_importance_chart(self, feature_importances):
        """Generar gráfico de importancia de características"""
        if not feature_importances:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convertir a DataFrame si es necesario
        if isinstance(feature_importances, list):
            df_importance = pd.DataFrame(feature_importances)
        else:
            df_importance = feature_importances
        
        # Tomar las top 10 características
        top_features = df_importance.head(10)
        
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importancia', fontsize=14)
        ax.set_title('Importancia de Características - Modelo de Churn', fontsize=16)
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_risk_distribution_chart(self, future_predictions):
        """Generar gráfico de distribución de riesgo"""
        if len(future_predictions) == 0:
            return None
        
        df_predictions = pd.DataFrame(future_predictions)
        
        # Calcular riesgo promedio por usuario
        risk_summary = df_predictions.groupby('user_id')['predicted_churn_risk'].mean()
        risk_levels = pd.cut(risk_summary, bins=[0, 0.3, 0.6, 1.0], labels=['Bajo', 'Medio', 'Alto'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        risk_counts = risk_levels.value_counts()
        colors = ['green', 'orange', 'red']
        
        bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7)
        ax.set_title('Distribución de Riesgo Futuro de Abandono', fontsize=16)
        ax.set_xlabel('Nivel de Riesgo', fontsize=14)
        ax.set_ylabel('Número de Usuarios', fontsize=14)
        
        # Añadir etiquetas con porcentajes
        total = len(risk_summary)
        for bar, count in zip(bars, risk_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def generate_model_performance_chart(self, y_test, y_pred_proba):
        """Generar gráfico de rendimiento del modelo"""
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = np.trapz(tpr, fpr)
        
        axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_xlabel('Tasa de Falsos Positivos')
        axes[0].set_ylabel('Tasa de Verdaderos Positivos')
        axes[0].set_title('Curva ROC')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        axes[1].plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}', linewidth=2)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Curva Precision-Recall')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def generate_all_visualizations(self, data_dict):
        """Generar todas las visualizaciones disponibles"""
        visualizations = {}
        
        # Predicciones futuras
        if 'future_predictions' in data_dict:
            viz = self.generate_future_predictions_chart(data_dict['future_predictions'])
            if viz:
                visualizations['future_predictions'] = viz
        
        # Distribución de churn
        if 'df' in data_dict:
            viz = self.generate_churn_distribution_chart(data_dict['df'])
            if viz:
                visualizations['churn_distribution'] = viz
        
        # Correlaciones
        if 'correlations' in data_dict:
            viz = self.generate_correlation_chart(data_dict['correlations'])
            if viz:
                visualizations['correlations'] = viz
        
        # Importancia de características
        if 'feature_importances' in data_dict:
            viz = self.generate_feature_importance_chart(data_dict['feature_importances'])
            if viz:
                visualizations['feature_importance'] = viz
        
        # Distribución de riesgo
        if 'future_predictions' in data_dict:
            viz = self.generate_risk_distribution_chart(data_dict['future_predictions'])
            if viz:
                visualizations['risk_distribution'] = viz
        
        # Rendimiento del modelo
        if 'y_test' in data_dict and 'y_pred_proba' in data_dict:
            viz = self.generate_model_performance_chart(data_dict['y_test'], data_dict['y_pred_proba'])
            if viz:
                visualizations['model_performance'] = viz
        
        return visualizations