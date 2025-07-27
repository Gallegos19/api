"""
Módulo de modelos de machine learning
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    mean_absolute_error, mean_squared_error
)
from config import MODEL_CONFIG, PREDICTION_CONFIG


class ChurnModel:
    """Modelo de predicción de churn"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.config = MODEL_CONFIG
    
    def train_ultra_conservative_model(self, X_train, y_train, X_test, y_test, feature_names):
        """Entrenar modelo ultra conservador"""
        # Modelo MUY conservador
        rf_ultra = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            max_features=self.config['max_features'],
            random_state=self.config['random_state'],
            class_weight='balanced'
        )
        
        # Entrenar
        rf_ultra.fit(X_train, y_train)
        self.model = rf_ultra
        self.feature_names = feature_names
        
        # Evaluar
        y_pred_proba = rf_ultra.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Métricas
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Validación cruzada
        cv_scores = cross_val_score(rf_ultra, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Análisis de características más importantes
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_ultra.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return rf_ultra, y_pred, y_pred_proba, {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importances': importances.to_dict('records')
        }
    
    def diagnose_model(self, X_test, y_test, y_pred_proba):
        """Diagnóstico avanzado del modelo"""
        # 1. Análisis de distribución de probabilidades
        prob_stats = {
            'min': float(y_pred_proba.min()),
            'max': float(y_pred_proba.max()),
            'mean': float(y_pred_proba.mean()),
            'std': float(y_pred_proba.std()),
            'median': float(np.median(y_pred_proba))
        }
        
        # 2. Verificar si hay separación perfecta
        churn_probs = y_pred_proba[y_test == 1]
        no_churn_probs = y_pred_proba[y_test == 0]
        
        if len(churn_probs) > 0 and len(no_churn_probs) > 0:
            overlap = np.sum((churn_probs.min() <= no_churn_probs) & (no_churn_probs <= churn_probs.max()))
            total_no_churn = len(no_churn_probs)
            overlap_ratio = overlap / total_no_churn if total_no_churn > 0 else 0
        else:
            overlap_ratio = 0
        
        # 3. Análisis de importancia de características
        importances = None
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # 4. Análisis de casos extremos
        extreme_high = int(np.sum(y_pred_proba > 0.95))
        extreme_low = int(np.sum(y_pred_proba < 0.05))
        total = len(y_pred_proba)
        
        return {
            'prob_stats': prob_stats,
            'overlap_ratio': float(overlap_ratio),
            'extreme_cases': {
                'high': extreme_high,
                'low': extreme_low,
                'total': total,
                'high_percentage': (extreme_high / total * 100) if total > 0 else 0,
                'low_percentage': (extreme_low / total * 100) if total > 0 else 0
            },
            'feature_importances': importances.to_dict('records') if importances is not None else []
        }


class FuturePredictionModel:
    """Modelo de predicción futura de comportamiento"""
    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        self.prediction_days = PREDICTION_CONFIG['prediction_days']
        self.sequence_length = PREDICTION_CONFIG['sequence_length']
    
    def create_user_sequences(self, users_df):
        """Crear secuencias temporales para cada usuario usando datos reales"""
        user_sequences = []
        user_info = []
        
        for user_idx in range(len(users_df)):
            user = users_df.iloc[user_idx]
            
            # Función auxiliar para obtener valores de usuario de forma segura
            def get_user_value(column_name, default_value=0):
                if column_name in user.index and pd.notna(user[column_name]):
                    return user[column_name]
                return default_value
            
            # Simular datos históricos basados en el comportamiento real del usuario
            sequence = []
            
            for day in range(-self.sequence_length, 0):  # Últimos días
                daily_metrics = {
                    'day_offset': day,
                    'session_duration': max(0, get_user_value('avg_session_duration', 15) + np.random.normal(0, 2)),
                    'sessions_count': max(0, int(get_user_value('session_frequency', 1) + np.random.poisson(0.5))),
                    'engagement_score': max(0, min(100, get_user_value('engagement_trend', 50) + np.random.normal(0, 5))),
                    'quiz_attempts': max(0, int(get_user_value('total_quiz_attempts', 5)/30 + np.random.poisson(0.3))),
                    'quiz_completions': 0,
                    'content_views': max(0, int(get_user_value('total_content_views', 10)/30 + np.random.poisson(0.4))),
                    'challenge_interactions': max(0, int(get_user_value('total_challenge_interactions', 3)/30 + np.random.poisson(0.2))),
                    'login_streak': max(0, get_user_value('current_streak', 5) + day + np.random.randint(-1, 2)),
                    'days_since_login': max(0, abs(day) + np.random.randint(0, 1))
                }
                
                # Quiz completions basado en attempts y tasa de completitud real
                if daily_metrics['quiz_attempts'] > 0:
                    completion_rate = get_user_value('quiz_completion_rate', 0.5)
                    daily_metrics['quiz_completions'] = np.random.binomial(
                        daily_metrics['quiz_attempts'], completion_rate
                    )
                
                sequence.append(daily_metrics)
            
            user_sequences.append(sequence)
            
            # Información del usuario usando las columnas reales
            user_info.append({
                'user_id': get_user_value('user_id', f'user_{user_idx}'),
                'user_key': get_user_value('user_key', user_idx),
                'current_engagement': get_user_value('engagement_trend', 50),
                'current_streak': get_user_value('current_streak', 5),
                'churned_status': get_user_value('churned', 0)
            })
        
        return user_sequences, user_info
    
    def prepare_sequences_for_ml_model(self, user_sequences):
        """Preparar secuencias para modelos de scikit-learn"""
        feature_names = ['session_duration', 'sessions_count', 'engagement_score', 
                        'quiz_attempts', 'quiz_completions', 'content_views', 
                        'challenge_interactions', 'login_streak', 'days_since_login']
        
        X_features = []
        y_targets = []
        
        for sequence in user_sequences:
            if len(sequence) >= self.sequence_length:
                # Usar los primeros días como características
                first_half = sequence[:self.prediction_days]
                second_half = sequence[self.prediction_days:]
                
                # Crear características agregadas
                features = []
                for feature in feature_names:
                    values = [day[feature] for day in first_half]
                    # Estadísticas agregadas
                    features.extend([
                        np.mean(values),      # Media
                        np.std(values),       # Desviación estándar
                        np.min(values),       # Mínimo
                        np.max(values),       # Máximo
                        values[-1],           # Último valor
                        np.sum(values),       # Suma total
                        len([v for v in values if v > 0])  # Días con actividad
                    ])
                
                # Añadir tendencias
                for feature in feature_names:
                    values = [day[feature] for day in first_half]
                    if len(values) > 1:
                        x_vals = np.arange(len(values))
                        trend = np.polyfit(x_vals, values, 1)[0]  # Pendiente
                        features.append(trend)
                    else:
                        features.append(0)
                
                X_features.append(features)
                
                # Objetivos: valores de los siguientes días
                targets = []
                for day in second_half:
                    for feature in feature_names:
                        targets.append(day[feature])
                
                y_targets.append(targets)
        
        return np.array(X_features), np.array(y_targets), feature_names
    
    def train_model(self, user_sequences):
        """Entrenar modelo para predicción futura"""
        # Preparar datos
        X, y, feature_names = self.prepare_sequences_for_ml_model(user_sequences)
        
        if len(X) == 0:
            raise Exception("No hay suficientes datos para entrenar el modelo")
        
        self.feature_names = feature_names
        
        # Dividir en entrenamiento y validación
        split_idx = max(1, int(0.8 * len(X)))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Escalar características
        self.scaler_X = MinMaxScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val) if len(X_val) > 0 else np.array([])
        
        # Escalar objetivos
        self.scaler_y = MinMaxScaler()
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val) if len(y_val) > 0 else np.array([])
        
        # Entrenar modelo multioutput
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluar modelo si hay datos de validación
        metrics = {}
        if len(X_val) > 0:
            y_pred_scaled = self.model.predict(X_val_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_val_original = self.scaler_y.inverse_transform(y_val_scaled)
            
            mae = mean_absolute_error(y_val_original, y_pred)
            mse = mean_squared_error(y_val_original, y_pred)
            
            metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'data_shape': {
                    'users': int(X.shape[0]),
                    'features': int(X.shape[1]),
                    'targets': int(y.shape[1])
                }
            }
        
        return metrics
    
    def predict_future_behavior(self, user_sequences, user_info):
        """Predecir comportamiento futuro usando modelo entrenado"""
        if self.model is None:
            raise Exception("Modelo no entrenado")
        
        # Preparar datos para predicción
        X, _, _ = self.prepare_sequences_for_ml_model(user_sequences)
        
        if len(X) == 0:
            raise Exception("No hay datos suficientes para hacer predicciones")
        
        # Escalar características
        X_scaled = self.scaler_X.transform(X)
        
        # Hacer predicciones
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        # Convertir predicciones a formato estructurado
        future_predictions = []
        
        for user_idx in range(len(user_info)):
            if user_idx < len(predictions):
                user_pred = predictions[user_idx]
                
                # Reshape predicciones
                user_pred_reshaped = user_pred.reshape(self.prediction_days, len(self.feature_names))
                
                for day in range(self.prediction_days):
                    day_prediction = {
                        'user_id': user_info[user_idx]['user_id'],
                        'user_key': user_info[user_idx]['user_key'],
                        'prediction_day': day + 1,
                        'predicted_date': (pd.Timestamp.now() + pd.Timedelta(days=day + 1)).isoformat(),
                        'current_churned_status': user_info[user_idx]['churned_status']
                    }
                    
                    # Añadir predicciones de cada característica
                    for feat_idx, feature in enumerate(self.feature_names):
                        day_prediction[f'predicted_{feature}'] = max(0, float(user_pred_reshaped[day, feat_idx]))
                    
                    # Calcular métricas derivadas
                    day_prediction['predicted_churn_risk'] = self._calculate_churn_risk(day_prediction)
                    day_prediction['predicted_engagement_level'] = self._calculate_engagement_level(day_prediction)
                    
                    future_predictions.append(day_prediction)
        
        return future_predictions
    
    def _calculate_churn_risk(self, day_prediction):
        """Calcular riesgo de abandono basado en métricas predichas"""
        risk_score = 0
        
        # Días sin login (mayor peso)
        if day_prediction['predicted_days_since_login'] > 7:
            risk_score += 0.4
        elif day_prediction['predicted_days_since_login'] > 3:
            risk_score += 0.2
        
        # Engagement bajo
        if day_prediction['predicted_engagement_score'] < 30:
            risk_score += 0.3
        elif day_prediction['predicted_engagement_score'] < 50:
            risk_score += 0.1
        
        # Actividad baja
        if day_prediction['predicted_sessions_count'] == 0:
            risk_score += 0.2
        
        if day_prediction['predicted_quiz_attempts'] == 0:
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _calculate_engagement_level(self, day_prediction):
        """Calcular nivel de engagement basado en métricas predichas"""
        engagement_score = day_prediction['predicted_engagement_score']
        
        if engagement_score >= 70:
            return 'Alto'
        elif engagement_score >= 40:
            return 'Medio'
        else:
            return 'Bajo'