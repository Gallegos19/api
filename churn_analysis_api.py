from flask import Flask, jsonify, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from datetime import datetime, timedelta
import os
import io
import base64
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Configuración de la base de datos
DB_CONFIG = {
    'dbname': os.environ.get('DB_NAME', 'postgres'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '6543')
}

class ChurnAnalysisAPI:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.future_model = None
        self.future_scaler_X = None
        self.future_scaler_y = None
        self.feature_names = None
        
        # Configurar estilo de visualización
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def connect_to_db(self):
        """Conectar a la base de datos"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            raise Exception(f"Error conectando a la base de datos: {e}")
    
    def extract_real_user_data(self):
        """Extraer datos REALES de usuarios del data warehouse para análisis de abandono"""
        conn = self.connect_to_db()
        if not conn:
            raise Exception("No se pudo conectar a la base de datos")
        
        # Consulta mejorada para obtener datos reales con métricas completas
        query = """
        WITH user_activity_summary AS (
            SELECT 
                u.user_key,
                u.user_id,
                u.email,
                u.age,
                u.account_status,
                u.is_verified,
                u.registration_date,
                -- Métricas de actividad agregadas (con manejo de nulos)
                COALESCE(AVG(a.avg_session_duration), 0) AS avg_session_duration,
                COALESCE(SUM(a.session_count), 0) AS total_sessions,
                COALESCE(MAX(a.last_activity_date), u.registration_date) AS last_activity_date,
                -- Métricas de engagement (con valores por defecto realistas)
                COALESCE(AVG(e.engagement_score), 50) AS avg_engagement,
                COALESCE(MAX(e.days_since_last_login), 15) AS days_since_last_login,
                COALESCE(MAX(e.current_streak), 0) AS current_streak,
                COALESCE(MAX(e.longest_streak), 0) AS longest_streak,
                COALESCE(AVG(e.quiz_accuracy_percentage), 50) AS quiz_accuracy,
                COALESCE(AVG(e.challenge_completion_rate), 50) AS challenge_completion,
                COALESCE(MAX(e.stagnation_days), 10) AS stagnation_days,
                -- Métricas de quiz
                COALESCE(SUM(a.quiz_attempts), 0) AS total_quiz_attempts,
                COALESCE(SUM(a.quiz_completions), 0) AS total_quiz_completions,
                COALESCE(SUM(a.quiz_abandons), 0) AS total_quiz_abandons,
                -- Métricas de contenido
                COALESCE(SUM(a.content_views), 0) AS total_content_views,
                -- Métricas de desafío
                COALESCE(SUM(a.challenge_interactions), 0) AS total_challenge_interactions
            FROM 
                analytics.dim_users u
                LEFT JOIN analytics.fact_user_activity a ON u.user_key = a.user_key
                LEFT JOIN analytics.fact_user_engagement e ON u.user_key = e.user_key
            WHERE 
                u.account_status IN ('active', 'pending', 'suspended')  -- Incluir diferentes estados
            GROUP BY 
                u.user_key, u.user_id, u.email, u.age, u.account_status, 
                u.is_verified, u.registration_date
        ),
        user_features_with_churn AS (
            SELECT 
                *,
                -- Calcular abandono con lógica compleja y realista
                CASE 
                    WHEN last_activity_date IS NULL THEN 1
                    WHEN last_activity_date < CURRENT_DATE - INTERVAL '14 days' THEN 1
                    WHEN days_since_last_login > 14 THEN 1
                    WHEN account_status = 'suspended' THEN 1
                    -- Casos ambiguos para hacer el problema más difícil
                    WHEN days_since_last_login > 10 AND avg_engagement < 30 THEN 1
                    WHEN days_since_last_login > 7 AND total_sessions < 5 THEN 1
                    WHEN stagnation_days > 20 THEN 1
                    ELSE 0
                END AS churned,
                -- Días desde última actividad
                CASE
                    WHEN last_activity_date IS NULL THEN 30
                    ELSE GREATEST(0, CURRENT_DATE - last_activity_date)
                END AS days_since_last_activity
            FROM user_activity_summary
        )
        SELECT 
            user_key,
            user_id,
            email,
            COALESCE(age, 13) AS age,  -- Edad por defecto
            avg_session_duration,
            total_sessions,
            last_activity_date,
            avg_engagement,
            days_since_last_login,
            current_streak,
            longest_streak,
            quiz_accuracy,
            challenge_completion,
            stagnation_days,
            total_quiz_attempts,
            total_quiz_completions,
            total_quiz_abandons,
            total_content_views,
            total_challenge_interactions,
            churned,
            days_since_last_activity,
            account_status,
            is_verified
        FROM user_features_with_churn
        ORDER BY user_key
        """
        
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Añadir características derivadas para el análisis
            df = self.add_derived_features(df)
            
            # Verificar distribución de churn
            churn_dist = df['churned'].value_counts()
            churn_rate = churn_dist[1] / len(df) * 100 if 1 in churn_dist else 0
            
            return df, {
                'total_records': len(df),
                'churn_distribution': churn_dist.to_dict(),
                'churn_rate': churn_rate
            }
        except Exception as e:
            if conn:
                conn.close()
            raise Exception(f"Error al extraer datos reales: {e}")

    def add_derived_features(self, df):
        """Añadir características derivadas para mejorar el análisis"""
        # Frecuencia de sesiones (sesiones por día)
        df['session_frequency'] = df['total_sessions'] / 30  # Asumiendo 30 días
        
        # Tasa de completitud de quizzes
        df['quiz_completion_rate'] = df['total_quiz_completions'] / df['total_quiz_attempts'].replace(0, 1)
        
        # Tasa de abandono de quizzes
        df['quiz_abandon_rate'] = df['total_quiz_abandons'] / df['total_quiz_attempts'].replace(0, 1)
        
        # Tendencia de engagement (engagement - estancamiento normalizado)
        df['engagement_trend'] = df['avg_engagement'] - (df['stagnation_days'] / 30 * 100)
        
        # Diversidad de actividades
        df['activity_diversity'] = (
            (df['total_quiz_attempts'] > 0).astype(int) +
            (df['total_content_views'] > 0).astype(int) +
            (df['total_challenge_interactions'] > 0).astype(int)
        )
        
        # Ratio de éxito en desafíos
        df['challenge_success_ratio'] = df['challenge_completion'] / 100
        
        # Índice de consistencia (streak actual / streak más largo)
        df['consistency_index'] = df['current_streak'] / df['longest_streak'].replace(0, 1)
        
        return df
    
    def create_realistic_churn_definition(self):
        """Crear una definición de churn más realista y desafiante"""
        conn = self.connect_to_db()
        if not conn:
            return None
        
        # Nueva consulta con definición de churn más sutil
        query = """
        WITH user_activity_summary AS (
            SELECT 
                u.user_key,
                u.user_id,
                u.email,
                u.age,
                u.account_status,
                u.registration_date,
                -- Métricas básicas
                COALESCE(AVG(a.avg_session_duration), 0) AS avg_session_duration,
                COALESCE(SUM(a.session_count), 0) AS total_sessions,
                COALESCE(MAX(a.last_activity_date), u.registration_date) AS last_activity_date,
                COALESCE(AVG(e.engagement_score), 50) AS avg_engagement,
                COALESCE(MAX(e.current_streak), 0) AS current_streak,
                COALESCE(MAX(e.longest_streak), 0) AS longest_streak,
                COALESCE(AVG(e.quiz_accuracy_percentage), 50) AS quiz_accuracy,
                COALESCE(AVG(e.challenge_completion_rate), 50) AS challenge_completion,
                COALESCE(SUM(a.quiz_attempts), 0) AS total_quiz_attempts,
                COALESCE(SUM(a.quiz_completions), 0) AS total_quiz_completions,
                COALESCE(SUM(a.quiz_abandons), 0) AS total_quiz_abandons,
                COALESCE(SUM(a.content_views), 0) AS total_content_views,
                COALESCE(SUM(a.challenge_interactions), 0) AS total_challenge_interactions
            FROM 
                analytics.dim_users u
                LEFT JOIN analytics.fact_user_activity a ON u.user_key = a.user_key
                LEFT JOIN analytics.fact_user_engagement e ON u.user_key = e.user_key
            WHERE 
                u.account_status IN ('active', 'pending', 'suspended')
            GROUP BY 
                u.user_key, u.user_id, u.email, u.age, u.account_status, u.registration_date
        )
        SELECT 
            *,
            -- NUEVA DEFINICIÓN DE CHURN MÁS SUTIL Y REALISTA
            CASE 
                -- Solo usar criterios muy conservadores
                WHEN last_activity_date < CURRENT_DATE - INTERVAL '30 days' THEN 1
                WHEN account_status = 'suspended' THEN 1
                -- Criterio más sutil: baja actividad Y bajo engagement
                WHEN total_sessions < 3 AND avg_engagement < 25 THEN 1
                ELSE 0
            END AS churned_realistic,
            
            -- También crear etiquetas de riesgo (más útil para negocio)
            CASE 
                WHEN last_activity_date < CURRENT_DATE - INTERVAL '21 days' THEN 'high_risk'
                WHEN last_activity_date < CURRENT_DATE - INTERVAL '14 days' THEN 'medium_risk'
                WHEN avg_engagement < 30 AND total_sessions < 5 THEN 'low_engagement'
                ELSE 'active'
            END AS risk_category
        FROM user_activity_summary
        ORDER BY user_key
        """
        
        try:
            df_realistic = pd.read_sql(query, conn)
            conn.close()
            
            # Estadísticas de la nueva definición
            churn_dist = df_realistic['churned_realistic'].value_counts()
            churn_rate = churn_dist[1] / len(df_realistic) * 100 if 1 in churn_dist else 0
            
            # Estadísticas de categorías de riesgo
            risk_dist = df_realistic['risk_category'].value_counts()
            
            return df_realistic, {
                'churn_distribution': churn_dist.to_dict(),
                'churn_rate': churn_rate,
                'risk_distribution': risk_dist.to_dict()
            }
            
        except Exception as e:
            if conn:
                conn.close()
            raise Exception(f"Error en definición realista de churn: {e}")

    def add_realistic_noise_to_data(self, df, noise_level=0.1):
        """Añadir ruido realista para hacer el problema más desafiante"""
        df_noisy = df.copy()
        
        # Añadir ruido a características numéricas
        numeric_cols = ['quiz_accuracy', 'challenge_completion', 'avg_engagement', 
                       'avg_session_duration', 'total_sessions']
        
        for col in numeric_cols:
            if col in df_noisy.columns:
                # Ruido gaussiano proporcional al valor
                noise = np.random.normal(0, df_noisy[col].std() * noise_level, len(df_noisy))
                df_noisy[col] = df_noisy[col] + noise
                # Mantener valores en rangos realistas
                if col in ['quiz_accuracy', 'challenge_completion', 'avg_engagement']:
                    df_noisy[col] = np.clip(df_noisy[col], 0, 100)
                else:
                    df_noisy[col] = np.maximum(df_noisy[col], 0)
        
        # Flip aleatorio de algunas etiquetas (simular incertidumbre real)
        flip_rate = 0.05  # 5% de etiquetas incorrectas
        n_flips = int(len(df_noisy) * flip_rate)
        flip_indices = np.random.choice(len(df_noisy), n_flips, replace=False)
        df_noisy.loc[flip_indices, 'churned'] = 1 - df_noisy.loc[flip_indices, 'churned']
        
        return df_noisy, n_flips

    def prepare_features_ultra_conservative(self, df):
        """Preparar características eliminando TODO posible data leakage"""
        df_prep = df.copy()
        
        # SOLO usar características que NO pueden estar relacionadas con la definición de churn
        # Eliminar TODAS las características temporales y de actividad reciente
        
        # Características SEGURAS (no relacionadas con tiempo reciente)
        safe_features = [
            # Características demográficas
            'age',
            
            # Métricas históricas acumuladas (totales, no recientes)
            'total_quiz_attempts',
            'total_quiz_completions', 
            'total_quiz_abandons',
            'total_content_views',
            'total_challenge_interactions',
            'total_sessions',
            
            # Métricas de rendimiento (no temporales)
            'quiz_accuracy',
            'challenge_completion',
            'avg_session_duration',
            
            # Streaks (pueden ser históricos)
            'longest_streak'  # Solo el más largo, no el actual
        ]
        
        # Manejar valores nulos de forma conservadora
        for col in safe_features:
            if col in df_prep.columns:
                if col == 'age':
                    df_prep[col].fillna(df_prep[col].median(), inplace=True)
                elif col in ['quiz_accuracy', 'challenge_completion', 'avg_session_duration']:
                    df_prep[col].fillna(df_prep[col].median(), inplace=True)
                else:
                    df_prep[col].fillna(0, inplace=True)
        
        # Crear características derivadas SEGURAS
        # Ratios de rendimiento (no temporales)
        df_prep['quiz_success_rate'] = df_prep['total_quiz_completions'] / df_prep['total_quiz_attempts'].replace(0, 1)
        df_prep['quiz_abandon_rate'] = df_prep['total_quiz_abandons'] / df_prep['total_quiz_attempts'].replace(0, 1)
        
        # Diversidad de actividades
        df_prep['activity_diversity'] = (
            (df_prep['total_quiz_attempts'] > 0).astype(int) +
            (df_prep['total_content_views'] > 0).astype(int) +
            (df_prep['total_challenge_interactions'] > 0).astype(int)
        )
        
        # Volumen total de actividad
        df_prep['total_activity_volume'] = (
            df_prep['total_quiz_attempts'] + 
            df_prep['total_content_views'] + 
            df_prep['total_challenge_interactions']
        )
        
        # Intensidad promedio (actividad por sesión)
        df_prep['activity_per_session'] = df_prep['total_activity_volume'] / df_prep['total_sessions'].replace(0, 1)
        
        # Eficiencia en quizzes
        df_prep['quiz_efficiency'] = df_prep['total_quiz_completions'] / df_prep['total_sessions'].replace(0, 1)
        
        # Lista final de características ULTRA SEGURAS
        ultra_safe_features = [
            'age',
            'total_quiz_attempts',
            'total_quiz_completions', 
            'total_quiz_abandons',
            'total_content_views',
            'total_challenge_interactions',
            'total_sessions',
            'quiz_accuracy',
            'challenge_completion',
            'avg_session_duration',
            'longest_streak',
            'quiz_success_rate',
            'quiz_abandon_rate',
            'activity_diversity',
            'total_activity_volume',
            'activity_per_session',
            'quiz_efficiency'
        ]
        
        # Verificar que existen
        available_features = [f for f in ultra_safe_features if f in df_prep.columns]
        
        # Verificar correlaciones
        X_temp = df_prep[available_features]
        y_temp = df_prep['churned']
        correlations = X_temp.corrwith(y_temp).sort_values(key=abs, ascending=False)
        
        # Eliminar características con correlación muy alta (posible leakage residual)
        high_corr_features = correlations[correlations.abs() > 0.7].index.tolist()
        if high_corr_features:
            available_features = [f for f in available_features if f not in high_corr_features]
        
        return df_prep, available_features, correlations

    def train_ultra_conservative_model(self, X_train, y_train, X_test, y_test, feature_names):
        """Entrenar modelo ultra conservador"""
        from sklearn.metrics import roc_curve
        
        # Modelo MUY conservador
        rf_ultra = RandomForestClassifier(
            n_estimators=50,        # Reducido
            max_depth=5,           # Muy limitado
            min_samples_split=20,  # Muy alto
            min_samples_leaf=10,   # Muy alto
            max_features=0.5,      # Limitado
            random_state=42,
            class_weight='balanced'
        )
        
        # Entrenar
        rf_ultra.fit(X_train, y_train)
        
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

    def advanced_model_diagnosis(self, model, X_test, y_test, y_pred_proba, feature_names):
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
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
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

    def create_user_sequences(self, users_df, sequence_length=30):
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
            
            for day in range(-sequence_length, 0):  # Últimos 30 días
                # Usar las características reales del usuario como base
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
                'churned_status': get_user_value('churned', 0)  # Añadir estado de churn
            })
        
        return user_sequences, user_info

    def prepare_sequences_for_ml_model(self, user_sequences, prediction_days=15):
        """Preparar secuencias para modelos de scikit-learn"""
        feature_names = ['session_duration', 'sessions_count', 'engagement_score', 
                        'quiz_attempts', 'quiz_completions', 'content_views', 
                        'challenge_interactions', 'login_streak', 'days_since_login']
        
        X_features = []
        y_targets = []
        
        for sequence in user_sequences:
            if len(sequence) >= 30:
                # Usar los primeros 15 días como características
                first_half = sequence[:15]
                second_half = sequence[15:]
                
                # Crear características agregadas de los primeros 15 días
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
                
                # Objetivos: valores de los siguientes 15 días
                targets = []
                for day in second_half:
                    for feature in feature_names:
                        targets.append(day[feature])
                
                y_targets.append(targets)
        
        return np.array(X_features), np.array(y_targets), feature_names

    def train_future_prediction_model_ml(self, user_sequences, prediction_days=15):
        """Entrenar modelo para predicción futura usando scikit-learn"""
        # Preparar datos
        X, y, feature_names = self.prepare_sequences_for_ml_model(user_sequences, prediction_days)
        
        if len(X) == 0:
            raise Exception("No hay suficientes datos para entrenar el modelo")
        
        # Dividir en entrenamiento y validación
        split_idx = max(1, int(0.8 * len(X)))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Escalar características
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val) if len(X_val) > 0 else np.array([])
        
        # Escalar objetivos
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val) if len(y_val) > 0 else np.array([])
        
        # Entrenar modelo multioutput
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model = MultiOutputRegressor(base_model)
        model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluar modelo si hay datos de validación
        metrics = {}
        if len(X_val) > 0:
            y_pred_scaled = model.predict(X_val_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_val_original = scaler_y.inverse_transform(y_val_scaled)
            
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
        
        return model, scaler_X, scaler_y, feature_names, metrics

# Instancia global de la clase
churn_analyzer = ChurnAnalysisAPI()

@app.route('/api/churn-analysis/full-analysis', methods=['POST'])
def full_churn_analysis():
    """
    Endpoint principal que ejecuta el análisis completo de churn
    Retorna: JSON con resultados + imágenes en base64
    """
    try:
        # Ejecutar análisis completo
        results = churn_analyzer.run_complete_analysis()
        return jsonify({
            'success': True,
            'data': results,
            'message': 'Análisis de churn completado exitosamente'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error durante el análisis de churn'
        }), 500

    @app.route('/api/churn-analysis/predictions', methods=['GET'])
    def get_predictions():
        """
        Obtener predicciones de usuarios de alto riesgo
        """
        try:
            # Lógica para obtener predicciones
            predictions = churn_analyzer.get_high_risk_predictions()
            return jsonify({
                'success': True,
                'data': predictions,
                'message': 'Predicciones obtenidas exitosamente'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/churn-analysis/visualizations', methods=['GET'])
    def get_visualizations():
        """
        Obtener visualizaciones como imágenes en base64
        """
        try:
            # Generar visualizaciones
            images = churn_analyzer.generate_visualizations()
            return jsonify({
                'success': True,
                'data': {
                    'images': images
                },
                'message': 'Visualizaciones generadas exitosamente'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=5001)    
    def predict_user_future_behavior_ml(self, model, user_sequences, scaler_X, scaler_y, 
                                        feature_names, user_info, prediction_days=15):
        """Predecir comportamiento futuro usando modelo de scikit-learn"""
        # Preparar datos para predicción
        X, _, _ = self.prepare_sequences_for_ml_model(user_sequences, prediction_days)
        
        if len(X) == 0:
            raise Exception("No hay datos suficientes para hacer predicciones")
        
        # Escalar características
        X_scaled = scaler_X.transform(X)
        
        # Hacer predicciones
        predictions_scaled = model.predict(X_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # Convertir predicciones a formato estructurado
        future_predictions = []
        
        for user_idx in range(len(user_info)):
            if user_idx < len(predictions):
                user_pred = predictions[user_idx]
                
                # Reshape predicciones
                user_pred_reshaped = user_pred.reshape(prediction_days, len(feature_names))
                
                for day in range(prediction_days):
                    day_prediction = {
                        'user_id': user_info[user_idx]['user_id'],
                        'user_key': user_info[user_idx]['user_key'],
                        'prediction_day': day + 1,
                        'predicted_date': (datetime.now() + timedelta(days=day + 1)).isoformat(),
                        'current_churned_status': user_info[user_idx]['churned_status']
                    }
                    
                    # Añadir predicciones de cada característica
                    for feat_idx, feature in enumerate(feature_names):
                        day_prediction[f'predicted_{feature}'] = max(0, float(user_pred_reshaped[day, feat_idx]))
                    
                    # Calcular métricas derivadas
                    day_prediction['predicted_churn_risk'] = self.calculate_churn_risk_from_metrics(day_prediction)
                    day_prediction['predicted_engagement_level'] = self.calculate_engagement_level(day_prediction)
                    
                    future_predictions.append(day_prediction)
        
        return future_predictions

    def calculate_churn_risk_from_metrics(self, day_prediction):
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

    def calculate_engagement_level(self, day_prediction):
        """Calcular nivel de engagement basado en métricas predichas"""
        engagement_score = day_prediction['predicted_engagement_score']
        
        if engagement_score >= 70:
            return 'Alto'
        elif engagement_score >= 40:
            return 'Medio'
        else:
            return 'Bajo'

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
            
            # Recomendaciones especiales para usuarios ya con churn
            if current_status == 1:
                user_recommendations['recommendations'].append("🔄 REACTIVACIÓN: Usuario ya abandonó - campaña de reactivación")
                user_recommendations['recommendations'].append("🎁 Ofrecer incentivos especiales de regreso")
                user_recommendations['recommendations'].append("📞 Contacto directo para entender motivos de abandono")
            
            # Recomendaciones basadas en riesgo futuro
            if avg_churn_risk > 0.7:
                user_recommendations['recommendations'].append("🚨 CRÍTICO: Contacto inmediato requerido")
                user_recommendations['recommendations'].append("📞 Llamada telefónica de retención")
                user_recommendations['recommendations'].append("🎁 Ofrecer incentivo especial")
            elif avg_churn_risk > 0.5:
                user_recommendations['recommendations'].append("⚠️ ALTO RIESGO: Enviar contenido motivacional")
                user_recommendations['recommendations'].append("👥 Invitar a desafíos grupales")
                user_recommendations['recommendations'].append("📧 Email personalizado del tutor")
            elif avg_churn_risk > 0.3:
                user_recommendations['recommendations'].append("📱 RIESGO MODERADO: Recordatorios de actividad")
                user_recommendations['recommendations'].append("🏆 Mostrar progreso y logros")
            
            # Recomendaciones basadas en engagement
            if avg_engagement < 30:
                user_recommendations['recommendations'].append("📚 Revisar dificultad del contenido")
                user_recommendations['recommendations'].append("🎮 Ofrecer contenido más interactivo")
            
            # Recomendaciones basadas en tendencia
            if trend_engagement < -10:
                user_recommendations['recommendations'].append("📉 Engagement decreciente: Intervención urgente")
                user_recommendations['recommendations'].append("🔄 Cambiar estrategia de contenido")
            
            # Recomendaciones basadas en actividad predicha
            low_activity_days = len(user_data[user_data['predicted_sessions_count'] == 0])
            if low_activity_days > 7:
                user_recommendations['recommendations'].append("🔔 Implementar notificaciones push")
                user_recommendations['recommendations'].append("💰 Ofrecer incentivos de reactivación")
            
            recommendations.append(user_recommendations)
        
        return recommendations

    def generate_visualizations(self, future_predictions, user_ids_to_show=5):
        """Generar visualizaciones como imágenes en base64"""
        if len(future_predictions) == 0:
            return {}
        
        df_predictions = pd.DataFrame(future_predictions)
        selected_users = df_predictions['user_id'].unique()[:user_ids_to_show]
        
        images = {}
        
        # Configurar matplotlib para no mostrar ventanas
        plt.ioff()
        
        # 1. Gráfico de riesgo de abandono
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Riesgo de abandono a lo largo del tiempo
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
        
        # Engagement predicho
        for user_id in selected_users:
            user_data = df_predictions[df_predictions['user_id'] == user_id]
            axes[0, 1].plot(user_data['prediction_day'], user_data['predicted_engagement_score'], 
                           marker='s', label=f'Usuario {user_id}', linewidth=2)
        
        axes[0, 1].set_title('Predicción de Engagement (15 días)', fontsize=14)
        axes[0, 1].set_xlabel('Día de Predicción', fontsize=12)
        axes[0, 1].set_ylabel('Score de Engagement', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actividad de sesiones predicha
        for user_id in selected_users:
            user_data = df_predictions[df_predictions['user_id'] == user_id]
            axes[1, 0].plot(user_data['prediction_day'], user_data['predicted_sessions_count'], 
                           marker='^', label=f'Usuario {user_id}', linewidth=2)
        
        axes[1, 0].set_title('Predicción de Sesiones Diarias (15 días)', fontsize=14)
        axes[1, 0].set_xlabel('Día de Predicción', fontsize=12)
        axes[1, 0].set_ylabel('Número de Sesiones', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actividad de quizzes predicha
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
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        images['future_predictions'] = image_base64
        
        return images

    def run_complete_analysis(self):
        """Ejecutar análisis completo de churn"""
        try:
            # 1. Extraer datos
            df, extraction_stats = self.extract_real_user_data()
            self.df = df
            
            # 2. Procesar datos (definición realista de churn o ruido)
            try:
                df_realistic, realistic_stats = self.create_realistic_churn_definition()
                if df_realistic is not None and df_realistic['churned_realistic'].sum() >= 10:
                    df_final = df_realistic.copy()
                    df_final['churned'] = df_final['churned_realistic']
                    df_final = df_final.drop('churned_realistic', axis=1)
                    approach_used = "Nueva definición de churn"
                    processing_stats = realistic_stats
                else:
                    df_noisy, n_flips = self.add_realistic_noise_to_data(df)
                    df_final = df_noisy.copy()
                    approach_used = "Datos con ruido añadido"
                    processing_stats = {'flipped_labels': n_flips}
            except Exception as e:
                df_noisy, n_flips = self.add_realistic_noise_to_data(df)
                df_final = df_noisy.copy()
                approach_used = "Datos con ruido añadido"
                processing_stats = {'flipped_labels': n_flips}
            
            # 3. Preparar características
            df_prepared, selected_features, correlations = self.prepare_features_ultra_conservative(df_final)
            self.selected_features = selected_features
            
            # 4. Dividir datos y entrenar modelo
            X = df_prepared[selected_features]
            y = df_prepared['churned']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scaler = scaler
            
            # 5. Entrenar modelo
            model, y_pred, y_pred_proba, model_metrics = self.train_ultra_conservative_model(
                X_train_scaled, y_train, X_test_scaled, y_test, selected_features
            )
            self.model = model
            
            # 6. Diagnóstico del modelo
            diagnosis = self.advanced_model_diagnosis(model, X_test_scaled, y_test, y_pred_proba, selected_features)
            
            # 7. Predicciones futuras
            user_sequences, user_info = self.create_user_sequences(df_final)
            future_model, future_scaler_X, future_scaler_y, feature_names, future_metrics = self.train_future_prediction_model_ml(
                user_sequences, prediction_days=15
            )
            
            self.future_model = future_model
            self.future_scaler_X = future_scaler_X
            self.future_scaler_y = future_scaler_y
            self.feature_names = feature_names
            
            # 8. Generar predicciones
            future_predictions = self.predict_user_future_behavior_ml(
                future_model, user_sequences, future_scaler_X, future_scaler_y, 
                feature_names, user_info, prediction_days=15
            )
            
            # 9. Generar recomendaciones
            recommendations = self.generate_intervention_recommendations(future_predictions)
            
            # 10. Generar visualizaciones
            visualizations = self.generate_visualizations(future_predictions)
            
            # 11. Crear resumen de riesgo
            df_pred = pd.DataFrame(future_predictions)
            risk_summary = df_pred.groupby('user_id')['predicted_churn_risk'].mean()
            risk_levels = pd.cut(risk_summary, bins=[0, 0.3, 0.6, 1.0], labels=['Bajo', 'Medio', 'Alto'])
            
            current_status_summary = df_pred.groupby(['user_id', 'current_churned_status']).first().reset_index()
            active_users_count = len(current_status_summary[current_status_summary['current_churned_status'] == 0])
            churned_users_count = len(current_status_summary[current_status_summary['current_churned_status'] == 1])
            
            risk_distribution = {}
            for level in ['Bajo', 'Medio', 'Alto']:
                count = (risk_levels == level).sum()
                percentage = count / len(risk_levels) * 100 if len(risk_levels) > 0 else 0
                risk_distribution[level] = {
                    'count': int(count),
                    'percentage': float(percentage)
                }
            
            return {
                'extraction_stats': extraction_stats,
                'processing_stats': processing_stats,
                'approach_used': approach_used,
                'model_metrics': model_metrics,
                'model_diagnosis': diagnosis,
                'future_model_metrics': future_metrics,
                'recommendations': recommendations[:10],  # Primeros 10 usuarios
                'visualizations': visualizations,
                'summary': {
                    'total_users_analyzed': len(user_info),
                    'active_users': active_users_count,
                    'churned_users': churned_users_count,
                    'risk_distribution': risk_distribution,
                    'prediction_days': 15,
                    'total_predictions': len(future_predictions)
                },
                'correlations': correlations.head(10).to_dict()
            }
            
        except Exception as e:
            raise Exception(f"Error en análisis completo: {str(e)}")

# Instancia global de la clase
churn_analyzer = ChurnAnalysisAPI()

@app.route('/api/churn-analysis/full-analysis', methods=['POST'])
def full_churn_analysis():
    """
    Endpoint principal que ejecuta el análisis completo de churn
    Retorna: JSON con resultados + imágenes en base64
    """
    try:
        # Ejecutar análisis completo
        results = churn_analyzer.run_complete_analysis()
        return jsonify({
            'success': True,
            'data': results,
            'message': 'Análisis de churn completado exitosamente'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error durante el análisis de churn'
        }), 500

@app.route('/api/churn-analysis/predictions', methods=['GET'])
def get_predictions():
    """
    Obtener predicciones de usuarios de alto riesgo
    """
    try:
        if churn_analyzer.future_model is None:
            return jsonify({
                'success': False,
                'error': 'Modelo no entrenado. Ejecuta primero el análisis completo.',
                'message': 'Modelo no disponible'
            }), 400
        
        # Obtener datos y generar predicciones
        df, _ = churn_analyzer.extract_real_user_data()
        user_sequences, user_info = churn_analyzer.create_user_sequences(df)
        
        future_predictions = churn_analyzer.predict_user_future_behavior_ml(
            churn_analyzer.future_model, user_sequences, 
            churn_analyzer.future_scaler_X, churn_analyzer.future_scaler_y, 
            churn_analyzer.feature_names, user_info, prediction_days=15
        )
        
        # Filtrar usuarios de alto riesgo
        df_pred = pd.DataFrame(future_predictions)
        high_risk_users = df_pred.groupby('user_id')['predicted_churn_risk'].mean()
        high_risk_users = high_risk_users[high_risk_users > 0.6].sort_values(ascending=False)
        
        return jsonify({
            'success': True,
            'data': {
                'high_risk_users': high_risk_users.head(20).to_dict(),
                'total_high_risk': len(high_risk_users),
                'predictions': future_predictions[:100]  # Primeras 100 predicciones
            },
            'message': 'Predicciones obtenidas exitosamente'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/churn-analysis/visualizations', methods=['GET'])
def get_visualizations():
    """
    Obtener visualizaciones como imágenes en base64
    """
    try:
        if churn_analyzer.future_model is None:
            return jsonify({
                'success': False,
                'error': 'Modelo no entrenado. Ejecuta primero el análisis completo.',
                'message': 'Modelo no disponible'
            }), 400
        
        # Obtener datos y generar predicciones
        df, _ = churn_analyzer.extract_real_user_data()
        user_sequences, user_info = churn_analyzer.create_user_sequences(df)
        
        future_predictions = churn_analyzer.predict_user_future_behavior_ml(
            churn_analyzer.future_model, user_sequences, 
            churn_analyzer.future_scaler_X, churn_analyzer.future_scaler_y, 
            churn_analyzer.feature_names, user_info, prediction_days=15
        )
        
        # Generar visualizaciones
        images = churn_analyzer.generate_visualizations(future_predictions)
        
        return jsonify({
            'success': True,
            'data': {
                'images': images
            },
            'message': 'Visualizaciones generadas exitosamente'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/churn-analysis/recommendations', methods=['GET'])
def get_recommendations():
    """
    Obtener recomendaciones de intervención
    """
    try:
        if churn_analyzer.future_model is None:
            return jsonify({
                'success': False,
                'error': 'Modelo no entrenado. Ejecuta primero el análisis completo.',
                'message': 'Modelo no disponible'
            }), 400
        
        # Obtener datos y generar predicciones
        df, _ = churn_analyzer.extract_real_user_data()
        user_sequences, user_info = churn_analyzer.create_user_sequences(df)
        
        future_predictions = churn_analyzer.predict_user_future_behavior_ml(
            churn_analyzer.future_model, user_sequences, 
            churn_analyzer.future_scaler_X, churn_analyzer.future_scaler_y, 
            churn_analyzer.feature_names, user_info, prediction_days=15
        )
        
        # Generar recomendaciones
        recommendations = churn_analyzer.generate_intervention_recommendations(future_predictions)
        
        # Ordenar por riesgo (mayor a menor)
        recommendations.sort(key=lambda x: x['avg_churn_risk'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': {
                'recommendations': recommendations[:20],  # Top 20 usuarios de mayor riesgo
                'total_users': len(recommendations)
            },
            'message': 'Recomendaciones generadas exitosamente'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/churn-analysis/health', methods=['GET'])
def health_check():
    """
    Verificar el estado de la API
    """
    try:
        # Verificar conexión a la base de datos
        conn = churn_analyzer.connect_to_db()
        if conn:
            conn.close()
            db_status = "connected"
        else:
            db_status = "disconnected"
        
        return jsonify({
            'success': True,
            'data': {
                'status': 'healthy',
                'database': db_status,
                'model_trained': churn_analyzer.model is not None,
                'future_model_trained': churn_analyzer.future_model is not None
            },
            'message': 'API funcionando correctamente'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error en verificación de salud'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)