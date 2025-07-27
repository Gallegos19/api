"""
Módulo de procesamiento y preparación de datos
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import PREDICTION_CONFIG


class DataProcessor:
    """Procesador de datos para análisis de churn"""
    
    def __init__(self):
        self.noise_level = PREDICTION_CONFIG['noise_level']
    
    def add_derived_features(self, df):
        """Añadir características derivadas para mejorar el análisis"""
        df = df.copy()
        
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
    
    def add_realistic_noise(self, df, noise_level=None):
        """Añadir ruido realista para hacer el problema más desafiante"""
        if noise_level is None:
            noise_level = self.noise_level
            
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
        
        # Características SEGURAS (no relacionadas con tiempo reciente)
        safe_features = [
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
            'longest_streak'
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
        df_prep['quiz_success_rate'] = df_prep['total_quiz_completions'] / df_prep['total_quiz_attempts'].replace(0, 1)
        df_prep['quiz_abandon_rate'] = df_prep['total_quiz_abandons'] / df_prep['total_quiz_attempts'].replace(0, 1)
        df_prep['activity_diversity'] = (
            (df_prep['total_quiz_attempts'] > 0).astype(int) +
            (df_prep['total_content_views'] > 0).astype(int) +
            (df_prep['total_challenge_interactions'] > 0).astype(int)
        )
        df_prep['total_activity_volume'] = (
            df_prep['total_quiz_attempts'] + 
            df_prep['total_content_views'] + 
            df_prep['total_challenge_interactions']
        )
        df_prep['activity_per_session'] = df_prep['total_activity_volume'] / df_prep['total_sessions'].replace(0, 1)
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
    
    def split_and_scale_data(self, df, features, target_col='churned'):
        """Dividir y escalar datos para entrenamiento"""
        X = df[features]
        y = df[target_col]
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def get_data_statistics(self, df):
        """Obtener estadísticas básicas del dataset"""
        stats = {
            'total_records': len(df),
            'churn_distribution': df['churned'].value_counts().to_dict() if 'churned' in df.columns else {},
            'churn_rate': df['churned'].mean() * 100 if 'churned' in df.columns else 0,
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_stats': df.describe().to_dict()
        }
        return stats