"""
Módulo de conexión y operaciones de base de datos
"""
import psycopg2
import pandas as pd
from config import DB_CONFIG


class DatabaseManager:
    """Gestor de conexiones y consultas a la base de datos"""
    
    def __init__(self):
        self.config = DB_CONFIG
    
    def connect(self):
        """Conectar a la base de datos"""
        try:
            # Si DB_CONFIG es una string (DATABASE_URL), usarla directamente
            if isinstance(self.config, str):
                conn = psycopg2.connect(self.config)
            else:
                # Si es un diccionario, usar parámetros individuales
                conn = psycopg2.connect(**self.config)
            return conn
        except Exception as e:
            raise Exception(f"Error conectando a la base de datos: {e}")
    
    def execute_query(self, query):
        """Ejecutar consulta y retornar DataFrame"""
        conn = self.connect()
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            if conn:
                conn.close()
            raise Exception(f"Error ejecutando consulta: {e}")
    
    def get_user_data(self):
        """Extraer datos REALES de usuarios del data warehouse"""
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
                u.account_status IN ('active', 'pending', 'suspended')
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
            COALESCE(age, 13) AS age,
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
        
        return self.execute_query(query)
    
    def get_realistic_churn_data(self):
        """Obtener datos con definición realista de churn"""
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
        
        return self.execute_query(query)
    
    def test_connection(self):
        """Probar la conexión a la base de datos"""
        try:
            conn = self.connect()
            if conn:
                conn.close()
                return True
            return False
        except:
            return False