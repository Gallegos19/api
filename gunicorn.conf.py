# Configuración de Gunicorn para Railway
import os

# Configuración del servidor
bind = f"0.0.0.0:{os.environ.get('PORT', 5001)}"
workers = int(os.environ.get('WEB_CONCURRENCY', 2))
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 300  # 5 minutos para análisis largos
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Configuración de proceso
preload_app = True
daemon = False
pidfile = None
tmp_upload_dir = None

# Configuración de memoria
max_requests_jitter = 100
worker_tmp_dir = "/dev/shm"

# Hooks
def when_ready(server):
    server.log.info("🚀 API de Análisis de Churn lista en Railway")

def worker_int(worker):
    worker.log.info("⚠️ Worker recibió INT o QUIT signal")

def pre_fork(server, worker):
    server.log.info(f"👷 Worker spawned (pid: {worker.pid})")

def post_fork(server, worker):
    server.log.info(f"✅ Worker spawned (pid: {worker.pid})")

def worker_abort(worker):
    worker.log.info(f"❌ Worker recibió SIGABRT signal")