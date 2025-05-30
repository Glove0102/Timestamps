
# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000

# Timeout settings
timeout = 300  # 5 minutes for long-running OpenAI API calls
keepalive = 2
graceful_timeout = 30

# Restart workers
max_requests = 1000
max_requests_jitter = 100

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "srt_timestamps_app"

# Server mechanics
preload_app = True
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (not needed for development)
keyfile = None
certfile = None

# Worker recycling
worker_tmp_dir = "/dev/shm"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
