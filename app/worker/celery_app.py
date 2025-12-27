"""
Celery Application Configuration
"""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "medical_imaging",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max per task
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time for GPU
    worker_concurrency=1,  # Only 1 worker for GPU tasks
    broker_connection_retry_on_startup=True,  # Fix deprecation warning
    
    # Fix RabbitMQ heartbeat timeout for long-running GPU tasks
    broker_heartbeat=0,  # Disable heartbeat (prevents disconnect during long tasks)
    broker_pool_limit=1,  # Single connection pool
    broker_transport_options={
        'confirm_publish': True,
    },
    
    # Prevent duplicate task execution
    task_acks_late=True,  # Acknowledge task only after completion
    task_reject_on_worker_lost=True,  # Reject task if worker dies
    
    # Route inference task to dedicated queue
    task_routes={
        "app.worker.tasks.process_inference": {"queue": "inference"},
    },
)

if __name__ == "__main__":
    celery_app.start()
