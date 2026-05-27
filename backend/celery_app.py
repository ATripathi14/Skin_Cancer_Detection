import os
from celery import Celery

# Redis broker and backend URL config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "skin_cancer_detection",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Automatically expire task results in Redis after 1 hour (3600 seconds)
    result_expires=3600,
    # Register tasks
    imports=["tasks"]
)
