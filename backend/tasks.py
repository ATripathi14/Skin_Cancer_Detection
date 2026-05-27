import os
from celery_app import celery_app
from model import predict_image

@celery_app.task(name="tasks.predict_image_task")
def predict_image_task(task_id: str, file_path: str):
    """
    Asynchronous Celery task that:
    1. Reads the uploaded image file.
    2. Runs inference using the PyTorch Vision Transformer model.
    3. Cleans up the image file from the disk to save space.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found at {file_path}")
        
    try:
        # Read the file contents
        with open(file_path, "rb") as f:
            image_bytes = f.read()
            
        # Call the inference logic
        result = predict_image(image_bytes)
        return result
        
    except Exception as e:
        # Re-raise so Celery result backend records the task as FAILURE
        raise e
        
    finally:
        # Always delete the uploaded image file to prevent disk bloat
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_err:
            # Log cleanup errors to stderr/stdout
            print(f"Warning: Failed to clean up file {file_path}: {cleanup_err}")
