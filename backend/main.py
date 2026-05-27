import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from celery_app import celery_app
from tasks import predict_image_task

app = FastAPI(
    title="Skin Cancer Detection API",
    description="Decoupled FastAPI backend for Skin Cancer Detection, providing asynchronous AI predictions via ViT and Celery.",
    version="1.0.0"
)

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, this should be restricted to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the uploads folder exists inside the workspace backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Skin Cancer Detection API!"}

@app.post("/predict", status_code=status.HTTP_202_ACCEPTED)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="File provided is not an image."
        )
    
    try:
        # Generate a unique task_id
        task_id = str(uuid.uuid4())
        
        # Determine the file extension and construct output path
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".jpg"
        file_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
        
        # Read uploaded image bytes and write to the secure uploads directory
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Dispatch the asynchronous task to Celery
        predict_image_task.delay(task_id, file_path)
        
        return {
            "task_id": task_id,
            "status": "PENDING",
            "message": "Model inference task has been queued successfully."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@app.get("/api/v1/tasks/{task_id}")
def get_task_status(task_id: str):
    try:
        res = AsyncResult(task_id, app=celery_app)
        
        response_data = {
            "task_id": task_id,
            "status": res.state,
        }
        
        if res.state == "SUCCESS":
            response_data["result"] = res.result
        elif res.state == "FAILURE":
            # res.info contains the exception instance or string representation
            response_data["error"] = str(res.info)
        else:
            response_data["result"] = None
            
        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch task status: {str(e)}"
        )
