import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from celery.result import AsyncResult
from pydantic import BaseModel

# Import local database and auth modules
from database import get_db, engine
import db_models
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user
)
from celery_app import celery_app
from tasks import predict_image_task

# Initialize database tables on application load
db_models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Skin Cancer Detection API",
    description="Asynchronous & Authenticated FastAPI backend for Skin Cancer Detection, leveraging ViT, Celery, and SQLite.",
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

# Custom HTTP Middleware to disable automatic documentation in production mode
@app.middleware("http")
async def disable_docs_in_production(request: Request, call_next):
    if os.getenv("ENV") == "production":
        forbidden_paths = ["/docs", "/redoc", "/openapi.json"]
        if any(request.url.path.startswith(p) for p in forbidden_paths):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "Not Found"}
            )
    return await call_next(request)

# Ensure the uploads folder exists inside the workspace backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic schemas for user registration
class UserCreate(BaseModel):
    username: str
    password: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Skin Cancer Detection API!"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Registers a new user account, secure-hashing their password using bcrypt.
    """
    # Check if username is already taken
    existing_user = db.query(db_models.User).filter(db_models.User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Hash password and store in SQLite
    hashed_pwd = get_password_hash(user_in.password)
    new_user = db_models.User(
        username=user_in.username,
        hashed_password=hashed_pwd
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"username": new_user.username, "message": "User registered successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Authenticates user credentials and returns a secure JWT access token.
    Compatible with standard OAuth2 password schemes and Swagger interactive testing.
    """
    # Authenticate credentials
    user = db.query(db_models.User).filter(db_models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate JWT Access Token
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", status_code=status.HTTP_202_ACCEPTED)
async def predict(
    file: UploadFile = File(...),
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Protected skin cancer classification endpoint. Saves images locally, maps
    task IDs to the authenticated user ID in SQLite, and dispatches Celery workers.
    Requires Bearer Token authentication.
    """
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
        
        # Record task ownership in SQLite database BEFORE dispatching to worker
        db_task = db_models.PredictionTask(
            task_id=task_id,
            user_id=current_user.id
        )
        db.add(db_task)
        db.commit()
        
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
def get_task_status(
    task_id: str,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Protected polling endpoint that checks task status. Access is restricted to
    the user who originally requested the task. Requires Bearer Token.
    """
    # Fetch task ownership from database
    db_task = db.query(db_models.PredictionTask).filter(
        db_models.PredictionTask.task_id == task_id
    ).first()
    
    if not db_task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found."
        )
    
    # Restrict result access to the owner of the task
    if db_task.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this prediction task's results."
        )
        
    try:
        res = AsyncResult(task_id, app=celery_app)
        
        response_data = {
            "task_id": task_id,
            "status": res.state,
        }
        
        if res.state == "SUCCESS":
            response_data["result"] = res.result
        elif res.state == "FAILURE":
            response_data["error"] = str(res.info)
        else:
            response_data["result"] = None
            
        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch task status from Celery: {str(e)}"
        )
