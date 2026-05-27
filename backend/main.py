from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import predict_image

app = FastAPI(
    title="Skin Cancer Detection API",
    description="FastAPI backend for Skin Cancer Detection, providing AI predictions via ViT.",
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Skin Cancer Detection API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read the uploaded image file
        contents = await file.read()
        
        # Run the model prediction
        prediction_result = predict_image(contents)
        
        return prediction_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
