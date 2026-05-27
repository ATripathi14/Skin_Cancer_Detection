import io
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image

# Global variables to hold the loaded model and device
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HAM10000 Class labels (Modify these if your model was trained on different labels)
CLASSES = [
    "Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic nevi (nv)",
    "Vascular lesions (vasc)"
]
# Extend to 14 classes to match the model checkpoint shape
CLASSES.extend([f"Additional Class {i}" for i in range(7, 14)])

def load_model():
    """
    Loads the PyTorch Vision Transformer (ViT) model from disk.
    This function should be called when the application starts.
    """
    global model
    try:
        # Adjust path if needed. Assuming vit.pt is at the project root
        model_path = "../vit.pt" 
        
        # The state_dict keys perfectly match the timm library architecture, 
        # but the classification head was modified during training to a Sequential block.
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.5), # Parameter-less layer at index 0
            nn.Linear(in_features, len(CLASSES)) # Linear layer at index 1
        )
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Model vit.pt loaded successfully with timm architecture.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We don't raise here immediately so the server can still start, 
        # but it will fail on the first prediction request.

def preprocess_image(image_bytes):
    """
    Preprocesses the image for the ViT model.
    """
    # ViT models typically expect 224x224 images and ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0) # Add batch dimension
    return tensor.to(device)

def predict_image(image_bytes):
    """
    Runs the inference on the preprocessed image and returns the prediction.
    """
    global model
    if model is None:
        load_model()
        if model is None:
            raise RuntimeError("Model is not loaded. Cannot make predictions.")
            
    tensor = preprocess_image(image_bytes)
    
    with torch.no_grad():
        outputs = model(tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get the predicted class index
        _, predicted_idx = torch.max(probabilities, 0)
        
        idx = predicted_idx.item()
        confidence = probabilities[idx].item()
        
    return {
        "class_id": idx,
        "class_name": CLASSES[idx] if idx < len(CLASSES) else "Unknown",
        "confidence": float(confidence),
        # Return top 3 predictions for a richer UI experience
        "all_probabilities": {CLASSES[i]: float(probabilities[i].item()) for i in range(len(CLASSES))} if len(probabilities) == len(CLASSES) else {}
    }
