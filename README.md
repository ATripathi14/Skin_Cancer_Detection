# 🩺 Skin Cancer Detection API

An advanced, high-performance deep learning backend API for skin cancer detection and classification. This project leverages a custom **Vision Transformer (ViT)** model trained on the HAM10000 dataset, wrapped in a lightweight, robust **FastAPI** web application.

---

## 🚀 Key Features

* **Advanced AI Classifier**: Employs a pre-trained **Vision Transformer (ViT-Base)** architecture fine-tuned for dermatological lesions.
* **FastAPI Backend**: Quick, responsive, and fully asynchronous endpoints for high-throughput prediction requests.
* **Rich Diagnostic Output**: Returns predicted category, confidence score, and complete class probabilities for enhanced medical diagnostics support.
* **CORS Configured**: Ready to plug into any modern frontend web dashboard or mobile app.

---

## 📂 Project Structure

```text
Skin_Cancer_Detection/
│
├── backend/
│   ├── main.py            # FastAPI main server & endpoint definitions
│   ├── model.py           # PyTorch loading, preprocessing, & inference logic
│   └── requirements.txt   # Backend python dependencies (FastAPI, PyTorch, timm, etc.)
│
├── vit.pt                 # PyTorch Vision Transformer weights (ignored by git - 343MB)
├── .gitignore             # Git ignore patterns (configured to exclude large model files)
└── README.md              # Project documentation (this file)
```

---

## 🧠 Model Architecture & Class Categories

The backend uses a customized `vit_base_patch16_224` backbone from the `timm` library. The classification head was modified to a custom classification block:
```python
model.head = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, len(CLASSES)) # Supports up to 14 class slots
)
```

### Supported Class Categories
The model classifies images into 7 primary categories based on the **HAM10000** dataset:
1. **akiec**: Actinic keratoses and intraepithelial carcinoma / Bowen's disease
2. **bcc**: Basal cell carcinoma
3. **bkl**: Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)
4. **df**: Dermatofibroma
5. **mel**: Melanoma
6. **nv**: Melanocytic nevi
7. **vasc**: Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)

---

## 🛠️ Getting Started & Setup

### 1. Prerequisites
* **Python**: Python 3.9 - 3.11 is recommended.
* **Weights file (`vit.pt`)**: Ensure you place the fine-tuned model checkpoint file (`vit.pt` of ~343MB) directly in the root directory.

### 2. Environment Setup
Create a virtual environment and activate it:
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate

# Activate venv (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
Navigate to the `backend/` directory and install the required packages:
```bash
pip install -r backend/requirements.txt
```

---

## 🏃 Running the API Server

Start the development server using `uvicorn` from the project root:

```bash
# Navigate to the backend directory
cd backend

# Run the live-reloading FastAPI application
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Once running, you can access:
* **Interactive API Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`
* **Alternative Docs (ReDoc)**: `http://127.0.0.1:8000/redoc`

---

## 🔌 API Endpoints Documentation

### 1. Welcome Status
* **Endpoint**: `GET /`
* **Response**:
  ```json
  {
    "message": "Welcome to the Skin Cancer Detection API!"
  }
  ```

### 2. Predict Image
* **Endpoint**: `POST /predict`
* **Body**: `multipart/form-data` with key `file` containing an image file (PNG/JPEG/JPG).
* **Response example**:
  ```json
  {
    "class_id": 4,
    "class_name": "Melanoma (mel)",
    "confidence": 0.8924,
    "all_probabilities": {
      "Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)": 0.0125,
      "Basal cell carcinoma (bcc)": 0.0310,
      "Benign keratosis-like lesions (bkl)": 0.0241,
      "Dermatofibroma (df)": 0.0053,
      "Melanoma (mel)": 0.8924,
      "Melanocytic nevi (nv)": 0.0312,
      "Vascular lesions (vasc)": 0.0035
    }
  }
  ```

---

## ⚠️ Important Production Considerations

* **CORS Settings**: The backend is currently configured with `allow_origins=["*"]`. Ensure you update this in `backend/main.py` with your staging or production domain.
* **Large Files**: The `vit.pt` model weights file is ignored in `.gitignore`. For production deployments, pull the weights from an external cloud storage provider (e.g., AWS S3, Google Cloud Storage, or Hugging Face Hub) during the build phase.
