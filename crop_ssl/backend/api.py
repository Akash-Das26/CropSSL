"""
CropSSL Backend API.

FastAPI server for model inference, training status, and dataset management.
Runs on localhost with automatic API documentation at /docs.

Usage:
    python -m crop_ssl.backend.api
    # or
    uvicorn crop_ssl.backend.api:app --host 0.0.0.0 --port 8000
"""

import io
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ============================================================
# App Configuration
# ============================================================
app = FastAPI(
    title="CropSSL API",
    description="Cross-Domain Robustness of Self-Supervised Vision Foundation Models for Crop Disease Detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global State
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS: Dict[str, torch.nn.Module] = {}
ACTIVE_MODEL: Optional[str] = None

DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(DISEASE_CLASSES)

# Training job tracking
TRAINING_JOBS: Dict[str, Dict] = {}


# ============================================================
# Request/Response Models
# ============================================================
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_5: List[Dict[str, float]]
    inference_time_ms: float


class ModelInfo(BaseModel):
    name: str
    architecture: str
    parameters: int
    device: str


class TrainingStatus(BaseModel):
    job_id: str
    status: str
    epoch: int
    loss: float
    accuracy: float


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: int
    uptime: float


# ============================================================
# Startup
# ============================================================
START_TIME = time.time()


@app.on_event("startup")
async def startup():
    """Load default model on startup."""
    global ACTIVE_MODEL
    try:
        from crop_ssl.models.ssl import create_ssl_model
        model = create_ssl_model("simclr", backbone="vit_small", embed_dim=384)
        model.eval()
        model.to(DEVICE)
        MODELS["simclr_vit_small"] = model
        ACTIVE_MODEL = "simclr_vit_small"
        print(f"Default model loaded on {DEVICE}")
    except Exception as e:
        print(f"Could not load default model: {e}")


# ============================================================
# Endpoints
# ============================================================
@app.get("/", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        models_loaded=len(MODELS),
        uptime=time.time() - START_TIME,
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded models."""
    infos = []
    for name, model in MODELS.items():
        params = sum(p.numel() for p in model.parameters())
        arch = "SSL"
        if hasattr(model, "student_backbone"):
            arch = "DINOv2"
        elif hasattr(model, "encoder") and hasattr(model, "projector"):
            arch = "SimCLR"
        elif hasattr(model, "query_encoder"):
            arch = "MoCo v3"
        elif hasattr(model, "encoder") and hasattr(model, "decoder_blocks"):
            arch = "MAE"

        infos.append(ModelInfo(
            name=name,
            architecture=arch,
            parameters=params,
            device=str(next(model.parameters()).device),
        ))
    return infos


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model_name: Optional[str] = None):
    """Predict disease from uploaded leaf image."""
    from PIL import Image
    import torchvision.transforms as T

    if model_name and model_name in MODELS:
        model = MODELS[model_name]
    elif ACTIVE_MODEL and ACTIVE_MODEL in MODELS:
        model = MODELS[ACTIVE_MODEL]
    else:
        raise HTTPException(status_code=500, detail="No model loaded")

    # Read and validate image
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty file uploaded")
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "File too large (max 10MB)")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file. Supported: JPG, PNG")

    # Preprocess
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    start = time.time()
    with torch.no_grad():
        if hasattr(model, "encode"):
            features = model.encode(tensor)
            if hasattr(model, "head") and hasattr(model.head, "weight"):
                logits = model.head(features)
            else:
                logits = features
        else:
            logits = model(tensor)

        probs = F.softmax(logits, dim=-1)
    inference_time = (time.time() - start) * 1000

    # Top-5
    top5_probs, top5_idx = probs.topk(5, dim=-1)
    top5 = [
        {"class": DISEASE_CLASSES[idx], "confidence": round(prob.item() * 100, 2)}
        for idx, prob in zip(top5_idx[0], top5_probs[0])
    ]

    return PredictionResponse(
        prediction=DISEASE_CLASSES[top5_idx[0][0].item()],
        confidence=round(top5_probs[0][0].item() * 100, 2),
        top_5=top5,
        inference_time_ms=round(inference_time, 2),
    )


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific SSL model."""
    global ACTIVE_MODEL

    from crop_ssl.models.ssl import create_ssl_model

    # Parse method_backbone (e.g., 'dinov2_vit_base' -> method='dinov2', backbone='vit_base')
    known_methods = ["dinov2", "moco_v3", "simclr", "mae"]
    method = "simclr"
    backbone = "vit_small"
    for m in known_methods:
        if model_name.startswith(m):
            method = m
            remainder = model_name[len(m):].lstrip("_")
            backbone = remainder if remainder else "vit_small"
            break

    embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
    embed_dim = embed_dims.get(backbone, 384)

    try:
        model = create_ssl_model(method, backbone=backbone, embed_dim=embed_dim)
        model.eval()
        model.to(DEVICE)
        MODELS[model_name] = model
        ACTIVE_MODEL = model_name
        return {"status": "loaded", "model": model_name, "device": DEVICE}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory."""
    if model_name in MODELS:
        del MODELS[model_name]
        if ACTIVE_MODEL == model_name:
            ACTIVE_MODEL = list(MODELS.keys())[0] if MODELS else None
        return {"status": "unloaded", "model": model_name}
    raise HTTPException(status_code=404, detail="Model not found")


@app.get("/classes")
async def list_classes():
    """List all disease classes."""
    return {"classes": DISEASE_CLASSES, "count": NUM_CLASSES}


@app.get("/training/status")
async def training_status():
    """Get status of all training jobs."""
    return {"jobs": list(TRAINING_JOBS.values())}


@app.post("/training/start")
async def start_training(
    method: str = "simclr",
    backbone: str = "vit_small",
    epochs: int = 10,
    lr: float = 1e-4,
):
    # Validate inputs
    if method not in ["simclr", "dinov2", "moco_v3", "mae"]:
        raise HTTPException(400, f"Unknown method: {method}")
    if backbone not in ["vit_small", "vit_base", "vit_large"]:
        raise HTTPException(400, f"Unknown backbone: {backbone}")
    epochs = max(1, min(epochs, 100))
    lr = max(1e-6, min(lr, 1.0))
    """Start a training job (background)."""
    job_id = str(uuid.uuid4())[:8]
    TRAINING_JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "method": method,
        "backbone": backbone,
        "epoch": 0,
        "total_epochs": epochs,
        "loss": 0.0,
        "accuracy": 0.0,
    }

    def train_background():
        try:
            from crop_ssl.models.ssl import create_ssl_model
            from torch.utils.data import TensorDataset, DataLoader

            embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
            model = create_ssl_model(
                method, backbone=backbone, embed_dim=embed_dims.get(backbone, 384)
            )
            model.to(DEVICE)
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            ds = TensorDataset(torch.randn(100, 3, 224, 224), torch.zeros(100))
            loader = DataLoader(ds, batch_size=16, shuffle=True)

            for epoch in range(epochs):
                total_loss = 0.0
                n = 0
                for images, _ in loader:
                    images = images.to(DEVICE)
                    if method in ("simclr", "moco_v3"):
                        result = model(images, torch.randn_like(images))
                    elif method == "mae":
                        result = model(images)
                    else:
                        crops = [images] + [torch.randn_like(images) for _ in range(9)]
                        result = model(crops)

                    optimizer.zero_grad()
                    result["loss"].backward()
                    optimizer.step()
                    total_loss += result["loss"].item()
                    n += 1

                TRAINING_JOBS[job_id]["epoch"] = epoch + 1
                TRAINING_JOBS[job_id]["loss"] = round(total_loss / max(n, 1), 4)

            TRAINING_JOBS[job_id]["status"] = "completed"

            # Store trained model
            model.eval()
            MODELS[f"{method}_{backbone}_trained"] = model

        except Exception as e:
            TRAINING_JOBS[job_id]["status"] = "failed"
            TRAINING_JOBS[job_id]["error"] = str(e)

    import threading
    thread = threading.Thread(target=train_background, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "started"}


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
