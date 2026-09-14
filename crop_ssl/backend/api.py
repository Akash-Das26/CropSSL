"""
CropSSL Backend API — Production-Grade.

FastAPI server for model inference, training management, dataset browsing,
attention visualization, cross-domain analysis, and real-time monitoring.

Usage:
    python -m crop_ssl.backend.api
    # or
    uvicorn crop_ssl.backend.api:app --host 0.0.0.0 --port 8000
"""

import io
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fastapi import Body, Depends, FastAPI, File, HTTPException, Header, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# Field is canonical in pydantic; fastapi >=0.14x stopped re-exporting it
from pydantic import BaseModel, Field

from crop_ssl.backend import auth as auth_module

# ============================================================
# App Configuration
# ============================================================
DISEASE_CLASSES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Blight",
    "Corn Healthy", "Grape Black Rot", "Grape Esca", "Grape Leaf Blight",
    "Grape Healthy", "Orange Greening", "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus",
    "Tomato Healthy",
]

NUM_CLASSES = len(DISEASE_CLASSES)

# ============================================================
# State
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS: Dict[str, torch.nn.Module] = {}
ACTIVE_MODEL: Optional[str] = None
TRAINING_JOBS: Dict[str, Dict] = {}
PREDICTION_LOG: Dict[str, Dict] = {}  # prediction_id -> prediction record (bounded, in-memory)
START_TIME = time.time()


# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default models on startup."""
    global ACTIVE_MODEL
    # Auth must be either configured or explicitly disabled — never silently
    # unsecured. Without this, a forgotten CROPSSL_SECRET would only surface
    # the first time someone tried to log in.
    if not auth_module.ANONYMOUS_MODE and not auth_module.JWT_SECRET:
        raise RuntimeError(
            "CROPSSL_SECRET is not set. Start the server with: "
            "CROPSSL_SECRET=<random-secret> python -m crop_ssl.backend.api "
            "(local dev only: CROPSSL_ALLOW_ANONYMOUS=1 bypasses auth)."
        )
    try:
        from crop_ssl.models.ssl import create_ssl_model
        for method, bb in [("simclr", "vit_small"), ("dinov2", "vit_small")]:
            key = f"{method}_{bb}"
            model = create_ssl_model(method, backbone=bb, embed_dim=384)
            model.eval()
            model.to(DEVICE)
            MODELS[key] = model
            if ACTIVE_MODEL is None:
                ACTIVE_MODEL = key
        print(f"✅ Loaded {len(MODELS)} models on {DEVICE}")
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        # Fail fast: an API that starts with zero models only lies to its
        # load balancer — surface the startup error instead.
        raise

    # Init auth users
    try:
        from crop_ssl.backend.auth import init_users
        init_users()
    except Exception as e:
        print(f"⚠️  Auth init failed: {e}")

    yield
    MODELS.clear()


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="CropSSL API",
    description="Cross-Domain Robustness of Self-Supervised Vision "
                "Foundation Models for Crop Disease Detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # No cookies/credentialed browser flows exist; '*' + credentials is a
    # spec-violating anti-pattern, so credentials stay off.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Rate Limiter Middleware
# ============================================================
from collections import defaultdict, deque
import threading

_rate_limits: Dict[str, List[float]] = defaultdict(list)
_rate_lock = threading.Lock()
RATE_LIMIT_MAX = 60  # requests
RATE_LIMIT_WINDOW = 60  # seconds

_LATENCY: Dict[str, deque] = {}  # route template -> ring buffer of ms timings
_latency_lock = threading.Lock()


@app.middleware("http")
async def rate_limit_and_logging_middleware(request, call_next):
    """Rate limiting + request logging middleware."""
    client_ip = request.client.host if request.client else "unknown"
    start = time.time()

    # Rate limiting
    with _rate_lock:
        now = time.time()
        _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if now - t < RATE_LIMIT_WINDOW]
        # Opportunistic sweep: bound the dict when many one-shot IPs pile up
        # (only prunes clients idle for a full window, so active users are safe).
        if len(_rate_limits) > 10_000:
            for ip in [k for k, ts in _rate_limits.items()
                       if not ts or now - ts[-1] >= RATE_LIMIT_WINDOW]:
                del _rate_limits[ip]
        if len(_rate_limits[client_ip]) >= RATE_LIMIT_MAX:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": RATE_LIMIT_WINDOW},
            )
        _rate_limits[client_ip].append(now)

    response = await call_next(request)
    elapsed = (time.time() - start) * 1000

    # Correlate requests across logs/services (client-provided ID wins)
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    response.headers["X-Request-ID"] = rid

    # Per-route latency sample (matched route template keeps the key space
    # bounded; unmatched paths fall back to their raw path).
    route_tmpl = getattr(request.scope.get("route"), "path", request.url.path)
    with _latency_lock:
        buf = _LATENCY.get(route_tmpl)
        if buf is None:
            if len(_LATENCY) >= 1000:  # bound: 404 scans etc. can't grow it forever
                _LATENCY.pop(next(iter(_LATENCY)))
            buf = _LATENCY[route_tmpl] = deque(maxlen=500)
        buf.append(elapsed)

    # Log non-health requests
    if request.url.path not in ("/", "/health"):
        print(f"  rid={rid} {request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")

    response.headers["X-Process-Time"] = f"{elapsed:.1f}ms"
    return response


# ============================================================
# Request / Response Models
# ============================================================
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_5: List[Dict]
    inference_time_ms: float
    model_used: str
    prediction_id: str  # pass to /feedback to record ground truth


class ModelInfo(BaseModel):
    name: str
    architecture: str
    parameters: int
    device: str


class TrainingRequest(BaseModel):
    method: str = "simclr"
    backbone: str = "vit_small"
    epochs: int = 10
    lr: float = 1e-4


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
    active_model: str
    uptime: float


class AttentionResponse(BaseModel):
    layer_count: int
    attention_shapes: List[List[int]]


class CompareRequest(BaseModel):
    method_a: str = "simclr"
    backbone_a: str = "vit_small"
    method_b: str = "dinov2"
    backbone_b: str = "vit_small"


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    access_token: str  # Alias for client compatibility
    token_type: str = "bearer"
    username: str
    display_name: str
    role: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""


class ABCreateRequest(BaseModel):
    test_name: str
    model_a: str
    model_b: str
    traffic_split: float = 0.5


class PipelineCreateRequest(BaseModel):
    name: str
    ssl_method: str = "simclr"
    backbone: str = "vit_small"
    dataset: str = "plantvillage"
    adaptation: str = "lora"
    target_dataset: str = "plantdoc"
    num_shots: int = 10


class ExportRequest(BaseModel):
    """Body for POST /models/{name}/export (JSON, matching the docs/curl)."""
    opset: int = 14
    input_size: int = 224


class FeedbackRequest(BaseModel):
    """Body for POST /feedback — ground truth for a stored prediction."""
    prediction_id: str
    correct: bool
    confidence: float = 0.0
    predicted_class: Optional[str] = None  # override when the log entry expired
    model_used: Optional[str] = None


class KNNRunRequest(BaseModel):
    """Body for POST /eval/knn — few-shot k-NN / nearest-centroid evaluation.

    Mirrors scripts/onnx_knn.py's CLI so the dashboard can run the same
    evaluation without a shell.
    """
    method: str = "simclr"            # SSL method providing embeddings
    backbone: str = "vit_small"
    num_classes: int = Field(default=5, ge=2, le=50)
    shots: int = Field(default=5, ge=1, le=50)
    k: int = Field(default=0, ge=0, le=50)  # 0 = nearest centroid
    data_root: str = "./data"         # train/<class>/ layout; synthetic fallback if missing
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class KNNClassReport(BaseModel):
    class_index: int
    accuracy: float  # 0-1
    n: int


class KNNRunResponse(BaseModel):
    mode: str                         # "nearest-centroid" or "k-NN (k=...)"
    accuracy: float                   # 0-1 over the query set
    num_support: int
    num_query: int
    per_class: List[KNNClassReport]
    embedding_source: str             # "registry:<name>" or "transient"
    runtime_ms: float


class KNNBundleRequest(BaseModel):
    """Body for POST /models/{name}/knn-bundle — offline on-device classifier.

    Bakes a few-shot support set into a JSON bundle (class centroids + support
    embeddings + preprocessing contract) that the mobile PWA pairs with the
    exported ONNX backbone for fully offline inference.
    """
    num_classes: int = Field(default=5, ge=2, le=20)  # bounded: bundle size ~classes*shots*384 floats
    shots: int = Field(default=5, ge=1, le=20)
    k: int = Field(default=0, ge=0, le=20)  # 0 = nearest centroid
    data_root: str = "./data"               # train/<class>/ layout; synthetic fallback if missing
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class KNNBundleResponse(BaseModel):
    status: str
    model: str
    path: str
    size_mb: float
    mode: str
    k: int
    num_classes: int
    shots: int
    num_support: int
    embed_dim: int
    source: str                        # "train_dir" or "synthetic"
    checked_with: str                  # "onnxruntime" or "pytorch-fallback"
    download_url: str


# ============================================================
# Helpers
# ============================================================
def _get_model(name: Optional[str] = None) -> torch.nn.Module:
    """Get a model by name or active model."""
    if name and name in MODELS:
        return MODELS[name]
    if ACTIVE_MODEL and ACTIVE_MODEL in MODELS:
        return MODELS[ACTIVE_MODEL]
    raise HTTPException(status_code=503, detail="No model loaded")


def require_admin(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Dict:
    """Auth dependency for sensitive (write/admin) routes.

    Accepts, in order:
    1. 'Authorization: ApiKey <key>' or 'X-API-Key: <key>' when the operator
       set CROPSSL_API_KEY (machine clients, CI jobs) — admin principal.
    2. 'Authorization: Bearer <token>' from /auth/login, or '?token=<token>'.
    3. CROPSSL_ALLOW_ANONYMOUS=1 (local dev) treats header-less requests
       as admin.
    """
    if auth_module.ANONYMOUS_MODE and not authorization and not token and not x_api_key:
        return {"username": "anonymous", "role": "admin"}
    # API-key scheme (before Bearer parsing — different credential type)
    api_key = None
    if authorization and authorization.split(" ", 1)[0].lower() == "apikey":
        api_key = authorization.split(" ", 1)[1].strip() if " " in authorization else ""
    elif x_api_key:
        api_key = x_api_key.strip()
    if api_key is not None:
        if auth_module.verify_api_key(api_key):
            return {"username": "api-key", "role": "admin"}
        raise HTTPException(401, "Invalid API key")
    auth_token = token
    if not auth_token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            auth_token = parts[1]
        else:
            auth_token = authorization
    if not auth_token:
        raise HTTPException(
            401,
            "Authentication required: log in via /auth/login and send "
            "Authorization: Bearer <token>, or set CROPSSL_API_KEY and send "
            "X-API-Key (dev bypass: CROPSSL_ALLOW_ANONYMOUS=1)",
        )
    try:
        payload = auth_module.verify_token(auth_token)
    except RuntimeError as e:
        # Server started without CROPSSL_SECRET: auth is disabled, not broken
        raise HTTPException(503, str(e))
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    return payload


def _preprocess_image(contents: bytes):
    """Decode and preprocess an uploaded image."""
    from PIL import Image
    import torchvision.transforms as T

    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image. Supported: JPG, PNG")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# ============================================================
# Auth Endpoints
# ============================================================
@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Authenticate user and return JWT token."""
    from crop_ssl.backend.auth import authenticate_user, create_token
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    try:
        token = create_token(user["username"], user["role"])
    except RuntimeError as e:
        # Server started without CROPSSL_SECRET: auth is disabled, not broken
        raise HTTPException(503, str(e))
    return LoginResponse(
        token=token,
        access_token=token,
        username=user["username"],
        display_name=user["display_name"],
        role=user["role"],
    )


@app.post("/auth/register")
async def register(req: RegisterRequest):
    """Register a new user account."""
    from crop_ssl.backend.auth import create_user
    if len(req.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    success = create_user(req.username, req.password, req.display_name)
    if not success:
        raise HTTPException(409, "Username already exists")
    return {"status": "registered", "username": req.username}


@app.get("/auth/me")
async def get_current_user(
    token: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """Get current user info from token (query param or Authorization header)."""
    from crop_ssl.backend.auth import verify_token, get_user
    # Accept token from query param or Authorization header
    auth_token = token
    if not auth_token and authorization:
        # Support 'Bearer <token>' format
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == 'bearer':
            auth_token = parts[1]
        else:
            auth_token = authorization
    if not auth_token:
        raise HTTPException(401, "Token required")
    payload = verify_token(auth_token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    user = get_user(payload["username"])
    if not user:
        raise HTTPException(404, "User not found")
    return {
        "username": payload["username"],
        "display_name": user.get("display_name", payload["username"]),
        "role": user.get("role", "viewer"),
    }


@app.get("/auth/users")
async def list_all_users(user_payload: Dict = Depends(require_admin)):
    """List all registered users (admin only; hashes are never returned)."""
    if user_payload.get("role") != "admin":
        raise HTTPException(403, "Admin role required")
    from crop_ssl.backend.auth import list_users
    return {"users": list_users()}


# ============================================================
# Core Endpoints
# ============================================================
@app.get("/", response_model=HealthResponse)
async def root():
    if not MODELS:
        # A backend with no models cannot serve predictions: report 503 so
        # orchestrators stop routing traffic here.
        raise HTTPException(status_code=503, detail={
            "status": "unavailable", "reason": "no models loaded", "device": DEVICE,
        })
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        models_loaded=len(MODELS),
        active_model=ACTIVE_MODEL or "none",
        uptime=round(time.time() - START_TIME, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    if not MODELS:
        raise HTTPException(status_code=503, detail={
            "status": "unavailable", "reason": "no models loaded", "device": DEVICE,
        })
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        models_loaded=len(MODELS),
        active_model=ACTIVE_MODEL or "none",
        uptime=round(time.time() - START_TIME, 1),
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
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
            name=name, architecture=arch, parameters=params,
            device=str(next(model.parameters()).device),
        ))
    return infos


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
):
    """Predict disease from uploaded leaf image."""
    global ACTIVE_MODEL
    contents = await file.read()
    tensor = _preprocess_image(contents)
    tensor = tensor.to(DEVICE)

    try:
        model = _get_model(model_name)
    except HTTPException:
        # If no model loaded, try to load one on-demand
        if not MODELS:
            try:
                from crop_ssl.models.ssl import create_ssl_model
                model = create_ssl_model('simclr', backbone='vit_small', embed_dim=384)
                model.eval()
                model.to(DEVICE)
                MODELS['simclr_vit_small'] = model
                ACTIVE_MODEL = 'simclr_vit_small'
            except Exception as load_err:
                raise HTTPException(503, f"No model loaded and on-demand loading failed: {load_err}")
        else:
            raise
    model_name_used = model_name or ACTIVE_MODEL

    start = time.time()
    with torch.no_grad():
        if hasattr(model, "encode"):
            features = model.encode(tensor)
            if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
                logits = model.head(features)
            else:
                # SSL model without classifier — project features to class space
                # Use the feature norm + a simple mapping for demo
                feat_dim = features.shape[-1]
                if feat_dim != NUM_CLASSES:
                    # Create a deterministic mapping from features to classes
                    # Use the top dimensions as pseudo-class scores
                    logits = features[:, :NUM_CLASSES]
                else:
                    logits = features
        else:
            logits = model(tensor)
        probs = F.softmax(logits, dim=-1)
    elapsed = (time.time() - start) * 1000

    top5_probs, top5_idx = probs.topk(5, dim=-1)
    top5 = []
    for idx, prob in zip(top5_idx[0], top5_probs[0]):
        i = idx.item()
        if i < NUM_CLASSES:
            top5.append({"class": DISEASE_CLASSES[i], "confidence": round(prob.item() * 100, 2)})

    top_idx = top5_idx[0][0].item()
    if top_idx >= NUM_CLASSES:
        top_idx = 0

    prediction_id = uuid.uuid4().hex[:12]
    if len(PREDICTION_LOG) >= 10_000:
        PREDICTION_LOG.pop(next(iter(PREDICTION_LOG)))  # bounded: drop oldest
    PREDICTION_LOG[prediction_id] = {
        "predicted_class": DISEASE_CLASSES[top_idx],
        "confidence": round(top5_probs[0][0].item() * 100, 2),
        "model_used": model_name_used or "unknown",
        "timestamp": time.time(),
    }

    return PredictionResponse(
        prediction=DISEASE_CLASSES[top_idx],
        confidence=round(top5_probs[0][0].item() * 100, 2),
        top_5=top5,
        inference_time_ms=round(elapsed, 2),
        model_used=model_name_used or "unknown",
        prediction_id=prediction_id,
    )


@app.post("/models/{model_name}/load")
async def load_model(model_name: str, user_payload: Dict = Depends(require_admin)):
    """Load a specific SSL model."""
    global ACTIVE_MODEL
    from crop_ssl.models.ssl import create_ssl_model

    known_methods = ["dinov2", "moco_v3", "simclr", "mae"]
    method, backbone = "simclr", "vit_small"
    for m in known_methods:
        if model_name.startswith(m):
            method = m
            remainder = model_name[len(m):].lstrip("_")
            backbone = remainder if remainder else "vit_small"
            break

    embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
    try:
        model = create_ssl_model(method, backbone=backbone, embed_dim=embed_dims.get(backbone, 384))
        model.eval()
        model.to(DEVICE)
        MODELS[model_name] = model
        ACTIVE_MODEL = model_name
        return {"status": "loaded", "model": model_name, "device": DEVICE}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/checkpoint")
async def upload_checkpoint_model(
    file: UploadFile = File(...),
    method: str = "simclr",
    backbone: str = "vit_small",
    model_name: Optional[str] = None,
    user_payload: Dict = Depends(require_admin),
):
    """Upload a trained checkpoint (from train_ssl / run_pipeline) and serve it.

    Accepts a ``.pth`` saved by :func:`crop_ssl.utils.checkpointing.save_checkpoint`
    (or a raw state dict) plus the matching method/backbone. The checkpoint is
    loaded into an SSL model, made the ACTIVE_MODEL, and registered in the
    model registry — so ``/predict`` and the mobile app use real weights.
    """
    global ACTIVE_MODEL
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > 500 * 1024 * 1024:
        raise HTTPException(400, "Checkpoint exceeds 500 MB")

    embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
    if backbone not in embed_dims:
        raise HTTPException(400, f"Unknown backbone: {backbone}")
    if method not in ("dinov2", "moco_v3", "simclr", "mae"):
        raise HTTPException(400, f"Unknown method: {method}")

    import io as _io
    try:
        ckpt = torch.load(_io.BytesIO(contents), map_location="cpu")
    except Exception as e:
        raise HTTPException(400, f"Not a valid PyTorch checkpoint: {e}")

    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise HTTPException(400, "Checkpoint contains no model state dict")

    from crop_ssl.models.ssl import create_ssl_model
    model = create_ssl_model(method, backbone=backbone, embed_dim=embed_dims[backbone])
    try:
        loaded = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        # Shape mismatch: checkpoint was trained with a different method/backbone
        raise HTTPException(
            400,
            f"Checkpoint does not match {method}/{backbone} "
            f"(shape mismatch). Detail: {str(e)[:200]}",
        )
    n_missing = len(loaded.missing_keys)
    n_unexpected = len(loaded.unexpected_keys)
    n_expected = sum(
        1 for k in state_dict
        if any(k.startswith(p) for p in ("encoder.", "student_backbone.",
                                          "teacher_backbone.", "query_encoder.",
                                          "decoder.", "projector.", "head."))
    )
    # Heuristic guard: if nothing recognizable was transferred, refuse silently
    if n_expected == 0 and loaded.unexpected_keys and n_missing > 0 and not any(
        k in state_dict for k in ("model_state_dict",)
    ):
        raise HTTPException(400, "Checkpoint keys do not match the requested method/backbone")

    name = model_name or f"{method}_{backbone}_trained"
    model.eval()
    model.to(DEVICE)
    MODELS[name] = model
    ACTIVE_MODEL = name

    params = sum(p.numel() for p in model.parameters())
    try:
        version_id = registry.register(name, model, metadata={
            "source": "checkpoint-upload",
            "method": method, "backbone": backbone,
        })
        audit_log.log("model_checkpoint_loaded", "system", {
            "model": name, "version": version_id,
        })
    except Exception:
        version_id = None

    return {
        "status": "loaded",
        "model": name,
        "method": method,
        "backbone": backbone,
        "parameters": params,
        "active": True,
        "missing_keys": n_missing,
        "unexpected_keys": n_unexpected,
        "registry_version": version_id,
        "device": DEVICE,
    }


@app.post("/models/{model_name}/export")
async def export_model(model_name: str, req: ExportRequest = Body(...)):
    """Export a loaded model to ONNX for on-device (Android/PWA) inference.

    Uses the same export utilities as the training pipeline. The exported
    file is written under ``./model_exports/`` and can be fetched via
    ``GET /models/{model_name}/export``.
    """
    # Explicit unknown names must 404 — never silently export the active model
    if model_name not in MODELS:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    model = MODELS[model_name]
    from crop_ssl.utils.export import export_ssl_backbone, verify_onnx
    from pathlib import Path as _P

    export_dir = _P("model_exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    out_path = export_dir / f"{safe_name}.onnx"

    try:
        export_ssl_backbone(
            model, str(out_path),
            backbone_type="teacher",
            input_shape=(1, 3, req.input_size, req.input_size),
        )
    except Exception as e:
        # Fall back to exporting the full model if backbone extraction fails
        from crop_ssl.utils.export import export_to_onnx
        try:
            export_to_onnx(
                model, str(out_path), input_shape=(1, 3, req.input_size, req.input_size),
                opset_version=req.opset,
            )
        except Exception as e2:
            raise HTTPException(500, f"Export failed: {e} | fallback: {e2}")

    size_mb = round(out_path.stat().st_size / (1024 * 1024), 2)

    # Verify with onnxruntime when available (best-effort, never fails the request)
    verified = False
    try:
        verified = verify_onnx(str(out_path), model, input_shape=(1, 3, req.input_size, req.input_size))
    except Exception:
        verified = False

    audit_log.log("model_exported", "system", {
        "model": model_name, "path": str(out_path), "size_mb": size_mb,
    })
    return {
        "status": "exported",
        "model": model_name,
        "path": str(out_path),
        "size_mb": size_mb,
        "input_shape": [1, 3, req.input_size, req.input_size],
        "opset": req.opset,
        "verified": verified,
        "download_url": f"/models/{model_name}/export",
    }


@app.get("/models/{model_name}/export")
async def download_export(model_name: str):
    """Download the ONNX export of a model (for on-device deployment)."""
    from pathlib import Path as _P
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    path = _P("model_exports") / f"{safe_name}.onnx"
    if not path.exists():
        raise HTTPException(404, f"No export for '{model_name}'. POST /models/{model_name}/export first.")
    return FileResponse(str(path), media_type="application/octet-stream",
                       filename=f"{safe_name}.onnx")


@app.post("/models/{model_name}/knn-bundle", response_model=KNNBundleResponse)
async def build_knn_bundle(model_name: str, req: KNNBundleRequest = Body(...)):
    """Build an offline k-NN bundle for the mobile PWA.

    Pairs the model's backbone-flavor ONNX export with a few-shot support set:
    the JSON bundle carries class centroids, raw support embeddings, class
    names, and the exact preprocessing contract (image size + ImageNet
    normalize stats) so the phone reproduces scripts/onnx_knn.py's math
    offline. Embeddings always come from the same backbone flavor that is
    exported — via onnxruntime when installed, otherwise the torch backbone
    (forward_features), which is what the graph computes.
    """
    if model_name not in MODELS:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    model = MODELS[model_name]
    from pathlib import Path as _P
    from crop_ssl.utils.export import export_ssl_backbone

    export_dir = _P("model_exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    onnx_path = export_dir / f"{safe_name}.onnx"

    # 1) Ensure the backbone-flavor ONNX exists (input 'input' → output
    #    'features'). This is the only flavor whose outputs the PWA can rely
    #    on, so unlike /export there is no full-model fallback here.
    if not onnx_path.exists():
        try:
            export_ssl_backbone(model, str(onnx_path),
                                input_shape=(1, 3, 224, 224))
        except Exception as e:
            raise HTTPException(400, f"Backbone export failed: {str(e)[:200]}")

    # 2) Few-shot support set (real train/<class>/ images or the same
    #    deterministic synthetic fallback the CLI and /eval/knn use).
    from crop_ssl.scripts.onnx_knn import load_fewshot_split, normalize as knn_normalize
    from crop_ssl.utils.reproducibility import set_seed
    set_seed(req.seed)
    support, _ = load_fewshot_split(
        _P(req.data_root), req.num_classes, req.shots,
        image_size=224, seed=req.seed,
    )
    if not support:
        raise HTTPException(400, "Few-shot split produced no support data")

    train_dir = _P(req.data_root) / "train"
    if train_dir.exists():
        class_dirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    else:
        class_dirs = []
    if len(class_dirs) >= req.num_classes:
        classes, source = class_dirs[:req.num_classes], "train_dir"
    else:
        classes = [f"Class {i}" for i in range(req.num_classes)]
        source = "synthetic"

    # 3) Embeddings through the same graph flavor that was exported.
    import numpy as np
    sup_feats, sup_labels = [], []
    checked_with = "pytorch-fallback"
    try:
        import onnxruntime as ort  # deliberately optional (see requirements)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        for x, y in support:
            out = sess.run(None, {input_name: knn_normalize(x[None].astype(np.float32))})[0]
            sup_feats.append(np.asarray(out).reshape(-1))
            sup_labels.append(int(y))
        checked_with = "onnxruntime"
    except ImportError:
        # Torch fallback mirrors export_ssl_backbone's BackboneWrapper:
        # teacher → student → encoder → query_encoder, then forward_features.
        backbone = (getattr(model, "teacher_backbone", None)
                    or getattr(model, "student_backbone", None)
                    or getattr(model, "encoder", None)
                    or getattr(model, "query_encoder", None)
                    or model)
        for x, y in support:
            t = torch.from_numpy(knn_normalize(x[None].astype(np.float32))).float()
            with torch.no_grad():
                feats = backbone.forward_features(t)
            sup_feats.append(feats.reshape(-1).cpu().numpy())
            sup_labels.append(int(y))

    sup_feats = np.stack(sup_feats).astype(np.float32)
    sup_labels = np.array(sup_labels)
    embed_dim = int(sup_feats.shape[1])

    # 4) One raw-mean centroid per class (L2 normalization happens at
    #    classify time — identical to nearest_centroid() in onnx_knn.py).
    centroids = np.stack([
        sup_feats[sup_labels == c].mean(axis=0) for c in range(req.num_classes)
    ]).astype(np.float32)

    # 5) Write the bundle.
    import json as _json
    bundle = {
        "version": 1,
        "model_name": model_name,
        "image_size": 224,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "mode": f"k-NN (k={req.k})" if req.k > 0 else "nearest-centroid",
        "k": req.k,
        "embed_dim": embed_dim,
        "classes": classes,
        "num_classes": req.num_classes,
        "shots": req.shots,
        "support": {
            "embeddings": [[round(float(v), 6) for v in row] for row in sup_feats],
            "labels": sup_labels.tolist(),
        },
        "centroids": [[round(float(v), 6) for v in row] for row in centroids],
        "source": source,
        "checked_with": checked_with,
        "onnx_file": f"{safe_name}.onnx",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    bundle_path = export_dir / f"{safe_name}-knn-bundle.json"
    bundle_path.write_text(_json.dumps(bundle))
    size_mb = round(bundle_path.stat().st_size / (1024 * 1024), 2)

    audit_log.log("knn_bundle_built", "system", {
        "model": model_name, "path": str(bundle_path), "size_mb": size_mb,
    })
    return KNNBundleResponse(
        status="built",
        model=model_name,
        path=str(bundle_path),
        size_mb=size_mb,
        mode=bundle["mode"],
        k=req.k,
        num_classes=req.num_classes,
        shots=req.shots,
        num_support=len(support),
        embed_dim=embed_dim,
        source=source,
        checked_with=checked_with,
        download_url=f"/models/{model_name}/knn-bundle",
    )


@app.get("/models/{model_name}/knn-bundle")
async def download_knn_bundle(model_name: str):
    """Download the offline k-NN bundle for a model (PWA fetches this)."""
    from pathlib import Path as _P
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    path = _P("model_exports") / f"{safe_name}-knn-bundle.json"
    if not path.exists():
        raise HTTPException(404, f"No k-NN bundle for '{model_name}'. POST /models/{model_name}/knn-bundle first.")
    return FileResponse(str(path), media_type="application/json",
                       filename=f"{safe_name}-knn-bundle.json")


@app.delete("/models/{model_name}")
async def unload_model(model_name: str, user_payload: Dict = Depends(require_admin)):
    """Unload a model from memory."""
    global ACTIVE_MODEL
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


@app.get("/attention/{model_name}")
async def get_attention_maps(model_name: str):
    """Get attention map shapes from a model's transformer blocks."""
    # Explicit unknown names must 404 — never silently serve another model
    if model_name not in MODELS:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    model = MODELS[model_name]
    if not hasattr(model, "student_backbone") and not hasattr(model, "encoder"):
        raise HTTPException(400, "Model has no transformer backbone")

    backbone = getattr(model, "student_backbone", getattr(model, "encoder", None))
    if not hasattr(backbone, "blocks"):
        raise HTTPException(400, "No transformer blocks found")

    layer_count = len(backbone.blocks)
    # Read the real per-head config from the first attention block
    # (vit_small=6, vit_base=12, vit_large=16 — never derivable from embed_dim)
    num_heads = backbone.blocks[0].attn.num_heads
    embed_dim = backbone.embed_dim if hasattr(backbone, "embed_dim") else 768

    shapes = [[num_heads, 197, 197] for _ in range(layer_count)]
    return AttentionResponse(layer_count=layer_count, attention_shapes=shapes)


@app.get("/training/status")
async def training_status():
    return {"jobs": list(TRAINING_JOBS.values())}


@app.post("/training/start")
async def start_training(req: TrainingRequest, user_payload: Dict = Depends(require_admin)):
    """Start a training job in background."""
    if req.method not in ["simclr", "dinov2", "moco_v3", "mae"]:
        raise HTTPException(400, f"Unknown method: {req.method}")
    if req.backbone not in ["vit_small", "vit_base", "vit_large"]:
        raise HTTPException(400, f"Unknown backbone: {req.backbone}")
    epochs = max(1, min(req.epochs, 100))
    lr = max(1e-6, min(req.lr, 1.0))

    job_id = str(uuid.uuid4())[:8]
    TRAINING_JOBS[job_id] = {
        "job_id": job_id, "status": "running",
        "method": req.method, "backbone": req.backbone,
        "epoch": 0, "total_epochs": epochs,
        "loss": 0.0, "accuracy": 0.0,
    }

    import threading

    def train_bg():
        try:
            from crop_ssl.models.ssl import create_ssl_model
            from torch.utils.data import TensorDataset, DataLoader

            embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
            model = create_ssl_model(req.method, backbone=req.backbone, embed_dim=embed_dims.get(req.backbone, 384))
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
                    if req.method in ("simclr", "moco_v3"):
                        result = model(images, torch.randn_like(images))
                    elif req.method == "mae":
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
            model.eval()
            MODELS[f"{req.method}_{req.backbone}_trained"] = model

        except Exception as e:
            TRAINING_JOBS[job_id]["status"] = "failed"
            TRAINING_JOBS[job_id]["error"] = str(e)

    thread = threading.Thread(target=train_bg, daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "started"}


@app.get("/pipeline/compare")
async def compare_architectures(
    method_a: str = "simclr",
    backbone_a: str = "vit_small",
    method_b: str = "dinov2",
    backbone_b: str = "vit_small",
):
    """Compare two SSL architectures side-by-side."""
    from crop_ssl.models.ssl import create_ssl_model
    from crop_ssl.utils.export import count_parameters

    embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
    try:
        model_a = create_ssl_model(method_a, backbone=backbone_a, embed_dim=embed_dims.get(backbone_a, 384))
        model_b = create_ssl_model(method_b, backbone=backbone_b, embed_dim=embed_dims.get(backbone_b, 384))
    except Exception as e:
        raise HTTPException(400, str(e))

    params_a = count_parameters(model_a)
    params_b = count_parameters(model_b)

    return {
        "model_a": {"method": method_a, "backbone": backbone_a, **params_a},
        "model_b": {"method": method_b, "backbone": backbone_b, **params_b},
    }


@app.get("/datasets")
async def list_datasets():
    """List all supported datasets."""
    from crop_ssl.data.datasets import DATASET_REGISTRY
    return {
        "datasets": list(DATASET_REGISTRY.keys()),
        "count": len(DATASET_REGISTRY),
    }


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = None,
):
    """Batch predict multiple images at once (up to 10)."""
    if len(files) > 10:
        raise HTTPException(400, "Max 10 images per batch")

    model = _get_model(model_name)
    results = []
    total_time = 0.0

    for f in files:
        contents = await f.read()
        try:
            tensor = _preprocess_image(contents)
            tensor = tensor.to(DEVICE)
            start = time.time()
            with torch.no_grad():
                if hasattr(model, "encode"):
                    features = model.encode(tensor)
                    logits = features[:, :NUM_CLASSES]
                else:
                    logits = model(tensor)
                probs = F.softmax(logits, dim=-1)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            top5p, top5i = probs.topk(5, dim=-1)
            top_idx = top5i[0][0].item()
            if top_idx >= NUM_CLASSES:
                top_idx = 0
            results.append({
                "filename": f.filename,
                "prediction": DISEASE_CLASSES[top_idx],
                "confidence": round(top5p[0][0].item() * 100, 2),
                "inference_time_ms": round(elapsed, 2),
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return {
        "results": results,
        "total_images": len(results),
        "total_time_ms": round(total_time, 2),
        "avg_time_ms": round(total_time / max(len(results), 1), 2),
        "model_used": model_name or ACTIVE_MODEL or "unknown",
    }


@app.post("/feedback")
async def prediction_feedback(req: FeedbackRequest):
    """Record ground truth for a stored prediction (closes the feedback loop).

    Feeds the existing auto-retrain monitor and drift detector, so accuracy
    drops and class-distribution shifts are tracked automatically instead of
    requiring manual calls to /auto-retrain/record and /drift/record.
    """
    if req.predicted_class is not None:
        pred_class = req.predicted_class
        model_used = req.model_used or "unknown"
    else:
        entry = PREDICTION_LOG.get(req.prediction_id)
        if entry is None:
            raise HTTPException(404, f"Unknown or expired prediction_id: {req.prediction_id}")
        pred_class = entry["predicted_class"]
        model_used = entry["model_used"]

    auto_retrain.record_prediction(model_used, correct=req.correct, confidence=req.confidence)
    drift_detector.record_prediction(pred_class, req.confidence)
    audit_log.log("prediction_feedback", "system", {
        "prediction_id": req.prediction_id,
        "correct": req.correct,
        "predicted_class": pred_class,
        "model": model_used,
    })
    return {
        "status": "recorded",
        "prediction_id": req.prediction_id,
        "correct": req.correct,
        "predicted_class": pred_class,
        "model": model_used,
    }


@app.get("/system/latency")
async def system_latency():
    """Per-route latency percentiles (ms) for capacity/ops monitoring.

    Aggregated over an in-memory ring buffer (last ≤500 requests per route);
    resets on restart and does not aggregate across replicas.
    """
    with _latency_lock:
        snapshot = {p: list(ts) for p, ts in _LATENCY.items()}
    routes = []
    for path, ts in snapshot.items():
        if not ts:
            continue
        s = sorted(ts)
        n = len(s)
        routes.append({
            "route": path,
            "samples": n,
            "p50_ms": round(s[n // 2], 1),
            "p95_ms": round(s[min(n - 1, int(n * 0.95))], 1),
            "mean_ms": round(sum(s) / n, 1),
        })
    routes.sort(key=lambda r: r["p95_ms"], reverse=True)
    return {
        "window": "ring buffer: last <=500 requests per route, in-memory, resets on restart",
        "routes": routes,
    }


@app.get("/system/metrics")
async def system_metrics():
    """System metrics for monitoring."""
    import sys
    import os
    metrics = {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "uptime_human": _format_uptime(time.time() - START_TIME),
        "device": DEVICE,
        "models_loaded": len(MODELS),
        "active_model": ACTIVE_MODEL,
        "training_jobs": len(TRAINING_JOBS),
        "python_version": sys.version,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
        metrics["gpu_memory_used_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 1)
        metrics["gpu_memory_total_mb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024**2, 1)
    return metrics


def _format_uptime(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ============================================================
# Automation Endpoints
# ============================================================
from crop_ssl.backend.automation import (
    registry, auto_retrain, webhooks, ab_tests,
    drift_detector, audit_log, orchestrator,
)


# --- Model Registry ---
@app.post("/registry/register")
async def registry_register(
    model_name: str = "default",
    user: str = "system",
    user_payload: Dict = Depends(require_admin),
):
    """Register the active model in the registry."""
    global ACTIVE_MODEL
    try:
        model = _get_model()
    except HTTPException:
        # No model loaded — try to create one on-demand
        if not MODELS:
            try:
                from crop_ssl.models.ssl import create_ssl_model
                model = create_ssl_model('simclr', backbone='vit_small', embed_dim=384)
                model.eval()
                model.to(DEVICE)
                MODELS['simclr_vit_small'] = model
                ACTIVE_MODEL = 'simclr_vit_small'
            except Exception as e:
                raise HTTPException(503, f"No model loaded and on-demand loading failed: {e}")
        else:
            raise
    version_id = registry.register(model_name, model, metadata={"user": user})
    audit_log.log("model_registered", user, {"model": model_name, "version": version_id})
    return {"version_id": version_id, "model_name": model_name}


@app.post("/registry/deploy")
async def registry_deploy(model_name: str, version_id: str, user: str = "system", user_payload: Dict = Depends(require_admin)):
    """Deploy a specific model version."""
    success = registry.deploy(model_name, version_id)
    if not success:
        raise HTTPException(404, "Version not found")
    audit_log.log("model_deployed", user, {"model": model_name, "version": version_id})
    webhooks.dispatch("model_deployed", {"model": model_name, "version": version_id})
    return {"status": "deployed", "model": model_name, "version": version_id}


@app.post("/registry/rollback")
async def registry_rollback(model_name: str, user: str = "system", user_payload: Dict = Depends(require_admin)):
    """Rollback to previous model version."""
    prev = registry.rollback(model_name)
    if not prev:
        raise HTTPException(400, "No previous version to rollback to")
    audit_log.log("model_rollback", user, {"model": model_name, "version": prev})
    return {"status": "rolled_back", "version": prev}


@app.get("/registry/versions")
async def registry_versions(model_name: Optional[str] = None):
    """List model versions."""
    if model_name:
        return {"model": model_name, "versions": registry.list_versions(model_name)}
    return {"models": registry.list_models()}


@app.get("/registry/deployed")
async def registry_deployed(model_name: Optional[str] = None):
    """Get currently deployed version."""
    if model_name:
        return registry.get_deployed(model_name) or {"error": "not found"}
    return {"deployed": {n: registry.get_deployed(n) for n in registry.list_models()}}


# --- Auto-Retrain Monitor ---
@app.post("/auto-retrain/record")
async def auto_retrain_record(
    model_name: str = "default",
    correct: Optional[bool] = None,
    confidence: float = 0.0,
):
    """Record a prediction for auto-retrain monitoring."""
    auto_retrain.record_prediction(model_name, correct, confidence)
    return {"status": "recorded", "model": model_name}


@app.get("/auto-retrain/stats")
async def auto_retrain_stats(model_name: str = "default"):
    """Get auto-retrain monitoring stats."""
    return auto_retrain.get_stats(model_name)


@app.get("/auto-retrain/alerts")
async def auto_retrain_alerts(model_name: Optional[str] = None):
    """Get retrain alerts."""
    return {"alerts": auto_retrain.get_alerts(model_name)}


# --- Webhooks ---
@app.post("/webhooks/register")
async def webhook_register(event: str, url: str, secret: Optional[str] = None, user_payload: Dict = Depends(require_admin)):
    """Register a webhook."""
    hook_id = webhooks.register(event, url, secret)
    audit_log.log("webhook_registered", "system", {"event": event, "url": url})
    return {"hook_id": hook_id, "event": event}


@app.post("/webhooks/unregister")
async def webhook_unregister(event: str, hook_id: str, user_payload: Dict = Depends(require_admin)):
    """Remove a webhook."""
    removed = webhooks.unregister(event, hook_id)
    if not removed:
        raise HTTPException(404, "Hook not found")
    return {"status": "removed"}


@app.get("/webhooks/list")
async def webhook_list(event: Optional[str] = None):
    """List webhooks."""
    return webhooks.list_hooks(event)


@app.get("/webhooks/deliveries")
async def webhook_deliveries(limit: int = 20):
    """Get recent webhook deliveries."""
    return {"deliveries": webhooks.get_delivery_log(limit)}


@app.post("/webhooks/test")
async def webhook_test(event: str = "test", user_payload: Dict = Depends(require_admin)):
    """Send a test webhook."""
    results = webhooks.dispatch(event, {"test": True, "timestamp": datetime.now().isoformat()})
    return {"dispatched": len(results), "event": event}


# --- A/B Testing ---
@app.post("/ab/create")
async def ab_create(req: ABCreateRequest, user_payload: Dict = Depends(require_admin)):
    """Create an A/B test."""
    test_id = ab_tests.create_test(
        req.test_name, req.model_a, req.model_b, req.traffic_split
    )
    audit_log.log("ab_test_created", "system", {"test": req.test_name, "id": test_id})
    return {"test_id": test_id}


@app.get("/ab/route/{test_id}")
async def ab_route(test_id: str):
    """Route to model A or B."""
    variant = ab_tests.route(test_id)
    if variant is None:
        raise HTTPException(404, "Test not found or stopped")
    return {"variant": variant, "model": "a" if variant == "a" else "b"}


@app.post("/ab/record")
async def ab_record(test_id: str, variant: str, correct: bool, confidence: float, user_payload: Dict = Depends(require_admin)):
    """Record an A/B test result."""
    ab_tests.record_result(test_id, variant, correct, confidence)
    return {"status": "recorded"}


@app.get("/ab/results/{test_id}")
async def ab_results(test_id: str):
    """Get A/B test results."""
    results = ab_tests.get_results(test_id)
    if not results:
        raise HTTPException(404, "Test not found")
    return results


@app.post("/ab/stop/{test_id}")
async def ab_stop(test_id: str, user_payload: Dict = Depends(require_admin)):
    """Stop an A/B test."""
    ab_tests.stop_test(test_id)
    return {"status": "stopped"}


@app.get("/ab/tests")
async def ab_list():
    """List all A/B tests."""
    return {"tests": ab_tests.list_tests()}


# --- Drift Detection ---
@app.post("/drift/set-reference")
async def drift_set_reference(distribution: Dict[str, float] = Body(...)):
    """Set the reference class distribution (JSON body)."""
    drift_detector.set_reference(distribution)
    return {"status": "set", "classes": len(distribution)}


@app.post("/drift/record")
async def drift_record(class_name: str, confidence: float = 0.0):
    """Record a prediction for drift monitoring."""
    drift_detector.record_prediction(class_name, confidence)
    return {"status": "recorded"}


@app.get("/drift/check")
async def drift_check():
    """Check for prediction drift."""
    return drift_detector.check_drift()


@app.get("/drift/alerts")
async def drift_alerts():
    """Get drift alerts."""
    return {"alerts": drift_detector.get_alerts()}


# --- Audit Log ---
@app.get("/audit/logs")
async def audit_logs(
    action: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 50,
):
    """Query audit logs."""
    return {"entries": audit_log.query(action, user, limit)}


@app.get("/audit/stats")
async def audit_stats():
    """Get audit statistics."""
    return audit_log.get_stats()


# --- Pipeline Orchestrator ---
@app.get("/pipeline/list")
async def pipeline_list():
    """List all pipelines."""
    return {"pipelines": orchestrator.list_pipelines()}


@app.post("/pipeline/create")
async def pipeline_create(req: PipelineCreateRequest, user_payload: Dict = Depends(require_admin)):
    """Create a new ML pipeline."""
    pipe_id = orchestrator.create_pipeline(
        req.name, req.ssl_method, req.backbone, req.dataset,
        req.adaptation, req.target_dataset, req.num_shots,
    )
    audit_log.log("pipeline_created", "system", {"pipeline": req.name, "id": pipe_id})
    return {"pipe_id": pipe_id, "name": req.name}


@app.get("/pipeline/{pipe_id}")
async def pipeline_get(pipe_id: str):
    """Get pipeline status."""
    pipe = orchestrator.get_pipeline(pipe_id)
    if not pipe:
        raise HTTPException(404, "Pipeline not found")
    return pipe


@app.post("/pipeline/{pipe_id}/step/{step_idx}")
async def pipeline_step(
    pipe_id: str,
    step_idx: int,
    status: str,
    result: Optional[Dict] = None,
    user_payload: Dict = Depends(require_admin),
):
    """Update a pipeline step."""
    pipe = orchestrator.get_pipeline(pipe_id)
    if not pipe:
        raise HTTPException(404, "Pipeline not found")
    if step_idx < 0 or step_idx >= len(pipe["steps"]):
        raise HTTPException(404, f"Invalid step index {step_idx} (pipeline has {len(pipe['steps'])} steps)")
    ok = orchestrator.update_step(pipe_id, step_idx, status, result)
    if not ok:
        raise HTTPException(404, "Pipeline not found")
    return {"status": "updated", "step": step_idx}


# --- Enhanced System Metrics with Automation ---
@app.get("/system/automation-status")
async def automation_status():
    """Get overall automation status."""
    return {
        "registry": {
            "models": registry.list_models(),
            "deployed": {n: (registry.get_deployed(n) or {}).get("version_id") for n in registry.list_models()},
        },
        "auto_retrain": {
            "alerts": len(auto_retrain.get_alerts()),
        },
        "webhooks": {
            "total_hooks": sum(len(v) for v in webhooks.list_hooks().values()),
        },
        "ab_tests": {
            "active": sum(1 for t in ab_tests.list_tests() if t["status"] == "running"),
            "total": len(ab_tests.list_tests()),
        },
        "drift": drift_detector.check_drift(),
        "audit_entries": audit_log.get_stats()["total_entries"],
        "pipelines": len(orchestrator.list_pipelines()),
    }


# ============================================================
# Few-Shot k-NN / Nearest-Centroid Evaluation
# ============================================================
_KNN_EMBED_MODELS: Dict[str, torch.nn.Module] = {}  # bounded transient-model cache


@app.post("/eval/knn", response_model=KNNRunResponse)
async def eval_knn(req: KNNRunRequest):
    """Run a few-shot k-NN / nearest-centroid evaluation on SSL embeddings.

    Reuses scripts/onnx_knn.py's data split and classifier so the API, the
    CLI, and the dashboard report the same numbers. Embeddings come from an
    already-loaded registry model when one matches method/backbone,
    otherwise a transient model is built (and cached, bounded to 2).

    Note: runs synchronously like /predict — a vit_large eval on CPU can
    hold the event loop for tens of seconds.
    """
    if req.method not in ("dinov2", "moco_v3", "simclr", "mae"):
        raise HTTPException(400, f"Unknown method: {req.method}")
    if req.backbone not in ("vit_small", "vit_base", "vit_large"):
        raise HTTPException(400, f"Unknown backbone: {req.backbone}")
    from crop_ssl.scripts.onnx_knn import load_fewshot_split, nearest_centroid
    from crop_ssl.utils.reproducibility import set_seed
    from pathlib import Path

    t0 = time.perf_counter()
    set_seed(req.seed)

    # --- embedding function: registry model if loaded, else transient ---
    registry_name = f"{req.method}_{req.backbone}"
    if registry_name in MODELS:
        model, embedding_source = MODELS[registry_name], f"registry:{registry_name}"
    else:
        if registry_name in _KNN_EMBED_MODELS:
            model = _KNN_EMBED_MODELS[registry_name]
        else:
            embed_dims = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}
            from crop_ssl.models.ssl import create_ssl_model
            model = create_ssl_model(
                req.method, backbone=req.backbone,
                embed_dim=embed_dims[req.backbone],
            )
            model.eval().to(DEVICE)
            # Bound: 2 entries max — each transient ViT-L is ~1.2 GB on CPU,
            # so a larger cache could pin gigabytes for a one-off eval.
            if len(_KNN_EMBED_MODELS) >= 2:
                _KNN_EMBED_MODELS.pop(next(iter(_KNN_EMBED_MODELS)))
            _KNN_EMBED_MODELS[registry_name] = model
        embedding_source = "transient"

    def embed(images) -> "torch.Tensor":
        x = torch.from_numpy(images).float().to(DEVICE)
        with torch.no_grad():
            return model.encode(x).cpu()

    # --- data: reuse the script's split (real train/<class>/ or synthetic) ---
    support, query = load_fewshot_split(
        Path(req.data_root), req.num_classes, req.shots,
        image_size=224, seed=req.seed,
    )
    if not support or not query:
        raise HTTPException(400, "Few-shot split produced no data")

    import numpy as np
    from crop_ssl.scripts.onnx_knn import normalize

    def to_feats(items):
        xs = np.stack([x for x, _ in items]).astype(np.float32)
        ys = np.array([y for _, y in items])
        feats = torch.cat([embed(normalize(xs[i:i + 32]))
                           for i in range(0, len(xs), 32)], dim=0).numpy()
        return feats, ys

    sup_feats, sup_labels = to_feats(support)
    qry_feats, qry_labels = to_feats(query)

    preds = nearest_centroid(sup_feats, sup_labels, qry_feats, k=req.k)
    acc = float((preds == qry_labels).mean())
    per_class = [
        KNNClassReport(
            class_index=int(c),
            accuracy=float((preds[qry_labels == c] == c).mean()),
            n=int((qry_labels == c).sum()),
        )
        for c in np.unique(qry_labels)
    ]
    runtime_ms = (time.perf_counter() - t0) * 1000
    return KNNRunResponse(
        mode=f"k-NN (k={req.k})" if req.k > 0 else "nearest-centroid",
        accuracy=round(acc, 4),
        num_support=len(support),
        num_query=len(query),
        per_class=per_class,
        embedding_source=embedding_source,
        runtime_ms=round(runtime_ms, 1),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler for production robustness.

    Logs the full traceback server-side; the response body carries only a
    short message (never the stack) so internals don't leak to clients.
    """
    print(f"⚠️  Unhandled error on {request.method} {request.url.path}: "
          f"{type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        # Generic message: str(exc) can itself leak paths/internals. The
        # full detail is in the server log line above.
        content={"error": "Internal server error", "type": type(exc).__name__},
    )


# ============================================================
# Mobile PWA (Android-ready) — served from /app
# ============================================================
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_MOBILE_DIR = Path(__file__).resolve().parent.parent / "frontend" / "mobile"
if _MOBILE_DIR.exists():
    app.mount(
        "/app",
        StaticFiles(directory=str(_MOBILE_DIR), html=True),
        name="mobile_app",
    )


@app.get("/health/mobile", include_in_schema=False)
async def mobile_health():
    """Mobile PWA availability probe."""
    return {"app": "mobile", "available": _MOBILE_DIR.exists()}


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
