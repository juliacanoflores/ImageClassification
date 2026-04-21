import base64
import io
import os

import torch
import torch.nn as nn
import torchvision
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# App configuration
app = FastAPI(title="Scene Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

CLASSES = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside city",
    "Kitchen", "Living room", "Mountain", "Office", "Open country",
    "Store", "Street", "Suburb", "Tall building"
]

MODEL_CONFIGS = {
    "ConvNeXt-Small": {"path": "models/ConvNeXt-Small.pt", "arch": "convnext_small"},
    "EfficientNetV2-S": {"path": "models/EfficientNetV2.pt", "arch": "efficientnet_v2_s"},
}
DEFAULT_MODEL = "ConvNeXt-Small"

# Image preprocessing pipeline (must match training)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


class SceneClassifier(nn.Module):
    """Transfer learning classifier with pluggable backbone."""

    def __init__(self, num_classes: int, arch: str = "convnext_small"):
        super().__init__()
        builders = {
            "convnext_small": torchvision.models.convnext_small,
            "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
        }
        base_model = builders[arch](weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


def load_model(path: str, arch: str) -> SceneClassifier | None:
    """Load a model from disk."""
    try:
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return None
        model = SceneClassifier(len(CLASSES), arch)
        weights = torch.load(path, map_location=DEVICE)
        model.load_state_dict(weights, strict=False)
        model.to(DEVICE)
        model.eval()
        print(f"Loaded {arch} from {path}")
        return model
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def _decode_base64_image(image_str: str) -> torch.Tensor:
    """Decode base64 image and convert it to a model-ready tensor."""
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)


def _predict_topk(image_tensor: torch.Tensor, model: SceneClassifier, k: int = 1) -> list[dict]:
    """Run inference and return top-k predictions with confidences."""
    k = max(1, min(k, len(CLASSES)))
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, indices = torch.topk(probabilities, k=k, dim=1)

    return [
        {"label": CLASSES[idx], "confidence": round(float(conf), 4)}
        for conf, idx in zip(confidences[0].tolist(), indices[0].tolist())
    ]


# Load all models at startup
MODELS: dict[str, SceneClassifier | None] = {
    name: load_model(cfg["path"], cfg["arch"])
    for name, cfg in MODEL_CONFIGS.items()
}


def _resolve_model(name: str) -> SceneClassifier | None:
    return MODELS.get(name) or MODELS.get(DEFAULT_MODEL)


@app.get("/")
async def health_check():
    loaded = [n for n, m in MODELS.items() if m is not None]
    return {"status": "ok", "models_loaded": loaded}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {n: m is not None for n, m in MODELS.items()},
        "device": str(DEVICE)
    }


@app.get("/classes")
async def classes():
    return {"status": "ok", "num_classes": len(CLASSES), "classes": CLASSES}


@app.get("/model-info")
async def model_info():
    return {
        "status": "ok",
        "available_models": list(MODEL_CONFIGS.keys()),
        "default_model": DEFAULT_MODEL,
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "num_classes": len(CLASSES),
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(data: dict):
    model_name = data.get("model", DEFAULT_MODEL)
    model = _resolve_model(model_name)
    if model is None:
        return {"status": "error", "label": "Error", "confidence": 0.0, "message": "Model not loaded"}

    try:
        image_str = data.get("image", "")
        filename = data.get("filename", "image.jpg")
        if not image_str:
            return {"status": "error", "label": "Error", "confidence": 0.0, "message": "No image data provided"}

        image_tensor = _decode_base64_image(image_str)
        top = _predict_topk(image_tensor, model, k=1)[0]
        return {"status": "success", "label": top["label"], "confidence": top["confidence"], "filename": filename, "model": model_name}

    except Exception as e:
        return {"status": "error", "label": "Error", "confidence": 0.0, "message": str(e)}


@app.post("/predict-topk")
async def predict_topk(data: dict):
    model_name = data.get("model", DEFAULT_MODEL)
    model = _resolve_model(model_name)
    if model is None:
        return {"status": "error", "predictions": [], "message": "Model not loaded"}

    try:
        image_str = data.get("image", "")
        filename = data.get("filename", "image.jpg")
        k = int(data.get("k", 3))
        if not image_str:
            return {"status": "error", "predictions": [], "message": "No image data provided"}

        image_tensor = _decode_base64_image(image_str)
        predictions = _predict_topk(image_tensor, model, k=k)
        return {"status": "success", "filename": filename, "k": len(predictions), "predictions": predictions, "model": model_name}

    except Exception as e:
        return {"status": "error", "predictions": [], "message": str(e)}


if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)