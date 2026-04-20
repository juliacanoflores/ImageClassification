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
MODEL_PATH = "models/convnext_base-8epoch.pt"
IMAGE_SIZE = 224

CLASSES = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside city",
    "Kitchen", "Living room", "Mountain", "Office", "Open country",
    "Store", "Street", "Suburb", "Tall building"
]

# Image preprocessing pipeline (must match training)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


class SceneClassifier(nn.Module):
    """Transfer learning classifier using ConvNeXT backbone."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        base_model = torchvision.models.convnext_base(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)


def load_model() -> SceneClassifier | None:
    """Load model from disk."""
    try:
        print("Loading model...")
        
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            print(f"Working directory: {os.getcwd()}")
            return None
        
        print(f"Model file found: {MODEL_PATH}")
        model = SceneClassifier(len(CLASSES))
        print("Model architecture created")
        
        weights = torch.load(MODEL_PATH, map_location=DEVICE)
        print(f"Weights loaded from disk ({len(weights)} keys)")
        
        model.load_state_dict(weights, strict=False)
        print("Weights assigned to model")
        
        model.to(DEVICE)
        model.eval()
        
        print(f"Model ready for inference on {DEVICE}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _decode_base64_image(image_str: str) -> torch.Tensor:
    """Decode base64 image and convert it to a model-ready tensor."""
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)


def _predict_topk(image_tensor: torch.Tensor, k: int = 1) -> list[dict]:
    """Run inference and return top-k predictions with confidences."""
    k = max(1, min(k, len(CLASSES)))
    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, indices = torch.topk(probabilities, k=k, dim=1)

    predictions = []
    for confidence, idx in zip(confidences[0].tolist(), indices[0].tolist()):
        predictions.append({
            "label": CLASSES[idx],
            "confidence": round(float(confidence), 4)
        })
    return predictions


# Load model at startup
MODEL = load_model()


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "ConvNeXT-Base" if MODEL else "Not loaded"
    }


@app.get("/health")
async def health():
    """Detailed health endpoint for service and model readiness."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }


@app.get("/classes")
async def classes():
    """Return the list of class labels used by the model."""
    return {
        "status": "ok",
        "num_classes": len(CLASSES),
        "classes": CLASSES
    }


@app.get("/model-info")
async def model_info():
    """Return metadata about the loaded model and runtime config."""
    return {
        "status": "ok",
        "architecture": "ConvNeXT-Base",
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "num_classes": len(CLASSES),
        "model_path": MODEL_PATH,
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(data: dict):
    """Predict scene class from Base64 encoded image."""
    if MODEL is None:
        return {
            "status": "error",
            "label": "Error",
            "confidence": 0.0,
            "message": "Model not loaded"
        }
    
    try:
        image_str = data.get("image", "")
        filename = data.get("filename", "image.jpg")
        
        if not image_str:
            return {
                "status": "error",
                "label": "Error",
                "confidence": 0.0,
                "message": "No image data provided"
            }

        image_tensor = _decode_base64_image(image_str)
        top_prediction = _predict_topk(image_tensor, k=1)[0]
        
        return {
            "status": "success",
            "label": top_prediction["label"],
            "confidence": top_prediction["confidence"],
            "filename": filename
        }
    
    except Exception as e:
        return {
            "status": "error",
            "label": "Error",
            "confidence": 0.0,
            "message": str(e)
        }


@app.post("/predict-topk")
async def predict_topk(data: dict):
    """Predict top-k classes from Base64 encoded image."""
    if MODEL is None:
        return {
            "status": "error",
            "predictions": [],
            "message": "Model not loaded"
        }

    try:
        image_str = data.get("image", "")
        filename = data.get("filename", "image.jpg")
        k = int(data.get("k", 3))

        if not image_str:
            return {
                "status": "error",
                "predictions": [],
                "message": "No image data provided"
            }

        image_tensor = _decode_base64_image(image_str)
        predictions = _predict_topk(image_tensor, k=k)

        return {
            "status": "success",
            "filename": filename,
            "k": len(predictions),
            "predictions": predictions
        }

    except Exception as e:
        return {
            "status": "error",
            "predictions": [],
            "message": str(e)
        }


if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)