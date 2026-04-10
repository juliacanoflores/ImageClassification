from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import io
import os
import base64

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


# Load model at startup
MODEL = load_model()


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "ConvNeXT-Base" if MODEL else "Not loaded"
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
        
        image_bytes = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        return {
            "status": "success",
            "label": CLASSES[predicted_idx.item()],
            "confidence": round(confidence.item(), 4),
            "filename": filename
        }
    
    except Exception as e:
        return {
            "status": "error",
            "label": "Error",
            "confidence": 0.0,
            "message": str(e)
        }


if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)