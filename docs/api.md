# API Documentation

This document contains the backend API reference for the Scene Image Classifier.

## Base URL

- Local backend: http://localhost:8000

## Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### GET /

Health check endpoint.

Response:
```json
{
  "status": "ok",
  "model": "ConvNeXT-Base"
}
```

### GET /health

Detailed service and runtime status.

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

### GET /classes

Returns class labels used by the backend.

Response:
```json
{
  "status": "ok",
  "num_classes": 15,
  "classes": ["Bedroom", "Coast", "Forest", "..."]
}
```

### GET /model-info

Returns metadata used by the Streamlit sidebar.

Response:
```json
{
  "status": "ok",
  "architecture": "ConvNeXT-Base",
  "input_size": [224, 224],
  "num_classes": 15,
  "model_path": "models/convnext_base-8epoch.pt",
  "model_loaded": true,
  "device": "cpu"
}
```

### POST /predict

Predict one class for a base64 image.

Request:
```json
{
  "image": "base64_encoded_image_string",
  "filename": "image.jpg"
}
```

Success response:
```json
{
  "status": "success",
  "label": "Forest",
  "confidence": 0.9453,
  "filename": "image.jpg"
}
```

Error response:
```json
{
  "status": "error",
  "label": "Error",
  "confidence": 0.0,
  "message": "Error description"
}
```

### POST /predict-topk

Predict top-k classes for a base64 image.

Request:
```json
{
  "image": "base64_encoded_image_string",
  "filename": "image.jpg",
  "k": 3
}
```

Success response:
```json
{
  "status": "success",
  "filename": "image.jpg",
  "k": 3,
  "predictions": [
    {"label": "Forest", "confidence": 0.9453},
    {"label": "Mountain", "confidence": 0.0331},
    {"label": "Open country", "confidence": 0.0102}
  ]
}
```

Error response:
```json
{
  "status": "error",
  "predictions": [],
  "message": "Error description"
}
```

## Quick Test with curl

You can quickly verify the metadata endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/classes
curl http://localhost:8000/model-info
```

For prediction endpoints, send a valid base64 image in JSON.
