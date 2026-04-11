# Scene Image Classifier

A web application that classifies images into 15 different scene categories using a ConvNeXT deep learning model trained with transfer learning.

## Architecture

The application uses a **microservices architecture** with two independent services:

```
┌─────────────────────────────────────────┐
│   Streamlit Frontend (Port 8501)        │
│   - User interface                      │
│   - Image upload/selection              │
│   - Result display                      │
└────────────┬────────────────────────────┘
             │ HTTP JSON
             ↓
┌─────────────────────────────────────────┐
│   FastAPI Backend (Port 8000)           │
│   - ConvNeXT model (8 epochs)           │
│   - Image processing                    │
│   - Classification inference            │
└─────────────────────────────────────────┘
```

## Supported Scene Classes

Bedroom, Coast, Forest, Highway, Industrial, Inside city, Kitchen, Living room, Mountain, Office, Open country, Store, Street, Suburb, Tall building

## Quick Start

### Installation

1. **Clone/Navigate to the project directory:**
   ```bash
   cd ImageClassification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Step 1: Start the backend server** (in a terminal)
```bash
python fastapi_backend.py
```

Expected output:
```
Loading model...
[OK] Model loaded successfully!
Starting server on http://0.0.0.0:8000...
```

**Step 2: Start the frontend** (in a new terminal)
```bash
streamlit run app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Step 3: Open in browser**
- Go to `http://localhost:8501`
- Start classifying images!

## How It Works

### Frontend (Streamlit)

- **Image Upload**: User selects or uploads an image
- **Encoding**: Image is converted to base64 (text-friendly format)
- **API Calls**: Frontend checks backend status and model metadata before inference
- **Info Panel**: A sidebar panel shows backend reachability, model loaded status, device, architecture, and number of classes
- **Top-K Mode**: Optional Top-K predictions with ranked probabilities
- **Display**: Result is shown as top-1 metric and optional Top-K table

### Backend (FastAPI)

1. **Receive Request**: API receives base64-encoded image
2. **Decode**: Image is decoded back to PIL Image
3. **Preprocess**: Image is resized to 224×224 pixels
4. **Inference**: Image passes through ConvNeXT backbone
5. **Classify**: Classification head outputs 15 probabilities
6. **Return**: Top prediction and confidence sent back to frontend

### Model Details

- **Architecture**: ConvNeXT-Base (pretrained on ImageNet)
- **Training**: Transfer learning with custom 15-class head
- **Epochs**: 8 epochs on scene classification dataset
- **Input Size**: 224×224 RGB images
- **Output**: 15 scene classes with confidence scores

## Project Structure

```
ImageClassification/
├── app.py                        # Streamlit frontend
├── fastapi_backend.py            # FastAPI backend server
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── models/
│   └── convnext_base-8epoch.pt   # Trained model weights
│
├── dataset/
│   ├── training/                 # Training images (15 classes)
│   └── validation/               # Validation images (15 classes)
│
└── wandb/                        # Training logs (optional)
```

## Configuration

### API Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| Backend Port | 8000 | Model inference server |
| Frontend Port | 8501 | Web interface |
| Timeout | 60s | API request timeout |
| Image Size | 224×224px | Model input size |

### Changing Ports

To use different ports, edit the configuration in the respective files:

**Backend (fastapi_backend.py)**:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change 8000 to your port
```

**Frontend (app.py)**:
```python
API_BASE_URL = "http://localhost:8000"  # Update if backend port changed
```

## Dependencies

- **streamlit**: Web UI framework
- **fastapi**: Backend API framework
- **uvicorn**: ASGI server for FastAPI
- **torch**: Deep learning framework
- **torchvision**: Computer vision models and utilities
- **pillow**: Image processing
- **requests**: HTTP client for frontend-backend communication

See `requirements.txt` for exact versions.

## Model Training

This model was trained using transfer learning:

1. **Backbone**: ConvNeXT-Base pretrained on ImageNet
2. **Head**: Custom linear classifier with 15 outputs
3. **Training**: 8 epochs on the scene classification dataset
4. **Validation**: Tested on held-out validation set

The model achieves high accuracy on scene classification tasks because:
- ConvNeXT learns good visual features from ImageNet
- Fine-tuning adapts these features to scene understanding
- 15 distinct scene categories have different visual characteristics

## API Documentation

How to access the API docs:

1. Start the backend:
   ```bash
   python fastapi_backend.py
   ```
2. Open Swagger UI in your browser: [http://localhost:8000/docs](http://localhost:8000/docs)
3. For a static markdown reference, check: [docs/api.md](docs/api.md)

- Full endpoint reference: [docs/api.md](docs/api.md)
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

## Streamlit Features

- Sidebar backend monitor (reachable, model loaded, device, architecture, classes)
- Toggle to switch between Top-1 and Top-K inference mode
- Slider to control K value in Top-K mode
- Ranked Top-K table with confidence percentages
- Works for both uploaded images and random validation images

## Tips & Tricks

- **Faster inference**: Use a GPU - predictions are 5-10x faster
- **Batch processing**: You can modify the backend to process multiple images
- **Confidence threshold**: Adjust the UI to only show predictions above a certain confidence
- **Dataset exploration**: Check the `dataset/validation/` folder to see example images

## Future Improvements

- [ ] Support for batch image processing
- [ ] Model download automation
- [ ] Different model architectures (ResNet, EfficientNet)
- [ ] Real-time webcam classification
- [ ] Explainability features (attention maps, saliency maps)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, Azure, Google Cloud)

## License

This project is for educational purposes.

## Author

Created as part of Machine Learning II course at ICAI.

## Troubleshooting

### Port Already in Use

If you get `error while attempting to bind on address ... Error 10048`:

**Windows:**
```bash
netstat -ano | findstr ":8000"              # Find process
taskkill /PID <PID> /F                       # Kill it
```

**Linux/Mac:**
```bash
lsof -i :8000
kill -9 <PID>
```

### Backend Not Responding

1. Check if backend is running on port 8000
2. Verify model file exists at `models/convnext_base-8epoch.pt`
3. Ensure GPU/CPU is available:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### Streamlit Not Starting

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version is 3.8+: `python --version`
3. Try running with explicit port: `streamlit run app.py --server.port 8501`

---

**Need help?** Check the troubleshooting section or review the code comments in `app.py` and `fastapi_backend.py`.
