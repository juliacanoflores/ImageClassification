# Scene Image Classifier

Automatic classification of real-estate scene images into 15 categories using transfer learning. Built as part of the Machine Learning II course at ICAI.

**Repository:** [github.com/juliacanoflores/ImageClassification](https://github.com/juliacanoflores/ImageClassification)  
**W&B project:** [javi_paula_julia/image-classification](https://wandb.ai/javi_paula_julia/image-classification)

**Best model:** ConvNeXt-Small — **96.47% val accuracy / 96.47% F1 macro** (best of 36 W&B sweep runs).

## Scene Categories

Bedroom · Coast · Forest · Highway · Industrial · Inside city · Kitchen · Living room · Mountain · Office · Open country · Store · Street · Suburb · Tall building

## Architecture

```
┌─────────────────────────────────────────┐
│   Streamlit Frontend  (Port 8501)       │
│   - Image upload / random validation    │
│   - Model selector (ConvNeXt / EffNet)  │
│   - Top-1 and Top-K prediction display  │
│   - Confidence bar chart                │
└────────────┬────────────────────────────┘
             │ HTTP JSON (base64 image)
             ↓
┌─────────────────────────────────────────┐
│   FastAPI Backend  (Port 8000)          │
│   - ConvNeXt-Small + EfficientNetV2-S   │
│   - /predict  /predict-topk             │
│   - /classes  /model-info  /health      │
└─────────────────────────────────────────┘
```

Both models are loaded at backend startup. The frontend lets you switch between them at inference time with no restart required.

## Reproducible Setup

All experiments use **seed 42** (fixed across `torch`, `numpy`, and `random`). The full environment is pinned in `requirements.txt` (Python 3.12, PyTorch 2.2.1, torchvision 0.17.1).

```bash
# 1. Clone the repository
git clone https://github.com/juliacanoflores/ImageClassification.git
cd ImageClassification

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# python -m venv .venv && .venv\Scripts\activate    # Windows

# 3. Install exact dependencies
pip install -r requirements.txt

# 4. Download trained model weights
wandb artifact get javi_paula_julia/image-classification/best-ConvNeXt-Small:latest --root models/
wandb artifact get javi_paula_julia/image-classification/best-EfficientNetV2-S:latest --root models/

# 5. Start the backend (terminal 1)
python src/fastapi_backend.py

# 6. Start the frontend (terminal 2)
streamlit run src/app.py        # original
streamlit run src/app_v2.py     # redesigned UI (dark theme, confidence bars)

# 7. Open http://localhost:8501
```

API docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)  
Static API reference: [docs/api.md](docs/api.md)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Quick health check — lists loaded models |
| GET | `/health` | Detailed health (models loaded, device) |
| GET | `/classes` | List of 15 class labels |
| GET | `/model-info` | Available models, input size, device |
| POST | `/predict` | Top-1 prediction from base64 image |
| POST | `/predict-topk` | Top-k predictions with confidences |

**POST /predict** — request body:
```json
{ "image": "<base64-encoded image>", "filename": "photo.jpg", "model": "ConvNeXt-Small" }
```

**POST /predict-topk** — request body:
```json
{ "image": "<base64-encoded image>", "k": 3, "model": "EfficientNetV2-S" }
```

`"model"` is optional and defaults to `"ConvNeXt-Small"`. Valid values: `"ConvNeXt-Small"`, `"EfficientNetV2-S"`.

## Project Structure

```
ImageClassification/
├── src/
│   ├── app.py                # Streamlit frontend
│   ├── fastapi_backend.py    # FastAPI backend (both models)
│   └── cnn.py                # CNN class + training utilities
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── sweep_train.py        # Single-run training function (W&B agent)
│   ├── launch_sweeps.py      # Create and launch W&B sweeps
│   └── analyze_results.py    # Parse results.csv and print comparison table
├── models/
│   ├── ConvNeXt-Small.pt     # Best ConvNeXt-Small weights (96.47%)
│   └── EfficientNetV2.pt     # Best EfficientNetV2-S weights (96.00%)
├── dataset/
│   ├── training/             # 2,985 training images (15 classes)
│   └── validation/           # 1,500 validation images (15 classes)
├── data/confusion_matrix/    # Per-model confusion matrices (CSV from W&B)
├── report/                   # LaTeX technical report
└── docs/api.md               # Static API documentation
```

## Models

| Model | Val Acc | F1 Macro | s/epoch | Notes |
|-------|---------|----------|---------|-------|
| **ConvNeXt-Small** | **96.47%** | **96.47%** | 99.9 | Default — highest accuracy |
| EfficientNetV2-S | 96.00% | 96.00% | 43.5 | 2.3× faster, recommended for CPU/serverless |
| Swin-T | 92.27% | 92.21% | 26.6 | Baseline only, not served |

Training uses a **two-phase transfer learning** strategy: backbone frozen during warmup, then last N blocks unfrozen with discriminative learning rates (`lr_backbone` ≪ `lr_head`). Hyperparameter search via **Bayesian optimisation** (W&B sweeps, 12 runs per model, 36 total).

### Reproducing the sweep

```bash
python scripts/launch_sweeps.py --create-only   # create sweeps (no GPU needed)
python scripts/launch_sweeps.py --count 12       # run all agents (requires GPU)
python scripts/launch_sweeps.py --model ConvNeXt-Small --count 12  # single model
```

## Troubleshooting

**Port already in use:**
```bash
lsof -i :8000 && kill -9 <PID>
```

**Model not found:** download weights with the `wandb artifact get` commands in step 4 above.

**Backend not responding:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Authors

- Javier Arroyo García
- Julia Cano Flores
- Paula Durá Fuster

ICAI · Machine Learning II · 2025–2026
