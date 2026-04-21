import base64
import io
import os
import random

import requests
import streamlit as st
from PIL import Image

# Configuration
st.set_page_config(page_title="Scene Classifier", layout="wide")

API_BASE_URL = "http://localhost:8000"
API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_PREDICT_TOPK_URL = f"{API_BASE_URL}/predict-topk"
API_HEALTH_URL = f"{API_BASE_URL}/health"
API_CLASSES_URL = f"{API_BASE_URL}/classes"
API_MODEL_INFO_URL = f"{API_BASE_URL}/model-info"
VALIDATION_DIR = "./dataset/validation"
ALLOWED_FORMATS = ("jpg", "jpeg", "png")
TIMEOUT = 60

SCENE_CLASSES = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside city",
    "Kitchen", "Living room", "Mountain", "Office", "Open country",
    "Store", "Street", "Suburb", "Tall building"
]


def get_api_health() -> dict:
    try:
        response = requests.get(API_HEALTH_URL, timeout=TIMEOUT)
        return response.json()
    except Exception:
        return {"status": "error", "model_loaded": False, "device": "unknown"}


def get_model_info() -> dict:
    try:
        response = requests.get(API_MODEL_INFO_URL, timeout=TIMEOUT)
        return response.json()
    except Exception:
        return {
            "status": "error",
            "architecture": "unknown",
            "num_classes": len(SCENE_CLASSES),
            "device": "unknown"
        }


def get_scene_classes() -> list[str]:
    try:
        response = requests.get(API_CLASSES_URL, timeout=TIMEOUT)
        payload = response.json()
        classes = payload.get("classes", [])
        if isinstance(classes, list) and classes:
            return classes
    except Exception:
        pass
    return SCENE_CLASSES


def load_image(file_path: str) -> Image.Image:
    return Image.open(file_path)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def send_to_api(image: Image.Image, filename: str, model: str = "ConvNeXt-Small") -> dict:
    try:
        payload = {
            "image": image_to_base64(image),
            "filename": filename,
            "model": model
        }
        response = requests.post(API_PREDICT_URL, json=payload, timeout=TIMEOUT)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "label": "Connection Error",
            "confidence": 0.0,
            "message": "Backend not running. Start fastapi_backend.py on port 8000"
        }
    except Exception as e:
        return {
            "status": "error",
            "label": "Error",
            "confidence": 0.0,
            "message": str(e)
        }


def send_to_api_topk(image: Image.Image, filename: str, k: int, model: str = "ConvNeXt-Small") -> dict:
    try:
        payload = {
            "image": image_to_base64(image),
            "filename": filename,
            "k": int(k),
            "model": model
        }
        response = requests.post(API_PREDICT_TOPK_URL, json=payload, timeout=TIMEOUT)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "predictions": [],
            "message": "Backend not running. Start fastapi_backend.py on port 8000"
        }
    except Exception as e:
        return {
            "status": "error",
            "predictions": [],
            "message": str(e)
        }


def display_prediction(result: dict) -> None:
    if result["status"] == "success":
        st.success(f"Prediction: **{result['label']}**")
        confidence = result["confidence"] * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    else:
        st.error(result.get('message', 'Unknown error'))


def display_topk_predictions(result: dict) -> None:
    if result.get("status") != "success":
        st.error(result.get("message", "Unknown error"))
        return

    predictions = result.get("predictions", [])
    if not predictions:
        st.warning("No predictions returned by backend")
        return

    best = predictions[0]
    st.success(f"Top prediction: **{best['label']}**")
    st.metric("Top-1 confidence", f"{best['confidence'] * 100:.1f}%")

    rows = []
    for idx, pred in enumerate(predictions, start=1):
        rows.append({
            "rank": idx,
            "label": pred["label"],
            "confidence (%)": round(pred["confidence"] * 100, 2)
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def get_random_image() -> tuple[str, str, Image.Image] | None:
    if not os.path.exists(VALIDATION_DIR):
        return None
    
    classes = [d for d in os.listdir(VALIDATION_DIR) 
               if os.path.isdir(os.path.join(VALIDATION_DIR, d))]
    
    if not classes:
        return None
    
    class_name = random.choice(classes)
    class_path = os.path.join(VALIDATION_DIR, class_name)
    
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(ALLOWED_FORMATS)]
    
    if not images:
        return None
    
    image_name = random.choice(images)
    image_path = os.path.join(class_path, image_name)
    
    return image_path, class_name, load_image(image_path)


health_payload = get_api_health()
model_info_payload = get_model_info()
scene_classes = get_scene_classes()

# Page header
st.title("Scene Image Classifier")
st.write("Classify scenes by uploading an image or selecting from the validation dataset.")

with st.sidebar:
    st.subheader("Backend Status")
    if health_payload.get("status") == "ok":
        st.success("Backend reachable")
    else:
        st.error("Backend not reachable")

    st.write(f"Model loaded: {health_payload.get('model_loaded', False)}")
    st.write(f"Device: {model_info_payload.get('device', 'unknown')}")
    st.write(f"Architecture: {model_info_payload.get('architecture', 'unknown')}")
    st.write(f"Classes: {model_info_payload.get('num_classes', len(scene_classes))}")

    st.divider()
    selected_model = st.selectbox(
        "Model",
        options=["ConvNeXt-Small", "EfficientNetV2-S"],
        index=0,
        help="ConvNeXt-Small: 96.47% acc · EfficientNetV2-S: 96.00% acc (faster)"
    )
    use_topk = st.toggle("Use Top-K prediction", value=True)
    topk_value = st.slider("K value", min_value=2, max_value=min(10, len(scene_classes)), value=min(3, len(scene_classes)))

# Two tabs for different input methods
tab1, tab2 = st.tabs(["Upload Image", "Random Image"])

with tab1:
    st.subheader("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image", type=list(ALLOWED_FORMATS))
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.write("")
            if st.button("Classify", key="btn_upload"):
                with st.spinner("Analyzing..."):
                    if use_topk:
                        result = send_to_api_topk(image, uploaded_file.name, topk_value, selected_model)
                        display_topk_predictions(result)
                    else:
                        result = send_to_api(image, uploaded_file.name, selected_model)
                        display_prediction(result)


with tab2:
    st.subheader("Random Validation Image")
    
    if st.button("Load Random Image"):
        random_data = get_random_image()
        if random_data:
            st.session_state.random_path, st.session_state.random_class, _ = random_data
        else:
            st.error("Dataset not found or empty")
    
    if "random_path" in st.session_state:
        image = load_image(st.session_state.random_path)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption=f"True class: {st.session_state.random_class}",
                    use_container_width=True)
        
        with col2:
            st.write("")
            if st.button("Classify", key="btn_random"):
                with st.spinner("Analyzing..."):
                    if use_topk:
                        result = send_to_api_topk(image, os.path.basename(st.session_state.random_path), topk_value, selected_model)
                        display_topk_predictions(result)
                        predicted_label = result.get("predictions", [{}])[0].get("label", "") if result.get("status") == "success" else ""
                    else:
                        result = send_to_api(image, os.path.basename(st.session_state.random_path), selected_model)
                        display_prediction(result)
                        predicted_label = result.get("label", "") if result.get("status") == "success" else ""

                    if result.get("status") == "success":
                        is_correct = predicted_label.lower() == st.session_state.random_class.lower()
                        if is_correct:
                            st.balloons()
                            st.success("Correct prediction!")
                        else:
                            st.warning(f"Expected: {st.session_state.random_class}")


# Footer
st.divider()
st.caption(f"Available classes: {', '.join(scene_classes)}")