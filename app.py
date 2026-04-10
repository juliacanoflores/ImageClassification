import streamlit as st
import requests
import os
import random
from PIL import Image
import base64
import io

# Configuration
st.set_page_config(page_title="Scene Classifier", layout="wide")

API_URL = "http://localhost:8000/predict"
VALIDATION_DIR = "./dataset/validation"
ALLOWED_FORMATS = ("jpg", "jpeg", "png")
TIMEOUT = 60

SCENE_CLASSES = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside city",
    "Kitchen", "Living room", "Mountain", "Office", "Open country",
    "Store", "Street", "Suburb", "Tall building"
]


def load_image(file_path: str) -> Image.Image:
    return Image.open(file_path)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def send_to_api(image: Image.Image, filename: str) -> dict:
    try:
        payload = {
            "image": image_to_base64(image),
            "filename": filename
        }
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
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


def display_prediction(result: dict) -> None:
    if result["status"] == "success":
        st.success(f"Prediction: **{result['label']}**")
        confidence = result["confidence"] * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    else:
        st.error(result.get('message', 'Unknown error'))


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


# Page header
st.title("Scene Image Classifier")
st.write("Classify scenes by uploading an image or selecting from the validation dataset.")

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
                    result = send_to_api(image, uploaded_file.name)
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
                    result = send_to_api(image, os.path.basename(st.session_state.random_path))
                    display_prediction(result)
                    
                    if result["status"] == "success":
                        is_correct = result["label"].lower() == st.session_state.random_class.lower()
                        if is_correct:
                            st.balloons()
                            st.success("Correct prediction!")
                        else:
                            st.warning(f"Expected: {st.session_state.random_class}")


# Footer
st.divider()
st.caption(f"Available classes: {', '.join(SCENE_CLASSES)}")