import base64
import io
import os
import random

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Scene Classifier",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2d3142;
}

/* Card-style containers */
.card {
    background: #1e2130;
    border: 1px solid #2d3142;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result badge */
.result-label {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.result-conf {
    font-size: 1.1rem;
    color: #a0aec0;
    margin-top: 0.25rem;
}

/* Confidence bar */
.conf-bar-bg {
    background: #2d3142;
    border-radius: 6px;
    height: 8px;
    margin: 4px 0 2px 0;
    width: 100%;
}
.conf-bar-fill {
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
    border-radius: 6px;
    height: 8px;
}

/* Status dot */
.dot-green { color: #48bb78; font-size: 0.8rem; }
.dot-red   { color: #fc8181; font-size: 0.8rem; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL      = "http://localhost:8000"
VALIDATION_DIR    = "./dataset/validation"
ALLOWED_FORMATS   = ("jpg", "jpeg", "png")
TIMEOUT           = 60
SCENE_CLASSES     = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial", "Inside city",
    "Kitchen", "Living room", "Mountain", "Office", "Open country",
    "Store", "Street", "Suburb", "Tall building"
]
CLASS_ICONS = {
    "Bedroom": "🛏️", "Coast": "🌊", "Forest": "🌲", "Highway": "🛣️",
    "Industrial": "🏭", "Inside city": "🏙️", "Kitchen": "🍳",
    "Living room": "🛋️", "Mountain": "⛰️", "Office": "💼",
    "Open country": "🌾", "Store": "🛒", "Street": "🚶",
    "Suburb": "🏡", "Tall building": "🏢"
}

# ── API helpers ───────────────────────────────────────────────────────────────
def _get(endpoint: str) -> dict:
    try:
        return requests.get(f"{API_BASE_URL}{endpoint}", timeout=TIMEOUT).json()
    except Exception:
        return {"status": "error"}

def _post(endpoint: str, payload: dict) -> dict:
    try:
        return requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=TIMEOUT).json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Backend not running on port 8000"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def classify(image: Image.Image, filename: str, model: str, k: int) -> dict:
    return _post("/predict-topk", {
        "image": image_to_base64(image),
        "filename": filename,
        "model": model,
        "k": k
    })

def get_random_image() -> tuple[str, str, Image.Image] | None:
    if not os.path.exists(VALIDATION_DIR):
        return None
    classes = [d for d in os.listdir(VALIDATION_DIR)
               if os.path.isdir(os.path.join(VALIDATION_DIR, d))]
    if not classes:
        return None
    cls = random.choice(classes)
    cls_path = os.path.join(VALIDATION_DIR, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(ALLOWED_FORMATS)]
    if not imgs:
        return None
    path = os.path.join(cls_path, random.choice(imgs))
    return path, cls, Image.open(path)

# ── Sidebar ───────────────────────────────────────────────────────────────────
health = _get("/health")
info   = _get("/model-info")
online = health.get("status") == "ok"

with st.sidebar:
    st.markdown("## 🏙️ Scene Classifier")
    st.markdown("---")

    status_html = (
        '<span class="dot-green">● Online</span>' if online
        else '<span class="dot-red">● Offline</span>'
    )
    st.markdown(f"**Backend** &nbsp; {status_html}", unsafe_allow_html=True)
    st.markdown(f"**Device** &nbsp; `{info.get('device', '—')}`")
    st.markdown("---")

    selected_model = st.selectbox(
        "🤖 Model",
        options=["ConvNeXt-Small", "EfficientNetV2-S"],
        index=0,
        help="ConvNeXt-Small: 96.47% acc  ·  EfficientNetV2-S: 96.00% acc, 2.3× faster"
    )
    topk = st.slider("Top-K results", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("**Classes**")
    for cls in SCENE_CLASSES:
        st.markdown(f"{CLASS_ICONS.get(cls, '•')} {cls}")

# ── Results renderer ──────────────────────────────────────────────────────────
def show_results(result: dict, true_class: str | None = None) -> None:
    if result.get("status") != "success":
        st.error(result.get("message", "Unknown error"))
        return

    preds = result.get("predictions", [])
    if not preds:
        st.warning("No predictions returned")
        return

    best = preds[0]
    icon = CLASS_ICONS.get(best["label"], "")
    conf = best["confidence"] * 100

    # Top result card
    correct = true_class and best["label"].lower() == true_class.lower()
    border_color = "#48bb78" if correct else ("#fc8181" if true_class else "#4f8ef7")

    st.markdown(f"""
    <div class="card" style="border-color:{border_color}; border-width:2px;">
        <div class="result-label">{icon} {best["label"]}</div>
        <div class="result-conf">{conf:.1f}% confidence · {selected_model}</div>
    </div>
    """, unsafe_allow_html=True)

    if true_class:
        if correct:
            st.success("✅ Correct prediction!")
        else:
            st.warning(f"❌ Expected: {CLASS_ICONS.get(true_class,'')} {true_class}")

    # Bar chart for all predictions
    st.markdown("**Confidence breakdown**")
    for pred in preds:
        label = pred["label"]
        c = pred["confidence"] * 100
        fill_pct = int(c)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; margin:4px 0;">
            <span style="width:130px; font-size:0.85rem; color:#cbd5e0;">{CLASS_ICONS.get(label,'')} {label}</span>
            <div class="conf-bar-bg" style="flex:1;">
                <div class="conf-bar-fill" style="width:{fill_pct}%;"></div>
            </div>
            <span style="width:46px; text-align:right; font-size:0.85rem; color:#a0aec0;">{c:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# Scene Image Classifier")
st.markdown("Upload a photo or test with a random validation image — the model will identify the scene.")
st.markdown("---")

tab1, tab2 = st.tabs(["📤 Upload Image", "🎲 Random Validation Image"])

with tab1:
    uploaded = st.file_uploader(
        "Drop an image here", type=list(ALLOWED_FORMATS), label_visibility="collapsed"
    )
    if uploaded:
        image = Image.open(uploaded)
        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            st.image(image, use_container_width=True, caption=uploaded.name)
        with col_res:
            if st.button("Classify →", key="btn_upload", use_container_width=True, type="primary"):
                with st.spinner("Running inference…"):
                    result = classify(image, uploaded.name, selected_model, topk)
                st.session_state["upload_result"] = result
            if "upload_result" in st.session_state:
                show_results(st.session_state["upload_result"])

with tab2:
    if st.button("🎲 Load random image", key="btn_load", use_container_width=True):
        data = get_random_image()
        if data:
            path, cls, _ = data
            st.session_state["rnd_path"] = path
            st.session_state["rnd_class"] = cls
            st.session_state.pop("rnd_result", None)
        else:
            st.error("Validation dataset not found at ./dataset/validation")

    if "rnd_path" in st.session_state:
        image = Image.open(st.session_state["rnd_path"])
        col_img, col_res = st.columns([1, 1], gap="large")
        with col_img:
            true_cls = st.session_state["rnd_class"]
            st.image(image, use_container_width=True,
                     caption=f"True class: {CLASS_ICONS.get(true_cls,'')} {true_cls}")
        with col_res:
            if st.button("Classify →", key="btn_rnd", use_container_width=True, type="primary"):
                with st.spinner("Running inference…"):
                    result = classify(
                        image,
                        os.path.basename(st.session_state["rnd_path"]),
                        selected_model, topk
                    )
                st.session_state["rnd_result"] = result
            if "rnd_result" in st.session_state:
                show_results(st.session_state["rnd_result"], st.session_state["rnd_class"])
