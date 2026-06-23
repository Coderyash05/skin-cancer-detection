"""
Streamlit web app for the HAM10000 skin lesion classifier
(EfficientNetB3, from Skin_Cancer_Detection.ipynb).

Run locally:
    streamlit run app.py

Loads the exact model produced by the notebook's last cell:
    model.save("skin_cancer_efficientnet_final.keras")

Preprocessing mirrors predict_single() in the notebook exactly:
resize -> keep 0-255 float32 -> efficientnet.preprocess_input.
(No /255 rescale -- that's the bug that collapsed training to majority-class-only.)

Grad-CAM mirrors grad_cam()/show_grad_cam() in the notebook exactly, including
calling the grad-model with training=False (the augmentation layer is inside
the graph, so this matters for inference).
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from matplotlib import colormaps

# ---------------------------------------------------------------------------
# Config -- mirrors the notebook's Step 2 config cell
# ---------------------------------------------------------------------------
IMG_SIZE = 224
NUM_CLASSES = 7

CLASS_NAMES = {
    0: "akiec - Actinic Keratoses / Intraepithelial Carcinoma",
    1: "bcc   - Basal Cell Carcinoma",
    2: "bkl   - Benign Keratosis-like Lesions",
    3: "df    - Dermatofibroma",
    4: "nv    - Melanocytic Nevi",
    5: "vasc  - Vascular Lesions",
    6: "mel   - Melanoma",
}

MODEL_PATH = os.environ.get("MODEL_PATH", "skin_cancer_efficientnet_final.keras")
# Optional fallback: download the model from a Hugging Face Hub *model* repo
# if it isn't found locally. Not required for Spaces (it auto-LFS's the
# ~43MB .keras file if you just commit it into the repo) -- only useful if
# you want to host/version the weights separately from the app code.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "skin_cancer_efficientnet_final.keras")


@st.cache_resource(show_spinner="Loading model (first request only)...")
def load_model():
    path = MODEL_PATH
    if not os.path.exists(path) and HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
    return tf.keras.models.load_model(path)


# ---------------------------------------------------------------------------
# Preprocessing -- identical pipeline to predict_single() in the notebook
# ---------------------------------------------------------------------------
def preprocess_uploaded_image(file_bytes):
    """file_bytes: raw bytes from st.file_uploader. Decodes jpg/png, resizes
    to IMG_SIZE, then applies EfficientNet's preprocess_input directly on
    0-255 values -- NOT a /255 rescale first."""
    img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)              # keep 0-255
    display_img = img.numpy().astype("uint8")   # for showing the original
    img = preprocess_input(img)                  # EfficientNet normalization
    batch = img.numpy()[np.newaxis]
    return display_img, batch


# ---------------------------------------------------------------------------
# Grad-CAM -- identical logic to grad_cam()/show_grad_cam() in the notebook
# ---------------------------------------------------------------------------
def last_conv_layer_name(model):
    """Last layer in the full model emitting a 4D feature map (EfficientNet's
    top_conv). Found by output rank -- same approach as the notebook, robust
    to layer-class naming."""
    name = None
    for layer in model.layers:
        try:
            if len(layer.output.shape) == 4:
                name = layer.name
        except Exception:
            continue
    if name is None:
        raise ValueError("No 4D feature-map layer found.")
    return name


def grad_cam(model, img_array, class_idx=None):
    conv_name = last_conv_layer_name(model)
    grad_model = Model(model.inputs, [model.get_layer(conv_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_heatmap(display_img, cam, alpha=0.45):
    cam_resized = tf.image.resize(
        cam[..., np.newaxis],
        [IMG_SIZE, IMG_SIZE]
    )[..., 0].numpy()

    from matplotlib import colormaps
    heat_rgb = colormaps["jet"](cam_resized)[..., :3]

    overlay = (
        (1 - alpha) * (display_img / 255.0)
        + alpha * heat_rgb
    )

    return np.clip(overlay, 0, 1)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="centered")

st.title("🔬 Skin Lesion Classifier")
st.caption("EfficientNetB3 transfer learning on HAM10000 — 7-class dermatoscopic lesion classification")

st.warning(
    "**This is a learning/portfolio project, not a medical tool.** "
    "It must never be used for real diagnosis. If you have a concerning "
    "skin lesion, please see a dermatologist."
)

uploaded_file = st.file_uploader("Upload a dermatoscopic image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    display_img, batch = preprocess_uploaded_image(file_bytes)

    st.image(display_img, caption="Uploaded image", width=300)

    with st.spinner("Running inference..."):
        model = load_model()
        probs = model.predict(batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    confidence = probs[pred_idx] * 100

    st.subheader("Prediction")
    st.write(f"**{CLASS_NAMES[pred_idx]}**")
    st.write(f"Confidence: {confidence:.1f}%")

    short_labels = {
        CLASS_NAMES[i].split("-")[0].strip(): float(probs[i]) for i in range(NUM_CLASSES)
    }
    st.bar_chart(short_labels)

    st.subheader("Grad-CAM")
    st.caption("Highlights which pixels most influenced this prediction.")
    try:
        cam = grad_cam(model, batch, class_idx=pred_idx)
        overlay = overlay_heatmap(display_img, cam)
        st.image(overlay, caption="Grad-CAM overlay", width=300)
    except Exception as e:
        st.info(f"Grad-CAM unavailable: {e}")
else:
    st.info("Upload an image to get a prediction.")
