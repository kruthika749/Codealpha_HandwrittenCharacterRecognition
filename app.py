# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import io
import os
import re

st.set_page_config(page_title="Handwritten Character Recognition", layout="centered")

# ---------------------------
# Model (same as training)
# ---------------------------
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ---------------------------
# Label lists (EMNIST Balanced reference)
# ---------------------------
EMNIST_BALANCED_CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s'
]

MNIST_CLASSES = [str(i) for i in range(10)]

# ---------------------------
# Preprocessing
# ---------------------------
def build_transform(dataset: str):

    def preprocess_image(img):
        # 1. Convert to grayscale
        img = img.convert("L")

        # 2. Invert image (EMNIST expects white text on black)
        img = ImageOps.invert(img)

        # 3. Threshold (remove noise)
        img = img.point(lambda x: 0 if x < 50 else 255, '1')
        img = img.convert("L")

        # 4. Crop tight around the character
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # 5. Resize longest side to 20px (MNIST convention)
        w, h = img.size
        scale = 20.0 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        # 6. Paste into 28×28 canvas centered
        new_img = Image.new("L", (28, 28), 0)
        left = (28 - img.size[0]) // 2
        top = (28 - img.size[1]) // 2
        new_img.paste(img, (left, top))

        # 7. EMNIST orientation fix
        if dataset == "emnist_balanced":
            new_img = new_img.rotate(-90, expand=True)
            new_img = ImageOps.mirror(new_img)

        return new_img

    # return transform object
    return transforms.Compose([
        transforms.Lambda(preprocess_image),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    # EMNIST REQUIRED FIXES
    if dataset == "emnist_balanced":
        new_img = new_img.rotate(90, expand=False)   # FIX: must be +90°
        new_img = ImageOps.mirror(new_img)           # EMNIST mirror

    # IMPORTANT: Convert to tensor + normalize (MUST for accuracy)
    tensor = transforms.ToTensor()(new_img)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)

    return tensor

    return transforms.Compose([
        transforms.Lambda(preprocess_image),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# ❗ FIXED INDENTATION: this function must be OUTSIDE build_transform
def preprocess_image_emnist(img):
    img = img.convert("L")
    img = img.resize((28, 28))
    img = img.rotate(-90, expand=True)
    img = ImageOps.mirror(img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform(img)

# ---------------------------
# Utility: normalize checkpoint dict keys (strip module.)
# ---------------------------
def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}
    return state_dict

# ---------------------------
# Robust detection of final fc weight key and num_classes
# ---------------------------
def infer_num_classes_from_state(state):
    candidates = []
    for k, v in state.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            candidates.append((k, v.shape))

    for k, shape in candidates:
        if k == "fc.5.weight":
            return shape[0], k

    for k, shape in candidates:
        if shape[1] == 128:
            return shape[0], k

    if candidates:
        k, shape = candidates[-1]
        return shape[0], k

    raise RuntimeError("Could not infer final linear layer from checkpoint state_dict.")

# ---------------------------
# Load model (auto-detect classes)
# ---------------------------
@st.cache_resource
def load_model_auto(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_state = torch.load(checkpoint_path, map_location="cpu")
    state = strip_module_prefix(raw_state)

    num_classes, final_weight_key = infer_num_classes_from_state(state)
    model = CNN(num_classes=int(num_classes))

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        model.fc[5] = nn.Linear(128, int(num_classes))

    model.eval()
    return model, int(num_classes)

# ---------------------------
# Helpers
# ---------------------------
def tensor_to_pil(img_t):
    arr = img_t.clone().detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    arr = (arr * 0.5 + 0.5) * 255
    return Image.fromarray(arr.astype(np.uint8))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("✍️ Handwritten Character Recognition")

col1, col2 = st.columns([2, 1])
with col2:
    dataset = st.selectbox("Dataset", options=["emnist_letters", "mnist"], index=0)

    checkpoint_path = st.text_input("Checkpoint path", value="pytorch_models/handwritten_cnn.pth")
    load_btn = st.button("Load model")

status = st.empty()
model = None
detected_num_classes = None
classes = None

if load_btn:
    try:
        status.info("Detecting & loading model...")
        model, detected_num_classes = load_model_auto(checkpoint_path)
        if detected_num_classes == 10:
            classes = MNIST_CLASSES
        else:
            classes = EMNIST_BALANCED_CLASSES[:detected_num_classes]
        status.success(f"Loaded model from {checkpoint_path} (num_classes={detected_num_classes})")
    except Exception as e:
        status.error(f"Failed to load model: {e}")
        st.stop()

if model is None and os.path.exists(checkpoint_path):
    try:
        model, detected_num_classes = load_model_auto(checkpoint_path)
        if detected_num_classes == 10:
            classes = MNIST_CLASSES
        else:
            classes = EMNIST_BALANCED_CLASSES[:detected_num_classes]
        status.success(f"Auto-loaded model (num_classes={detected_num_classes})")
    except Exception as e:
        status.warning(f"Auto-load failed: {e}")
        model = None
        detected_num_classes = None
        classes = None

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if uploaded:
    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("L")
    except Exception:
        st.error("Unable to read the uploaded file as an image.")
        st.stop()

    st.subheader("Uploaded Image")
    st.image(image, width=200)

    transform = build_transform(dataset)

    if dataset == "emnist_balanced":
        img_t = preprocess_image_emnist(image)
    else:
        img_t = transform(image)

    st.subheader("Processed Image")
    st.image(tensor_to_pil(img_t), width=150)

    if model is None:
        st.error("Model not loaded. Load a compatible checkpoint first.")
        st.stop()

    with torch.no_grad():
        out = model(img_t.unsqueeze(0))
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_idx = int(pred.item())
        confidence = float(conf.item()) * 100.0
        label = classes[pred_idx] if classes and pred_idx < len(classes) else str(pred_idx)

    st.subheader("Prediction")
    st.markdown(f"**Predicted:** `{label}` — **Confidence:** `{confidence:.2f}%`")

    st.markdown("**Top predictions:**")
    topk = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
    top_idxs = topk.indices[0].tolist()
    top_vals = topk.values[0].tolist()
    for idx, val in zip(top_idxs, top_vals):
        name = classes[idx] if classes and idx < len(classes) else str(idx)
        st.write(f"- {name}: {val*100:.2f}%")

st.markdown("---")
st.write("Tip: if your checkpoint was trained on EMNIST Balanced, keep dataset = emnist_balanced (the UI uses this only for preprocessing).")
