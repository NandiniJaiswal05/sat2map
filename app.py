import streamlit as st
from PIL import Image
import torch
import os
import requests
import torchvision.transforms as transforms

# === Configurations ===
MODEL_URL = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
MODEL_PATH = "checkpoints.pth"

# === Load Model ===
@st.cache_resource
def load_generator():
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'unet',
        in_channels=3,
        out_channels=3,
        init_features=64,
        pretrained=False,
        trust_repo=True
    )
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['gen_model_state_dict'] if 'gen_model_state_dict' in checkpoint else checkpoint)
    model.eval()
    return model

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    satellite = image.crop((0, 0, w // 2, h))
    tensor = transform(satellite).unsqueeze(0)
    return image, satellite, tensor

# === UI ===
st.set_page_config(page_title="Change Detection", layout="centered")
st.markdown("<h3 style='text-align: center; color: gray;'>NRSC, ISRO</h3>", unsafe_allow_html=True)
st.title("🛰️ Change Detection")

# Upload section
col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("📤 Upload Satellite Image 1", type=["jpg", "jpeg", "png"], key="img1")
with col2:
    uploaded_file2 = st.file_uploader("📤 Upload Satellite Image 2", type=["jpg", "jpeg", "png"], key="img2")

# Load model once
generator = None
if uploaded_file1 or uploaded_file2:
    with st.spinner("🔧 Loading model..."):
        generator = load_generator()

# Display results
if uploaded_file1 and uploaded_file2:
    st.markdown("### 📁 Both Images Side by Side")
    col1, col2 = st.columns(2)

    with col1:
        image, satellite, tensor = process_image(uploaded_file1)
        st.subheader("📸 Uploaded Image 1")
        st.image(image, use_container_width=True)
        st.subheader("🧭 Cropped Satellite 1")
        st.image(satellite, use_container_width=True)
        with torch.no_grad():
            output = generator(tensor)
            roadmap = tensor_to_pil(output)
        st.subheader("🗺 Predicted Roadmap 1")
        st.image(roadmap, use_container_width=True)

    with col2:
        image, satellite, tensor = process_image(uploaded_file2)
        st.subheader("📸 Uploaded Image 2")
        st.image(image, use_container_width=True)
        st.subheader("🧭 Cropped Satellite 2")
        st.image(satellite, use_container_width=True)
        with torch.no_grad():
            output = generator(tensor)
            roadmap = tensor_to_pil(output)
        st.subheader("🗺 Predicted Roadmap 2")
        st.image(roadmap, use_container_width=True)

elif uploaded_file1:
    st.markdown("### 🖼 Single Image Mode")
    image, satellite, tensor = process_image(uploaded_file1)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    st.image(satellite, caption="🧭 Cropped Satellite", use_container_width=True)
    with torch.no_grad():
        output = generator(tensor)
        roadmap = tensor_to_pil(output)
    st.image(roadmap, caption="🗺 Predicted Roadmap", use_container_width=True)

elif uploaded_file2:
    st.markdown("### 🖼 Single Image Mode")
    image, satellite, tensor = process_image(uploaded_file2)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    st.image(satellite, caption="🧭 Cropped Satellite", use_container_width=True)
    with torch.no_grad():
        output = generator(tensor)
        roadmap = tensor_to_pil(output)
    st.image(roadmap, caption="🗺 Predicted Roadmap", use_container_width=True)
