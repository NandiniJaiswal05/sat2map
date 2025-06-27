import streamlit as st
from PIL import Image
import torch
import os
import requests
import torchvision.transforms as transforms

# === Configurations ===
MODEL_URL = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
MODEL_PATH = "checkpoints.pth"

@st.cache_resource
def load_generator():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Dropbox...")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

    try:
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=3,
            out_channels=3,
            init_features=64,
            pretrained=False,
            trust_repo=True
        )
    except Exception as e:
        st.error(f"❌ Failed to load model architecture: {e}")
        st.stop()

    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(checkpoint, dict) and 'gen_model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['gen_model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        st.error(f"❌ Failed to load weights: {e}")
        st.stop()

    return model

# === Preprocessing and Display Functions ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

# === Streamlit App ===
st.set_page_config(page_title="Change Detection", layout="centered")
st.markdown("<h3 style='text-align: center; color: gray;'>NRSC, ISRO</h3>", unsafe_allow_html=True)
st.title("Change Detection")

# Upload two satellite images side-by-side
col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("📤 Upload Satellite Image 1", type=["jpg", "jpeg", "png"], key="uploader1")
with col2:
    uploaded_file2 = st.file_uploader("📤 Upload Satellite Image 2", type=["jpg", "jpeg", "png"], key="uploader2")

# Load model only once
if uploaded_file1 or uploaded_file2:
    generator = load_generator()

def process_image(uploaded_file, image_label):
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    satellite = image.crop((0, 0, w // 2, h))
    input_tensor = transform(satellite).unsqueeze(0)
    with torch.no_grad():
        output = generator(input_tensor)
    roadmap = tensor_to_pil(output)
    return satellite, roadmap, image

# === Single image uploaded ===
if (uploaded_file1 and not uploaded_file2) or (uploaded_file2 and not uploaded_file1):
    uploaded_file = uploaded_file1 if uploaded_file1 else uploaded_file2
    idx = 1 if uploaded_file1 else 2
    satellite, roadmap, full_image = process_image(uploaded_file, idx)

    st.markdown(f"---\n### 📁 Image {idx}")
    st.image(full_image, caption=f"📸 Uploaded Image {idx}", use_container_width=True)

    st.subheader(f"🧭 Satellite Input {idx}")
    st.image(satellite, use_container_width=True)

    st.subheader(f"🗺 Predicted Roadmap {idx}")
    st.image(roadmap, use_container_width=True)

# === Both images uploaded ===
if uploaded_file1 and uploaded_file2:
    st.markdown("---\n### 📁 Both Images")
    col_a, col_b = st.columns(2)

    with col_a:
        satellite1, roadmap1, full_image1 = process_image(uploaded_file1, 1)
        st.subheader("📸 Uploaded Image 1")
        st.image(full_image1, use_container_width=True)
        st.subheader("🧭 Satellite Input 1")
        st.image(satellite1, use_container_width=True)
        st.subheader("🗺 Predicted Roadmap 1")
        st.image(roadmap1, use_container_width=True)

    with col_b:
        satellite2, roadmap2, full_image2 = process_image(uploaded_file2, 2)
        st.subheader("📸 Uploaded Image 2")
        st.image(full_image2, use_container_width=True)
        st.subheader("🧭 Satellite Input 2")
        st.image(satellite2, use_container_width=True)
        st.subheader("🗺 Predicted Roadmap 2")
        st.image(roadmap2, use_container_width=True)
