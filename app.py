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

# === Load model once ===
generator = None
if uploaded_file1 or uploaded_file2:
    generator = load_generator()

# === Handle first uploaded file ===
if uploaded_file1:
    st.markdown("### 📁 Image 1")
    image1 = Image.open(uploaded_file1).convert("RGB")
    w, h = image1.size
    satellite1 = image1.crop((0, 0, w // 2, h))

    input_tensor1 = transform(satellite1).unsqueeze(0)

    with st.spinner("🔧 Generating roadmap for Image 1..."):
        with torch.no_grad():
            output1 = generator(input_tensor1)
        roadmap1 = tensor_to_pil(output1)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🧭 Satellite Input 1")
        st.image(satellite1, use_container_width=True)
    with col_b:
        st.subheader("🗺 Predicted Roadmap 1")
        st.image(roadmap1, use_container_width=True)

# === Handle second uploaded file ===
if uploaded_file2:
    st.markdown("### 📁 Image 2")
    image2 = Image.open(uploaded_file2).convert("RGB")
    w, h = image2.size
    satellite2 = image2.crop((0, 0, w // 2, h))

    input_tensor2 = transform(satellite2).unsqueeze(0)

    with st.spinner("🔧 Generating roadmap for Image 2..."):
        with torch.no_grad():
            output2 = generator(input_tensor2)
        roadmap2 = tensor_to_pil(output2)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("🧭 Satellite Input 2")
        st.image(satellite2, use_container_width=True)
    with col_d:
        st.subheader("🗺 Predicted Roadmap 2")
        st.image(roadmap2, use_container_width=True)
