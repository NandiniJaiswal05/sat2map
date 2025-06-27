import streamlit as st
from PIL import Image
import torch
import os
import requests
import torchvision.transforms as transforms

# === Configuration ===
MODEL_URL = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
MODEL_PATH = "checkpoints.pth"

# === Load Generator Model ===
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
    if isinstance(checkpoint, dict) and 'gen_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['gen_model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# === Transform and Tensor to Image ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

def crop_left_half(image):
    w, h = image.size
    return image.crop((0, 0, w // 2, h))

# === Streamlit App ===
st.set_page_config(page_title="Change Detection", layout="wide")
st.markdown("<h3 style='text-align: center; color: gray;'>NRSC, ISRO</h3>", unsafe_allow_html=True)
st.title("🛰️ Change Detection")

# === Upload Two Images Side by Side ===
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("📤 Upload Satellite Image 1", type=["jpg", "jpeg", "png"], key="img1")
with col2:
    uploaded_file2 = st.file_uploader("📤 Upload Satellite Image 2", type=["jpg", "jpeg", "png"], key="img2")

# === Process and Display Each ===
generator = load_generator()

if uploaded_file1 or uploaded_file2:
    st.markdown("### 📁 Output Comparison")
    out_col1, out_col2 = st.columns(2)

    # Image 1 Processing
    if uploaded_file1:
        with out_col1:
            image1 = Image.open(uploaded_file1).convert("RGB")
            st.subheader("📸 Uploaded Image 1")
            st.image(image1, use_container_width=True)

            sat1 = crop_left_half(image1)
            st.subheader("🧭 Cropped Satellite 1")
            st.image(sat1, use_container_width=True)

            with st.spinner("🧠 Generating Roadmap 1..."):
                input_tensor1 = transform(sat1).unsqueeze(0)
                with torch.no_grad():
                    output1 = generator(input_tensor1)
                roadmap1 = tensor_to_pil(output1)
                st.subheader("🗺️ Predicted Roadmap 1")
                st.image(roadmap1, use_container_width=True)

    # Image 2 Processing
    if uploaded_file2:
        with out_col2:
            image2 = Image.open(uploaded_file2).convert("RGB")
            st.subheader("📸 Uploaded Image 2")
            st.image(image2, use_container_width=True)

            sat2 = crop_left_half(image2)
            st.subheader("🧭 Cropped Satellite 2")
            st.image(sat2, use_container_width=True)

            with st.spinner("🧠 Generating Roadmap 2..."):
                input_tensor2 = transform(sat2).unsqueeze(0)
                with torch.no_grad():
                    output2 = generator(input_tensor2)
                roadmap2 = tensor_to_pil(output2)
                st.subheader("🗺️ Predicted Roadmap 2")
                st.image(roadmap2, use_container_width=True)
