


import streamlit as st
from PIL import Image
import torch
import os
import requests
import torchvision.transforms as transforms
import torch.nn as nn

# === Configuration ===
MODEL_URL = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
MODEL_PATH = "checkpoints.pth"

# === UNet Model Definition ===
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = UNet._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = UNet._block(features * 2, features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.decoder2 = UNet._block(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.decoder1 = UNet._block(features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

# === Load Generator with Weights ===
@st.cache_resource
def load_generator():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Dropbox...")
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

    model = UNet(in_channels=3, out_channels=3)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(checkpoint, dict) and 'gen_model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['gen_model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        st.stop()

    return model

# === Utilities ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

def process_image_before_model(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    satellite = image.crop((0, 0, w // 2, h))
    return image, satellite

def run_model_on_satellite(satellite_tensor):
    generator = load_generator()
    with torch.no_grad():
        output = generator(satellite_tensor)
    return tensor_to_pil(output)

# === Streamlit UI ===
st.set_page_config(page_title="Satellite to Roadmap", layout="centered")
st.markdown("<h3 style='text-align: center; color: gray;'>NRSC, ISRO</h3>", unsafe_allow_html=True)
st.title("🛰 Change Detection")

col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("📤 Upload Satellite Image 1", type=["jpg", "jpeg", "png"], key="uploader1")
with col2:
    uploaded_file2 = st.file_uploader("📤 Upload Satellite Image 2", type=["jpg", "jpeg", "png"], key="uploader2")

# === Single Image Upload ===
if (uploaded_file1 and not uploaded_file2) or (uploaded_file2 and not uploaded_file1):
    uploaded_file = uploaded_file1 if uploaded_file1 else uploaded_file2
    idx = 1 if uploaded_file1 else 2
    image, satellite = process_image_before_model(uploaded_file)

    st.markdown(f"---\n### 📁 Image {idx}")
    st.image(image, caption=f"📸 Uploaded Image {idx} (Full)", use_container_width=True)
    st.subheader(f"🧭 Cropped Satellite {idx}")
    st.image(satellite, use_container_width=True)

    with st.spinner(f"🔧 Generating Roadmap {idx}..."):
        try:
            tensor = transform(satellite).unsqueeze(0)
            roadmap = run_model_on_satellite(tensor)
            st.subheader(f"🗺 Predicted Roadmap {idx}")
            st.image(roadmap, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Model error: {e}")


# === Both Images Uploaded ===
elif uploaded_file1 and uploaded_file2:
    st.markdown("---\n### 📁 Processing Both Images Side by Side")

    try:
        # Process both
        image1, satellite1 = process_image_before_model(uploaded_file1)
        image2, satellite2 = process_image_before_model(uploaded_file2)

        tensor1 = transform(satellite1).unsqueeze(0)
        tensor2 = transform(satellite2).unsqueeze(0)

        with st.spinner("🔧 Running model on both images..."):
            roadmap1 = run_model_on_satellite(tensor1)
            roadmap2 = run_model_on_satellite(tensor2)

        # Display side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image 1")
            st.image(image1, use_container_width=True)
            st.subheader("🧭 Cropped Satellite 1")
            st.image(satellite1, use_container_width=True)
            st.subheader("🗺 Predicted Roadmap 1")
            st.image(roadmap1, use_container_width=True)

        with col2:
            st.subheader("📸 Uploaded Image 2")
            st.image(image2, use_container_width=True)
            st.subheader("🧭 Cropped Satellite 2")
            st.image(satellite2, use_container_width=True)
            st.subheader("🗺 Predicted Roadmap 2")
            st.image(roadmap2, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error during image processing: {e}")
