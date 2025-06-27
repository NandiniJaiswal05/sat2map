import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import requests

# === MODEL CONFIG ===
MODEL_URL = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
MODEL_PATH = "checkpoints.pth"

# === UNet MODEL ===
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super().__init__()
        f = init_features
        self.encoder1 = self._block(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._block(f, f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = self._block(f*2, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, 2)
        self.decoder2 = self._block(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, 2)
        self.decoder1 = self._block(f*2, f)
        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        return self.final(self.decoder1(dec1))

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    model = UNet()
    weights = torch.load(MODEL_PATH, map_location='cpu')
    if isinstance(weights, dict) and 'gen_model_state_dict' in weights:
        model.load_state_dict(weights['gen_model_state_dict'])
    else:
        model.load_state_dict(weights)
    model.eval()
    return model

# === UTILITIES ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    cropped = image.crop((0, 0, image.width // 2, image.height))
    return image, cropped

def predict(model, cropped_img):
    input_tensor = transform(cropped_img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    output = output.squeeze(0).clamp(0, 1)
    return transforms.ToPILImage()(output)

# === UI ===
st.set_page_config(page_title="Satellite to Roadmap", layout="wide")
st.title("🛰 Change Detection - Side by Side")

# === Upload Area ===
col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    uploaded_file1 = st.file_uploader("📤 Upload Satellite Image 1", type=["jpg", "jpeg", "png"], key="img1")
with col_upload2:
    uploaded_file2 = st.file_uploader("📤 Upload Satellite Image 2", type=["jpg", "jpeg", "png"], key="img2")

# === Process Both if Available ===
if uploaded_file1 and uploaded_file2:
    col_left, col_right = st.columns(2)
    model = load_model()

    # === LEFT IMAGE ===
    with col_left:
        img1, crop1 = process_image(uploaded_file1)
        st.subheader("📸 Image 1")
        st.image(img1, use_container_width=True)
        st.subheader("🧭 Cropped Satellite 1")
        st.image(crop1, use_container_width=True)
        st.subheader("🗺 Roadmap 1")
        output1 = predict(model, crop1)
        st.image(output1, use_container_width=True)

    # === RIGHT IMAGE ===
    with col_right:
        img2, crop2 = process_image(uploaded_file2)
        st.subheader("📸 Image 2")
        st.image(img2, use_container_width=True)
        st.subheader("🧭 Cropped Satellite 2")
        st.image(crop2, use_container_width=True)
        st.subheader("🗺 Roadmap 2")
        output2 = predict(model, crop2)
        st.image(output2, use_container_width=True)

elif uploaded_file1 or uploaded_file2:
    uploaded_file = uploaded_file1 if uploaded_file1 else uploaded_file2
    index = 1 if uploaded_file1 else 2
    img, crop = process_image(uploaded_file)
    model = load_model()
    st.subheader(f"📸 Uploaded Image {index}")
    st.image(img, use_container_width=True)
    st.subheader(f"🧭 Cropped Satellite {index}")
    st.image(crop, use_container_width=True)
    st.subheader(f"🗺 Predicted Roadmap {index}")
    output = predict(model, crop)
    st.image(output, use_container_width=True)




# import streamlit as st
# from PIL import Image
# import torch
# import os
# import requests
# import torchvision.transforms as transforms
# import torch.nn as nn

# # === Configuration ===
# MODEL_URL = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
# MODEL_PATH = "checkpoints.pth"

# # === UNet Model Definition ===
# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, init_features=64):
#         super(UNet, self).__init__()
#         features = init_features
#         self.encoder1 = UNet._block(in_channels, features)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.encoder2 = UNet._block(features, features * 2)
#         self.pool2 = nn.MaxPool2d(2, 2)

#         self.bottleneck = UNet._block(features * 2, features * 4)

#         self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
#         self.decoder2 = UNet._block(features * 4, features * 2)
#         self.up1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
#         self.decoder1 = UNet._block(features * 2, features)

#         self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         bottleneck = self.bottleneck(self.pool2(enc2))
#         dec2 = self.up2(bottleneck)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.up1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         return self.conv(dec1)

#     @staticmethod
#     def _block(in_channels, features):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(features, features, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )

# # === Load Generator with Weights ===
# @st.cache_resource
# def load_generator():
#     if not os.path.exists(MODEL_PATH):
#         st.info("📥 Downloading model from Dropbox...")
#         try:
#             with requests.get(MODEL_URL, stream=True) as r:
#                 r.raise_for_status()
#                 with open(MODEL_PATH, 'wb') as f:
#                     for chunk in r.iter_content(8192):
#                         f.write(chunk)
#         except Exception as e:
#             st.error(f"❌ Failed to download model: {e}")
#             st.stop()

#     model = UNet(in_channels=3, out_channels=3)
#     try:
#         checkpoint = torch.load(MODEL_PATH, map_location='cpu')
#         if isinstance(checkpoint, dict) and 'gen_model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['gen_model_state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
#         model.eval()
#     except Exception as e:
#         st.error(f"❌ Failed to load model weights: {e}")
#         st.stop()

#     return model

# # === Utilities ===
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# def tensor_to_pil(tensor_img):
#     tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
#     return transforms.ToPILImage()(tensor_img)

# def process_image_before_model(uploaded_file):
#     image = Image.open(uploaded_file).convert("RGB")
#     w, h = image.size
#     satellite = image.crop((0, 0, w // 2, h))
#     return image, satellite

# def run_model_on_satellite(satellite_tensor):
#     generator = load_generator()
#     with torch.no_grad():
#         output = generator(satellite_tensor)
#     return tensor_to_pil(output)

# # === Streamlit UI ===
# st.set_page_config(page_title="Satellite to Roadmap", layout="centered")
# st.markdown("<h3 style='text-align: center; color: gray;'>NRSC, ISRO</h3>", unsafe_allow_html=True)
# st.title("🛰 Change Detection")

# col1, col2 = st.columns(2)
# with col1:
#     uploaded_file1 = st.file_uploader("📤 Upload Satellite Image 1", type=["jpg", "jpeg", "png"], key="uploader1")
# with col2:
#     uploaded_file2 = st.file_uploader("📤 Upload Satellite Image 2", type=["jpg", "jpeg", "png"], key="uploader2")

# # === Single Image Upload ===
# if (uploaded_file1 and not uploaded_file2) or (uploaded_file2 and not uploaded_file1):
#     uploaded_file = uploaded_file1 if uploaded_file1 else uploaded_file2
#     idx = 1 if uploaded_file1 else 2
#     image, satellite = process_image_before_model(uploaded_file)

#     st.markdown(f"---\n### 📁 Image {idx}")
#     st.image(image, caption="📸 Uploaded Full Image", use_container_width=True)
#     st.subheader("🧭 Cropped Left Half (Satellite)")
#     st.image(satellite, use_container_width=True)

#     with st.spinner("🔧 Running model..."):
#         try:
#             tensor = transform(satellite).unsqueeze(0)
#             roadmap = run_model_on_satellite(tensor)
#             st.subheader("🗺 Predicted Roadmap")
#             st.image(roadmap, use_container_width=True)
#         except Exception as e:
#             st.error(f"❌ Model error: {e}")

# # === Both Images Uploaded ===
# if uploaded_file1 and uploaded_file2:
#     st.markdown("### 📁 Both Images Side by Side")

#     try:
#         # Load images individually
#         image1 = Image.open(uploaded_file1).convert("RGB")
#         image2 = Image.open(uploaded_file2).convert("RGB")

#         col1, col2 = st.columns(2)

#         # === Image 1 ===
#         with col1:
#             st.subheader("📸 Uploaded Image 1")
#             st.image(image1, caption="Image 1", use_container_width=True)
#             satellite1 = image1.crop((0, 0, image1.width // 2, image1.height))
#             st.subheader("🧭 Cropped Satellite 1")
#             st.image(satellite1, use_container_width=True)
#             with st.spinner("🔧 Generating Roadmap 1..."):
#                 tensor1 = transform(satellite1).unsqueeze(0)
#                 roadmap1 = run_model_on_satellite(tensor1)
#                 st.subheader("🗺 Predicted Roadmap 1")
#                 st.image(roadmap1, use_container_width=True)

#         # === Image 2 ===
#         with col2:
#             st.subheader("📸 Uploaded Image 2")
#             st.image(image2, caption="Image 2", use_container_width=True)
#             satellite2 = image2.crop((0, 0, image2.width // 2, image2.height))
#             st.subheader("🧭 Cropped Satellite 2")
#             st.image(satellite2, use_container_width=True)
#             with st.spinner("🔧 Generating Roadmap 2..."):
#                 tensor2 = transform(satellite2).unsqueeze(0)
#                 roadmap2 = run_model_on_satellite(tensor2)
#                 st.subheader("🗺 Predicted Roadmap 2")
#                 st.image(roadmap2, use_container_width=True)

#     except Exception as e:
#         st.error(f"❌ Unexpected error: {e}")



# # # === Both Images Uploaded ===
# # if uploaded_file1 and uploaded_file2:
# #     st.markdown("---\n### 📁 Both Images Side by Side")
# #     col_a, col_b = st.columns(2)

# #     # === Image 1 Processing ===
# #     with col_a:
# #         image1, satellite1 = process_image_before_model(uploaded_file1)
# #         st.subheader("📸 Uploaded Image 1")
# #         st.image(image1, use_container_width=True)
# #         st.subheader("🧭 Cropped Satellite 1")
# #         st.image(satellite1, use_container_width=True)

# #         with st.spinner("🔧 Generating Roadmap 1..."):
# #             try:
# #                 tensor1 = transform(satellite1).unsqueeze(0)
# #                 roadmap1 = run_model_on_satellite(tensor1)
# #                 st.subheader("🗺 Predicted Roadmap 1")
# #                 st.image(roadmap1, use_container_width=True)
# #                 st.markdown("&nbsp;", unsafe_allow_html=True)
# #             except Exception as e:
# #                 st.error(f"❌ Error in Image 1: {e}")

# #     # === Image 2 Processing ===
# #     with col_b:
# #         image2, satellite2 = process_image_before_model(uploaded_file2)
# #         st.subheader("📸 Uploaded Image 2")
# #         st.image(image2, use_container_width=True)
# #         st.subheader("🧭 Cropped Satellite 2")
# #         st.image(satellite2, use_container_width=True)

# #         with st.spinner("🔧 Generating Roadmap 2..."):
# #             try:
# #                 tensor2 = transform(satellite2).unsqueeze(0)
# #                 roadmap2 = run_model_on_satellite(tensor2)
# #                 st.subheader("🗺 Predicted Roadmap 2")
# #                 st.image(roadmap2, use_container_width=True)
# #                 st.markdown("&nbsp;", unsafe_allow_html=True)
# #             except Exception as e:
# #                 st.error(f"❌ Error in Image 2: {e}")
