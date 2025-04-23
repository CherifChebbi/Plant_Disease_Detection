# 🌿 Plant Disease Detection App – Streamlit Frontend

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import cv2
import matplotlib.cm as cm

# ===== App Config =====
st.set_page_config(page_title="🌿 Plant Disease Detection", layout="wide")
st.title("🌿 Plant Disease Detection using CNN & VGG16")
st.markdown("""
Upload a leaf image and choose a model to detect possible plant diseases. Grad-CAM will highlight the most relevant regions.
""")

# ===== Load models =====
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
    vgg_model = tf.keras.models.load_model("models/vgg16_model.h5")
    return cnn_model, vgg_model

cnn_model, vgg_model = load_models()
class_labels = sorted(os.listdir("data/split/train"))

# ===== Sidebar UI =====
st.sidebar.header("🌿 App Configuration")
uploaded_file = st.sidebar.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
model_choice = st.sidebar.selectbox("🧠 Select Model:", ["Custom CNN", "VGG16"])
model = cnn_model if model_choice == "Custom CNN" else vgg_model

# ===== Grad-CAM VGG16 only =====
def generate_gradcam_vggonly(img_array, full_model):
    vgg_model = full_model.get_layer('vgg16')
    conv_layer = vgg_model.get_layer('block5_conv3')

    grad_model = tf.keras.models.Model([vgg_model.input], [conv_layer.output])
    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array)
        tape.watch(conv_outputs)
        output = tf.reduce_mean(conv_outputs, axis=[1, 2])

    grads = tape.gradient(output, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = np.uint8(255 * heatmap.numpy())
    return heatmap

# ===== Main Display =====
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch)
    pred_class = class_labels[np.argmax(preds)]
    pred_conf = np.max(preds) * 100

    # Grad-CAM
    try:
        if model_choice == "VGG16":
            heatmap = generate_gradcam_vggonly(img_batch, model)
        else:
            from utils.gradcam import generate_gradcam
            last_conv = model.layers[-5].name
            heatmap = generate_gradcam(model, img_batch, last_conv_layer_name=last_conv)

        # Resize heatmap and overlay
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_resized]
        jet_heatmap = np.uint8(jet_heatmap * 255)
        jet_img = Image.fromarray(jet_heatmap)
        superimposed = Image.blend(img_resized, jet_img, alpha=0.5)

        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Uploaded Image")
            st.image(img_resized, caption="Input Leaf Image", use_container_width=True)
            st.success(f"✅ Predicted Class: **{pred_class}**")
            st.write(f"📊 Confidence: **{pred_conf:.2f}%**")
        with col2:
            st.subheader("🔍 Grad-CAM Visualization")
            st.image(superimposed, caption="Grad-CAM", use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Grad-CAM Error: {e}")
else:
    st.info("👈 Upload an image from the sidebar to begin.")

# ===== Footer =====
st.markdown("---")
st.markdown("""
📘 *Developed as part of a Deep Learning project on plant disease classification using the PlantVillage dataset.*

🔬 Models: Custom CNN & VGG16 (ImageNet pretrained)  
📊 Dataset: PlantVillage  
🧪 Frameworks: TensorFlow, Keras, Streamlit
""")
