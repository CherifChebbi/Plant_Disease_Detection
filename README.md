# 🌿 Plant Disease Detection using Deep Learning (CNN & VGG16):

This project leverages deep learning to detect plant diseases from leaf images using a **custom CNN** and a **fine-tuned VGG16** model. It covers the entire ML lifecycle — from data exploration to deployment — and includes a user-friendly **Streamlit web application** for real-time inference and Grad-CAM visualizations.

---

## 📌 Overview:

- 📁 Dataset: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- 🧠 Models: Custom CNN, VGG16 (ImageNet pretrained)  
- 🧪 Frameworks: TensorFlow, Keras, OpenCV, Streamlit  
- 🎯 Goal: Build a reliable leaf disease detection system with visual explanations (Grad-CAM)

---

## 🧠 Models Used:

| Model        | Description                                         |
|--------------|-----------------------------------------------------|
| **Custom CNN** | A small CNN architecture trained from scratch       |
| **VGG16**     | Transfer learning using pretrained weights on ImageNet |

---

## 📌 Functionality:

- 📤 Upload a leaf image
- 🧠 Choose model (CNN or VGG16)
- ✅ Predict disease class
- 🔍 Grad-CAM heatmap overlay
- 📊 Confidence score display
