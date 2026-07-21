# 🌾 FasalAI — Agricultural Intelligence Platform

**FasalAI** is a comprehensive, client-side web application designed to empower farmers with hyper-local agricultural intelligence, AI-driven disease diagnosis, real-time satellite telemetry, and multilingual voice assistance.

---

## ✨ Features

- 🛰️ **Real-Time Satellite Data**: Live NDVI vegetation tracking, soil moisture levels, and weather monitoring powered by NASA and Open-Meteo APIs.
- 🧪 **Smart Crop Rankings**: AI-driven crop matching algorithm ranking optimal crops based on soil composition, climate, and seasonal suitability.
- 🔬 **AI Plant Disease Diagnosis**: On-device plant leaf disease identification running ConvNeXt Tiny via ONNX Runtime Web in-browser with zero cloud latency.
- 🤖 **CropLens AI Assistant**: Multilingual AI assistant powered by Groq LLaMA supporting English, Hindi, and Punjabi with speech recognition.
- 📊 **Market Trends & Mandi Prices**: Live price trends, Mandi updates, and Minimum Support Price (MSP) tracking for popular produce.
- 🌌 **Interactive 3D Visualizations**: 3D farm field models rendered using Three.js and interactive maps using Leaflet.

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, Vanilla CSS (Dark & Light "Sunlit Loam" Theme), Vanilla JavaScript
- **Geospatial & 3D**: Leaflet.js, Three.js
- **Machine Learning**: ONNX Runtime Web (`convnext_tiny_plantdisease.onnx`)
- **AI LLM**: Groq API (LLaMA 3)
- **Database & Sync**: Firebase Realtime Database
- **Data Visualizations**: Chart.js

---

## 📁 Directory Structure

```
FasalAI/
├── assets/
│   └── models/
│       ├── class_names.json                # Plant disease diagnostic labels
│       └── convnext_tiny_plantdisease.onnx # On-device ONNX model for plant diagnosis
├── images/
│   ├── crops/                              # SVG icons for crop representations
│   ├── logo.png                            # Primary dark logo
│   └── logo-light.png                      # Light theme logo
├── .gitignore
├── index.html                              # Core platform entry point
└── README.md                               # Project documentation
```

---

## 🚀 Getting Started

Simply serve `index.html` using any standard HTTP server (e.g. VS Code Live Server or `npx serve`).

```bash
# Clone the repository
git clone https://github.com/hyper10203/FasalAI.git

# Navigate to the project directory
cd FasalAI

# Serve the static website
npx serve .
```

Open `http://localhost:3000` in your browser.

---

## 👥 Developers

Developed by:
- **Subham Paul Choudhury** ([@hyper10203](https://github.com/hyper10203))
- **Vansh Prajapati**
- **Aarav Narula**
