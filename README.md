# 🌱 SoilScope — Agricultural Intelligence Platform

**SoilScope** is a comprehensive full-stack agricultural intelligence platform designed to empower farmers with hyper-local telemetry, multi-tier AI disease & pest diagnosis, real-time satellite data, MSP market price tracking, and multilingual voice assistance.

---

## ✨ Features

- 🛰️ **Real-Time Satellite Data**: Live NDVI vegetation tracking, surface soil moisture, and meteorological telemetry powered by NASA Power and Open-Meteo APIs.
- 🧪 **Smart Crop Recommendations**: Algorithmic scoring engine ranking optimal crops based on soil composition, climate, and seasonal suitability.
- 🔬 **AI Pest & Disease Detection (Pest Patrol)**: Multi-tier vision pipeline running custom Roboflow workflows with Hugging Face open-source model database failover.
- 🤖 **CropLens AI Assistant**: 24/7 context-aware voice agronomist powered by Groq LLaMA 3.1 supporting English, Hindi, and Punjabi with speech recognition.
- 📊 **Market Prices & Mandi Trends**: Real-time Agmarknet APMC Mandi rates compared directly against Government Minimum Support Prices (MSP).
- 🌐 **Global Multilingual Support**: 243+ languages supported, covering all 22 official scheduled Indian languages.
- 🌌 **Interactive 3D Visualizations**: 3D farm field models rendered using Three.js and interactive GIS maps using Leaflet.

---

## 🛠️ Technology Stack

- **Backend**: Node.js, Express.js (High-performance static & API server)
- **Frontend**: HTML5, Vanilla CSS ("Sunlit Loam" Light & Dark Theme), Vanilla JavaScript
- **Geospatial & 3D**: Leaflet.js, Three.js
- **Machine Learning & Vision**: ONNX Runtime Web (`convnext_tiny_plantdisease.onnx`), Roboflow Vision API, Hugging Face Inference API
- **AI LLM**: Groq API (LLaMA 3.1)
- **Database & Sync**: Firebase Realtime Database
- **Data Visualizations**: Chart.js

---

## 📁 Directory Structure

```
SoilScope/
├── assets/
│   └── models/
│       ├── class_names.json                # Plant disease diagnostic labels
│       └── convnext_tiny_plantdisease.onnx # 114MB ONNX model for on-device/backend plant diagnosis
├── images/
│   ├── crops/                              # SVG icons for crop representations
│   ├── logo.png                            # Primary dark logo
│   └── logo-light.png                      # Light theme logo
├── api/
│   ├── chat.js                             # Chat API handler
│   └── pdx.js                              # Plant diagnosis API handler
├── Procfile                                # Process definition for deployment
├── railway.json                            # Railway configuration
├── package.json                            # Node.js dependencies & scripts
├── server.js                               # Production Express web server
├── index.html                              # Core platform frontend
└── README.md                               # Project documentation
```

---

## 🚀 Deployment on Railway

### 1-Click / Git Deployment via Railway Dashboard
1. Go to [railway.com](https://railway.com) and create a **New Project**.
2. Select **Deploy from GitHub repo** and choose `hyper10203/FasalAI`.
3. Add any optional Environment Variables (e.g. `GROQ_API_KEY`, `ROBOFLOW_API_KEY`, `HF_TOKEN`).
4. Click **Deploy**. Railway will automatically detect `package.json` / `Procfile`, build the application, and assign a live production domain (e.g. `https://soilscope-production.up.railway.app`).

### Local Development
```bash
# Clone the repository
git clone https://github.com/hyper10203/FasalAI.git

# Navigate to the project directory
cd FasalAI

# Install dependencies
npm install

# Start the server
npm start
```
Open `http://localhost:3000` in your browser.

---

## 👥 Developers

Developed by:
- **Subham Paul Choudhury** — Delhi Public School, Bopal ([@hyper10203](https://github.com/hyper10203))
- **Aarav Narula** — Manav Mangal SMART World, Zirakpur
- **Vansh Prajapati**
