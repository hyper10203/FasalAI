const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.raw({ type: 'application/octet-stream', limit: '50mb' }));

// MIME types & Static File Serving
express.static.mime.define({ 'application/octet-stream': ['onnx'] });

app.use(express.static(path.join(__dirname), {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.onnx')) {
      res.setHeader('Content-Type', 'application/octet-stream');
      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
    }
  }
}));

// Health Check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'FasalAI Backend', timestamp: new Date().toISOString() });
});

// ── 1. CHAT API (Groq LLaMA 3.1) ──
app.post('/api/chat', async (req, res) => {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return res.status(400).json({ error: 'GROQ_API_KEY not configured on server' });
  }

  try {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(req.body)
    });

    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('Chat API Error:', error);
    return res.status(500).json({ error: error.message });
  }
});

// ── 2. PLANT DISEASE / PEST VISION API (Hugging Face + Roboflow Fallback) ──
app.post('/api/pdx', async (req, res) => {
  try {
    let imageBuffer;

    if (Buffer.isBuffer(req.body)) {
      imageBuffer = req.body;
    } else if (typeof req.body === 'string') {
      const b64Data = req.body.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else if (req.body && req.body.image) {
      const b64Data = req.body.image.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else {
      imageBuffer = Buffer.from(JSON.stringify(req.body));
    }

    // 1. Primary: HuggingFace Plant Disease Model
    const hfUrl = 'https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification';
    const hfHeaders = { 'Content-Type': 'application/octet-stream' };
    if (process.env.HF_TOKEN) {
      hfHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
    }

    try {
      const hfResponse = await fetch(hfUrl, {
        method: 'POST',
        headers: hfHeaders,
        body: imageBuffer
      });

      if (hfResponse.ok) {
        const data = await hfResponse.json();
        if (Array.isArray(data) && data.length && data[0].label) {
          return res.status(200).json({
            predictions: data.map(d => ({ class: d.label, confidence: d.score || 0 })),
            source: 'hf'
          });
        }
      }
    } catch (e) {
      console.warn('HF inference failed:', e.message);
    }

    // 2. Secondary: Roboflow Direct Model (if key set)
    if (process.env.ROBOFLOW_API_KEY) {
      try {
        const roboflowUrl = `https://serverless.roboflow.com/soilscope/4?api_key=${process.env.ROBOFLOW_API_KEY}`;
        const rfResponse = await fetch(roboflowUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: imageBuffer.toString('base64')
        });

        if (rfResponse.ok) {
          const rfData = await rfResponse.json();
          return res.status(200).json(rfData);
        }
      } catch (e) {
        console.warn('Roboflow inference failed:', e.message);
      }
    }

    return res.status(200).json({ status: 'fallback', message: 'Use client heuristics' });
  } catch (error) {
    console.error('PDX API Error:', error);
    return res.status(500).json({ error: error.message });
  }
});

// Serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start Server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🌾 FasalAI server running on port ${PORT}`);
  console.log(`📍 URL: http://localhost:${PORT}`);
});
