const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

let ort = null;
let Jimp = null;
let onnxSession = null;
let classNames = [];

try {
  ort = require('onnxruntime-node');
  const jimpPkg = require('jimp');
  Jimp = jimpPkg.Jimp || jimpPkg;
  const modelPath = path.join(__dirname, 'assets/models/convnext_tiny_plantdisease.onnx');
  const classesPath = path.join(__dirname, 'assets/models/class_names.json');
  
  if (fs.existsSync(classesPath)) {
    classNames = JSON.parse(fs.readFileSync(classesPath, 'utf8'));
    console.log(`Loaded ${classNames.length} plant disease / pest classes.`);
  }
  
  if (fs.existsSync(modelPath) && ort) {
    ort.InferenceSession.create(modelPath).then(session => {
      onnxSession = session;
      console.log('✅ ConvNeXt Tiny ONNX Model loaded on backend server!');
    }).catch(err => {
      console.warn('⚠️ ONNX session creation failed on startup:', err.message);
    });
  }
} catch (e) {
  console.warn('⚠️ ONNX backend init skipped:', e.message);
}

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
  res.json({ 
    status: 'ok', 
    service: 'FasalAI Backend', 
    onnxLoaded: !!onnxSession,
    totalClasses: classNames.length,
    timestamp: new Date().toISOString() 
  });
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

// ── 2. HIGH ACCURACY PLANT DISEASE & PEST VISION API ──
const PEST_KEYWORDS = ['aphid', 'worm', 'weevil', 'midge', 'mealybug', 'mite', 'moth', 'hispa', 'dead_heart', 'bug', 'fly', 'borer', 'thrip', 'hopper'];

app.post('/api/pdx', async (req, res) => {
  try {
    let imageBuffer;
    let mode = 'both'; // 'leaf' | 'pest' | 'both'
    let crop = '';

    if (Buffer.isBuffer(req.body)) {
      imageBuffer = req.body;
    } else if (typeof req.body === 'string') {
      const b64Data = req.body.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else if (req.body && req.body.image) {
      mode = req.body.mode || 'both';
      crop = req.body.crop || '';
      const b64Data = req.body.image.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else {
      imageBuffer = Buffer.from(JSON.stringify(req.body));
    }

    // 1. Primary: ConvNeXt Tiny ONNX inference if session is active
    if (onnxSession && Jimp && classNames.length > 0) {
      try {
        const image = await (Jimp.read ? Jimp.read(imageBuffer) : new Jimp(imageBuffer));
        
        // PyTorch standard ImageNet transform: Resize(256) maintaining aspect ratio, then CenterCrop(224)
        const minDim = Math.min(image.bitmap.width, image.bitmap.height);
        const scale = 256 / minDim;
        const newW = Math.round(image.bitmap.width * scale);
        const newH = Math.round(image.bitmap.height * scale);
        
        if (typeof image.resize === 'function') {
          try {
            image.resize({ w: newW, h: newH });
          } catch(e) {
            image.resize(newW, newH);
          }
        }
        
        const startX = Math.max(0, Math.round((newW - 224) / 2));
        const startY = Math.max(0, Math.round((newH - 224) / 2));
        
        if (typeof image.crop === 'function') {
          try {
            image.crop({ x: startX, y: startY, w: 224, h: 224 });
          } catch(e) {
            image.crop(startX, startY, 224, 224);
          }
        }

        const float32Data = new Float32Array(3 * 224 * 224);
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        image.scan(0, 0, 224, 224, function (x, y, idx) {
          const r = this.bitmap.data[idx + 0] / 255.0;
          const g = this.bitmap.data[idx + 1] / 255.0;
          const b = this.bitmap.data[idx + 2] / 255.0;
          const pixelIdx = y * 224 + x;

          float32Data[pixelIdx] = (r - mean[0]) / std[0];
          float32Data[pixelIdx + 224 * 224] = (g - mean[1]) / std[1];
          float32Data[pixelIdx + 2 * 224 * 224] = (b - mean[2]) / std[2];
        });

        const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
        const results = await onnxSession.run({ [onnxSession.inputNames[0]]: inputTensor });
        const logits = results[onnxSession.outputNames[0]].data;

        // Softmax
        let maxLogit = -Infinity;
        for (let i = 0; i < logits.length; i++) {
          if (logits[i] > maxLogit) maxLogit = logits[i];
        }
        let sumExp = 0;
        const exps = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
          exps[i] = Math.exp(logits[i] - maxLogit);
          sumExp += exps[i];
        }
        const probs = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
          probs[i] = exps[i] / sumExp;
        }

        let scored = classNames.map((name, i) => {
          const nameLower = name.toLowerCase();
          const isPest = PEST_KEYWORDS.some(k => nameLower.includes(k));
          const isHealthy = nameLower.includes('healthy') || nameLower.includes('normal');
          return {
            label: name,
            score: probs[i] || 0,
            isPest,
            isHealthy
          };
        });

        // Filter based on user target mode: 'leaf' vs 'pest' vs 'both'
        if (mode === 'leaf') {
          // In leaf mode, prioritize leaf diseases and healthy foliage
          scored = scored.filter(s => !s.isPest || s.score > 0.6);
        } else if (mode === 'pest') {
          // In pest mode, prioritize insect pests
          scored = scored.filter(s => s.isPest || s.score > 0.6);
        }

        // Filter based on crop if specified
        if (crop && crop.trim()) {
          const cNorm = crop.toLowerCase().replace(/[^a-z]/g, '');
          const cropMatches = scored.filter(s => s.label.toLowerCase().replace(/[^a-z]/g, '').includes(cNorm));
          if (cropMatches.length > 0) scored = cropMatches;
        }

        scored.sort((a, b) => b.score - a.score);

        return res.status(200).json({
          topK: scored.slice(0, 6).map(s => ({ label: s.label, score: s.score })),
          source: 'ai-local',
          model: 'ConvNeXt Tiny PlantDisease ONNX'
        });
      } catch (onnxErr) {
        console.warn('Backend ONNX execution error:', onnxErr.message);
      }
    }

    // 2. Secondary: Roboflow Direct Model (if key set and pest mode)
    if (process.env.ROBOFLOW_API_KEY && (mode === 'pest' || mode === 'both')) {
      try {
        const roboflowUrl = `https://serverless.roboflow.com/soilscope/4?api_key=${process.env.ROBOFLOW_API_KEY}`;
        const rfResponse = await fetch(roboflowUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: imageBuffer.toString('base64')
        });

        if (rfResponse.ok) {
          const rfData = await rfResponse.json();
          const preds = (rfData.predictions || []).slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
          if (preds.length && preds[0].class) {
            return res.status(200).json({
              topK: preds.map(p => ({ label: p.class, score: p.confidence || 0 })),
              source: 'pest-patrol'
            });
          }
        }
      } catch (e) {
        console.warn('Roboflow inference failed:', e.message);
      }
    }

    // 3. Fallback: HuggingFace Plant Disease Model
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
            topK: data.map(d => ({ label: d.label, score: d.score || 0 })),
            source: 'ai-hf'
          });
        }
      }
    } catch (e) {
      console.warn('HF inference failed:', e.message);
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
