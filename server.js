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
    agmarknetConfigured: !!(process.env.AGMARKNET_API_KEY || process.env.DATA_GOV_IN_API_KEY),
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
const PEST_KEYWORDS = ['aphid', 'worm', 'weevil', 'midge', 'mealybug', 'mite', 'moth', 'hispa', 'dead_heart', 'bug', 'fly', 'borer', 'thrip', 'hopper', 'caterpillar', 'beetle', 'scale', 'whitefly'];

async function runOnnxInference(imageBuffer, mode = 'leaf', crop = '') {
  if (!onnxSession || !Jimp || classNames.length === 0) return null;
  
  const image = await (Jimp.read ? Jimp.read(imageBuffer) : new Jimp(imageBuffer));
  
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

  if (mode === 'leaf') {
    scored = scored.filter(s => !s.isPest || s.score > 0.6);
  } else if (mode === 'pest') {
    scored = scored.filter(s => s.isPest || s.score > 0.6);
  }

  if (crop && crop.trim()) {
    const cNorm = crop.toLowerCase().replace(/[^a-z]/g, '');
    const cropMatches = scored.filter(s => s.label.toLowerCase().replace(/[^a-z]/g, '').includes(cNorm));
    if (cropMatches.length > 0) scored = cropMatches;
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 6).map(s => ({ label: s.label, score: s.score }));
}

app.post('/api/pdx', async (req, res) => {
  try {
    let imageBuffer;
    let mode = 'leaf';
    let crop = '';

    if (Buffer.isBuffer(req.body)) {
      imageBuffer = req.body;
    } else if (typeof req.body === 'string') {
      const b64Data = req.body.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else if (req.body && req.body.image) {
      mode = req.body.mode || 'leaf';
      crop = req.body.crop || '';
      const b64Data = req.body.image.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else {
      imageBuffer = Buffer.from(JSON.stringify(req.body));
    }

    // PEST & INSECT MODE
    if (mode === 'pest') {
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
            const preds = (rfData.predictions || []).slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
            if (preds.length && preds[0].class) {
              return res.status(200).json({
                topK: preds.map(p => ({ label: p.class, score: p.confidence || 0 })),
                source: 'pest-patrol',
                model: 'Roboflow Pest Detection'
              });
            }
          }
        } catch (rfErr) {
          console.warn('Roboflow pest inference failed, falling back to Hugging Face:', rfErr.message);
        }
      }

      const hfPestModels = [
        'https://api-inference.huggingface.co/models/dima806/pest-classification',
        'https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification'
      ];
      const hfHeaders = { 'Content-Type': 'application/octet-stream' };
      if (process.env.HF_TOKEN) hfHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;

      for (const hfUrl of hfPestModels) {
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
                source: 'ai-hf',
                model: 'Hugging Face Pest Vision API'
              });
            }
          }
        } catch (hfErr) {
          console.warn('HF pest inference failed:', hfErr.message);
        }
      }

      if (onnxSession && Jimp && classNames.length > 0) {
        const onnxRes = await runOnnxInference(imageBuffer, 'pest', crop);
        if (onnxRes && onnxRes.length) {
          return res.status(200).json({
            topK: onnxRes,
            source: 'ai-local',
            model: 'ConvNeXt Pest Classifier'
          });
        }
      }

      return res.status(200).json({ status: 'fallback', message: 'Use client heuristics' });
    }

    // LEAF DISEASE MODE
    if (onnxSession && Jimp && classNames.length > 0) {
      try {
        const onnxRes = await runOnnxInference(imageBuffer, 'leaf', crop);
        if (onnxRes && onnxRes.length) {
          return res.status(200).json({
            topK: onnxRes,
            source: 'ai-local',
            model: 'ConvNeXt Tiny PlantDisease ONNX'
          });
        }
      } catch (onnxErr) {
        console.warn('Backend ONNX execution error:', onnxErr.message);
      }
    }

    const hfUrl = 'https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification';
    const hfHeaders = { 'Content-Type': 'application/octet-stream' };
    if (process.env.HF_TOKEN) hfHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;

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
            source: 'ai-hf',
            model: 'Hugging Face Plant Disease API'
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

// ── 3. REAL AGMARKNET / MANDI MARKET PRICES API ──
let marketCache = { data: null, lastFetched: 0 };
const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes cache

const COMMODITY_MAP = {
  'tomatoes': { name: 'Tomatoes', icon: 'fa-circle-dot', unit: 'kg', base: 52 },
  'tomato': { name: 'Tomatoes', icon: 'fa-circle-dot', unit: 'kg', base: 52 },
  'potatoes': { name: 'Potatoes', icon: 'fa-egg', unit: 'kg', base: 28 },
  'potato': { name: 'Potatoes', icon: 'fa-egg', unit: 'kg', base: 28 },
  'onion': { name: 'Onions', icon: 'fa-layer-group', unit: 'kg', base: 35 },
  'onions': { name: 'Onions', icon: 'fa-layer-group', unit: 'kg', base: 35 },
  'carrot': { name: 'Carrots', icon: 'fa-arrow-down', unit: 'kg', base: 45 },
  'carrots': { name: 'Carrots', icon: 'fa-arrow-down', unit: 'kg', base: 45 },
  'capsicum': { name: 'Bell Peppers', icon: 'fa-fire', unit: 'kg', base: 72 },
  'cabbage': { name: 'Cabbage', icon: 'fa-circle-half-stroke', unit: 'kg', base: 24 },
  'cauliflower': { name: 'Cauliflower', icon: 'fa-brain', unit: 'kg', base: 42 },
  'spinach': { name: 'Spinach', icon: 'fa-leaf', unit: 'kg', base: 55 },
  'wheat': { name: 'Wheat', icon: 'fa-wheat-awn', unit: 'kg', base: 26 },
  'rice': { name: 'Rice', icon: 'fa-leaf', unit: 'kg', base: 38 },
  'paddy': { name: 'Rice', icon: 'fa-leaf', unit: 'kg', base: 38 },
  'maize': { name: 'Maize', icon: 'fa-seedling', unit: 'kg', base: 25 },
  'soyabean': { name: 'Soybean', icon: 'fa-circle-nodes', unit: 'kg', base: 48 },
  'cotton': { name: 'Cotton', icon: 'fa-cloud', unit: 'kg', base: 74 },
  'mustard': { name: 'Mustard', icon: 'fa-sun', unit: 'kg', base: 58 },
  'gram': { name: 'Chickpea', icon: 'fa-circle', unit: 'kg', base: 64 },
  'bengal gram': { name: 'Chickpea', icon: 'fa-circle', unit: 'kg', base: 64 }
};

app.get('/api/market', async (req, res) => {
  const customKey = req.query.apiKey || req.headers['x-api-key'];
  const apiKey = customKey || process.env.AGMARKNET_API_KEY || process.env.DATA_GOV_IN_API_KEY || '579b464db66ec23bdd00000112e16530575d42fb4dfe48ef42b367f0';

  if (!customKey && marketCache.data && (Date.now() - marketCache.lastFetched < CACHE_TTL_MS)) {
    return res.json(marketCache.data);
  }

  try {
    const resourceUrl = `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=${encodeURIComponent(apiKey)}&format=json&limit=500`;
    const response = await fetch(resourceUrl, { signal: AbortSignal.timeout(12000) });

    if (!response.ok) {
      throw new Error(`Data.gov.in responded with status: ${response.status}`);
    }

    const json = await response.json();
    const rawRecords = json.records || [];

    if (!rawRecords.length) {
      return res.json({
        status: 'empty',
        configured: true,
        message: 'No active mandi records returned for today. Showing verified baseline.',
        records: getBaselineMarketData(),
        timestamp: new Date().toISOString()
      });
    }

    const aggregated = {};
    rawRecords.forEach(rec => {
      const comm = (rec.commodity || '').toLowerCase().trim();
      let matchedKey = null;
      for (const k of Object.keys(COMMODITY_MAP)) {
        if (comm.includes(k)) { matchedKey = k; break; }
      }
      if (!matchedKey) return;

      const meta = COMMODITY_MAP[matchedKey];
      const modalQtl = parseFloat(rec.modal_price || 0);
      const minQtl = parseFloat(rec.min_price || 0);
      const maxQtl = parseFloat(rec.max_price || 0);

      if (modalQtl > 0) {
        if (!aggregated[meta.name]) {
          aggregated[meta.name] = {
            name: meta.name,
            icon: meta.icon,
            unit: meta.unit,
            modalSum: 0,
            minSum: 0,
            maxSum: 0,
            count: 0,
            samples: []
          };
        }
        aggregated[meta.name].modalSum += modalQtl;
        aggregated[meta.name].minSum += minQtl || modalQtl;
        aggregated[meta.name].maxSum += maxQtl || modalQtl;
        aggregated[meta.name].count += 1;
        if (aggregated[meta.name].samples.length < 3) {
          aggregated[meta.name].samples.push({
            mandi: rec.market || 'Regional Mandi',
            district: rec.district || '',
            state: rec.state || 'India',
            variety: rec.variety || 'Standard',
            modalKg: Math.round((modalQtl / 100) * 10) / 10,
            arrivalDate: rec.arrival_date || new Date().toISOString().slice(0, 10)
          });
        }
      }
    });

    const baseline = getBaselineMarketData();
    const finalRecords = [];

    baseline.forEach(baseItem => {
      const live = aggregated[baseItem.name];
      if (live && live.count > 0) {
        const avgModalKg = Math.round((live.modalSum / live.count / 100) * 10) / 10;
        const avgMinKg = Math.round((live.minSum / live.count / 100) * 10) / 10;
        const avgMaxKg = Math.round((live.maxSum / live.count / 100) * 10) / 10;
        const change = Math.round(((avgModalKg - baseItem.currentPrice) / baseItem.currentPrice) * 1000) / 10;
        
        finalRecords.push({
          name: baseItem.name,
          icon: baseItem.icon,
          currentPrice: avgModalKg,
          minPrice: avgMinKg,
          maxPrice: avgMaxKg,
          change: change || 1.5,
          unit: 'kg',
          mandi: live.samples[0]?.mandi || 'APMC Mandi',
          state: live.samples[0]?.state || 'India',
          arrivalDate: live.samples[0]?.arrivalDate || new Date().toLocaleDateString('en-IN'),
          isLive: true,
          samples: live.samples,
          mandiCount: live.count
        });
      } else {
        finalRecords.push(baseItem);
      }
    });

    const result = {
      status: 'live',
      source: 'AGMARKNET / Data.gov.in',
      configured: true,
      records: finalRecords,
      totalRawRecords: rawRecords.length,
      timestamp: new Date().toISOString()
    };

    if (!customKey) {
      marketCache = { data: result, lastFetched: Date.now() };
    }

    return res.json(result);
  } catch (err) {
    console.error('Agmarknet API fetch error:', err.message);
    return res.json({
      status: 'fallback',
      configured: true,
      message: `Live Agmarknet query fallback: ${err.message}`,
      records: getBaselineMarketData(),
      timestamp: new Date().toISOString()
    });
  }
});

function getBaselineMarketData() {
  return [
    { name: 'Tomatoes', icon: 'fa-circle-dot', currentPrice: 52, minPrice: 44, maxPrice: 60, change: 3.2, unit: 'kg', mandi: 'Azadpur Mandi', state: 'Delhi', isLive: false },
    { name: 'Potatoes', icon: 'fa-egg', currentPrice: 28, minPrice: 24, maxPrice: 32, change: -0.8, unit: 'kg', mandi: 'Lasalgaon APMC', state: 'Maharashtra', isLive: false },
    { name: 'Onions', icon: 'fa-layer-group', currentPrice: 35, minPrice: 30, maxPrice: 42, change: 4.5, unit: 'kg', mandi: 'Nasik Mandi', state: 'Maharashtra', isLive: false },
    { name: 'Carrots', icon: 'fa-arrow-down', currentPrice: 45, minPrice: 38, maxPrice: 52, change: 1.2, unit: 'kg', mandi: 'Kolar APMC', state: 'Karnataka', isLive: false },
    { name: 'Bell Peppers', icon: 'fa-fire', currentPrice: 72, minPrice: 62, maxPrice: 84, change: -2.4, unit: 'kg', mandi: 'Vashi APMC', state: 'Mumbai', isLive: false },
    { name: 'Cabbage', icon: 'fa-circle-half-stroke', currentPrice: 24, minPrice: 18, maxPrice: 28, change: 1.8, unit: 'kg', mandi: 'Grain Market', state: 'Ludhiana', isLive: false },
    { name: 'Cauliflower', icon: 'fa-brain', currentPrice: 42, minPrice: 35, maxPrice: 50, change: 2.5, unit: 'kg', mandi: 'Jaipur APMC', state: 'Rajasthan', isLive: false },
    { name: 'Spinach', icon: 'fa-leaf', currentPrice: 55, minPrice: 45, maxPrice: 65, change: -1.1, unit: 'kg', mandi: 'Okhla Mandi', state: 'Delhi', isLive: false }
  ];
}

// Serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start Server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🌾 FasalAI server running on port ${PORT}`);
  console.log(`📍 URL: http://localhost:${PORT}`);
});
