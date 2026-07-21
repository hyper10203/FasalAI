export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    let imageBuffer;
    if (typeof req.body === 'string') {
      const b64Data = req.body.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else if (req.body && req.body.image) {
      const b64Data = req.body.image.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(b64Data, 'base64');
    } else {
      imageBuffer = req.body;
    }

    // HuggingFace Plant Disease Identification Model
    const hfUrl = 'https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification';
    const hfHeaders = { 'Content-Type': 'application/octet-stream' };
    if (process.env.HF_TOKEN) {
      hfHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
    }

    const hfResponse = await fetch(hfUrl, {
      method: 'POST',
      headers: hfHeaders,
      body: imageBuffer
    }).catch(() => null);

    if (hfResponse && hfResponse.ok) {
      const data = await hfResponse.json();
      if (Array.isArray(data) && data.length && data[0].label) {
        return res.status(200).json({
          predictions: data.map(d => ({ class: d.label, confidence: d.score || 0 })),
          source: 'hf'
        });
      }
    }

    // Roboflow Direct Model (if key available)
    if (process.env.ROBOFLOW_API_KEY) {
      const roboflowUrl = `https://serverless.roboflow.com/soilscope/4?api_key=${process.env.ROBOFLOW_API_KEY}`;
      const rfResponse = await fetch(roboflowUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: imageBuffer.toString('base64')
      }).catch(() => null);

      if (rfResponse && rfResponse.ok) {
        const rfData = await rfResponse.json();
        return res.status(200).json(rfData);
      }
    }

    return res.status(200).json({ status: 'fallback' });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
