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

    // Roboflow Direct Model
    const roboflowKey = process.env.ROBOFLOW_API_KEY || 'XCb25NxLnpNfA24YIaNo';
    const roboflowUrl = `https://serverless.roboflow.com/soilscope/4?api_key=${roboflowKey}`;
    const rfResponse = await fetch(roboflowUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: imageBuffer.toString('base64'),
      signal: AbortSignal.timeout(20000)
    }).catch(() => null);

    if (rfResponse && rfResponse.ok) {
      const rfData = await rfResponse.json();
      return res.status(200).json(rfData);
    }

    return res.status(200).json({ status: 'invalid', predictions: [] });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
