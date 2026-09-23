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

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return res.status(400).json({ error: 'GROQ_API_KEY not configured in environment variables' });
  }

  const GROQ_PRIMARY_MODELS = ['qwen/qwen3.8-27b', 'openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'allam-2-7b'];
  const deprecatedModels = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'llama3-8b-8192', 'groq/compound-mini', 'groq/compound', 'mixtral-8x7b-32768', 'gemma2-9b-it'];

  let modelToUse = req.body && req.body.model;
  if (!modelToUse || deprecatedModels.includes(modelToUse) || !GROQ_PRIMARY_MODELS.includes(modelToUse)) {
    modelToUse = GROQ_PRIMARY_MODELS[0];
  }

  const payload = {
    ...req.body,
    model: modelToUse,
    max_tokens: Math.max(req.body.max_tokens || 500, 500)
  };

  try {
    let response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(payload)
    });

    let data = await response.json();

    if (!response.ok && (data?.error?.code === 'model_not_found' || response.status === 404)) {
      payload.model = GROQ_PRIMARY_MODELS[1];
      response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(payload)
      });
      data = await response.json();
    }

    return res.status(response.status).json(data);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
