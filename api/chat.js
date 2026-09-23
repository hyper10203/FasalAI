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

  const GROQ_PRIMARY_MODELS = ['openai/gpt-oss-120b', 'qwen/qwen3.8-27b', 'openai/gpt-oss-20b', 'allam-2-7b'];
  const deprecatedModels = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'llama3-8b-8192', 'groq/compound-mini', 'groq/compound', 'mixtral-8x7b-32768', 'gemma2-9b-it'];

  let requestedModel = req.body && req.body.model;
  if (!requestedModel || deprecatedModels.includes(requestedModel) || !GROQ_PRIMARY_MODELS.includes(requestedModel)) {
    requestedModel = GROQ_PRIMARY_MODELS[0];
  }

  const maxTokens = Math.min(Math.max(req.body.max_tokens || 300, 100), 350);
  const modelsToTry = [requestedModel, ...GROQ_PRIMARY_MODELS.filter(m => m !== requestedModel)];
  let lastErrorData = null;
  let lastStatus = 500;

  for (const model of modelsToTry) {
    const payload = {
      ...req.body,
      model: model,
      max_tokens: maxTokens
    };

    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (response.ok) {
        return res.status(200).json(data);
      }

      lastStatus = response.status;
      lastErrorData = data;

      if (response.status !== 429 && data?.error?.code !== 'model_not_found' && response.status !== 404) {
        break;
      }
    } catch (err) {
      lastErrorData = { error: { message: err.message } };
    }
  }

  return res.status(lastStatus).json(lastErrorData || { error: { message: 'All Groq models failed' } });
}
