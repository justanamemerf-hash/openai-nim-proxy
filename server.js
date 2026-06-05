const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({ origin: '*', methods: ['GET', 'POST', 'OPTIONS'], allowedHeaders: ['Content-Type', 'Authorization'] }));
app.use(express.json());

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v4-flash',
  'claude-3-opus': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro'
};

// ── catch-all GET for any probe/info request ──────────────────────────────────
app.get(['/', '/v1', '/health', '/v1/health'], (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy' });
});

// ── models list ───────────────────────────────────────────────────────────────
app.get(['/v1/models', '/models'], (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
    }))
  });
});

// ── chat completions ──────────────────────────────────────────────────────────
app.post(['/v1/chat/completions', '/chat/completions'], async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      const m = (model || '').toLowerCase();
      if (m.includes('gpt-4') || m.includes('405b'))      nimModel = 'deepseek-ai/deepseek-v4-pro';
      else if (m.includes('claude') || m.includes('70b')) nimModel = 'deepseek-ai/deepseek-v4-flash';
      else                                                 nimModel = 'deepseek-ai/deepseek-v4-flash';
    }

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature || 1.1,
      max_tokens: max_tokens || 50000,
      stream: stream || false,
      ...(ENABLE_THINKING_MODE && { extra_body: { chat_template_kwargs: { thinking: true } } })
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buf = '', reasoningOpen = false;
      response.data.on('data', chunk => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) { res.write(line + '\n'); return; }
          try {
            const d = JSON.parse(line.slice(6));
            const delta = d.choices?.[0]?.delta;
            if (delta) {
              const rc = delta.reasoning_content, c = delta.content;
              if (SHOW_REASONING) {
                let out = '';
                if (rc && !reasoningOpen) { out = '<think>\n' + rc; reasoningOpen = true; }
                else if (rc) out = rc;
                if (c && reasoningOpen) { out += '</think>\n\n' + c; reasoningOpen = false; }
                else if (c) out += c;
                delta.content = out;
              } else {
                delta.content = c || '';
              }
              delete delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(d)}\n\n`);
          } catch { res.write(line + '\n'); }
        });
      });
      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());
    } else {
      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(ch => {
          let content = ch.message?.content || '';
          if (SHOW_REASONING && ch.message?.reasoning_content)
            content = '<think>\n' + ch.message.reasoning_content + '\n</think>\n\n' + content;
          return { index: ch.index, message: { role: ch.message.role, content }, finish_reason: ch.finish_reason };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }
  } catch (err) {
    res.status(err.response?.status || 500).json({
      error: { message: err.message || 'Internal server error', type: 'invalid_request_error', code: err.response?.status || 500 }
    });
  }
});

// ── fallback ──────────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => console.log(`Proxy running on port ${PORT}`));
