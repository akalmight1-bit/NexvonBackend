# Self-host backup

Run models yourself when cloud APIs are unavailable or you want private inference.

## Ollama (easiest)

```bash
cd hosting/ollama
docker compose up -d

# pull a small chat model
docker exec -it nexvon-ollama ollama pull llama3.2:3b

# optional: coder + vision used by orchestrator.py
docker exec -it nexvon-ollama ollama pull qwen2.5-coder:1.5b
docker exec -it nexvon-ollama ollama pull qwen2.5vl:3b
```

Then in project `.env`:

```env
DEFAULT_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
```

## vLLM / TGI / llama.cpp (later)

Point the generic OpenAI-compatible provider at your server:

```env
LOCAL_BASE_URL=http://127.0.0.1:8001/v1
LOCAL_MODEL=your-model-id
DEFAULT_PROVIDER=local
```

Any server that implements `POST /v1/chat/completions` with streaming works.
