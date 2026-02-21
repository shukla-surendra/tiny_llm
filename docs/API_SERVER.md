# FastAPI Serving Guide

## 1) Start server

```bash
.venv/bin/uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

## 2) Health check

```bash
curl http://127.0.0.1:8000/health
```

## 3) Generate text

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "System: You are a helpful coding assistant for docker workflows.\nUser: How do I debug container startup failures?\nAssistant:",
    "max_new_tokens": 80,
    "do_sample": true,
    "temperature": 0.9,
    "top_k": 40
  }'
```

## 4) Python sample client

```bash
.venv/bin/python examples/api_client.py
```
