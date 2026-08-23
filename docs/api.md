# API and web UI

`llama serve` exposes llama.cpp's built-in web UI, an OpenAI-compatible surface,
and a practical subset of the Ollama API on one address. The default base URL is
`http://127.0.0.1:8080`.

The public proxy starts only after the underlying model is ready.

## Routes

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/` | Proxies llama.cpp's built-in web UI |
| `HEAD` | `/` | Connectivity check used by Ollama clients |
| `POST` | `/v1/chat/completions` | Raw OpenAI-compatible passthrough |
| `GET` | `/v1/models` | Reports the currently loaded model |
| `POST` | `/api/chat` | Ollama chat translation |
| `POST` | `/api/generate` | Ollama generation translation |
| `GET` | `/api/tags` | Ollama-compatible model list |
| `POST` | `/api/show` | Ollama-compatible loaded-model information |
| `GET` | `/api/version` | Ollama-compatible wrapper version |
| `GET` | `/health` | Wrapper health check |

Unmatched paths, including UI assets and llama.cpp-specific APIs, are forwarded to
the internal server unchanged.

## Built-in llama.cpp UI

Start a model, wait for `Model loaded!`, then open the root address:

```bash
llama serve --config profiles/model.yaml
```

```text
http://127.0.0.1:8080/
```

The UI is owned by the installed `llama-server` build, so its features and layout
track that engine version.

## OpenAI-compatible API

The chat request body and upstream response are relayed without schema
translation. Streaming and non-streaming requests are both supported, and
llama.cpp-specific request fields are preserved.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Say hello."}],
    "temperature": 0.7,
    "stream": false
  }'
```

Most OpenAI SDKs can use the service by changing their base URL. No API key is
validated, although a client library may require a placeholder value:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(response.choices[0].message.content)
```

The `model` request value does not switch models; each `llama serve` process owns
one loaded GGUF.

## Ollama-compatible API

The wrapper translates Ollama JSON or NDJSON to and from llama-server's OpenAI
chat endpoint. It is intentionally a subset, aimed at local chat frontends rather
than complete Ollama emulation.

```bash
curl http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Say hello."}],
    "stream": false
  }'
```

Supported per-request `options` include:

| Ollama option | Upstream field |
| --- | --- |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `top_k` | `top_k` |
| `num_predict` | `max_tokens` |
| `repeat_penalty` | `repeat_penalty` |
| `seed` | `seed` |
| `stop` | `stop` |

Streaming responses use newline-delimited JSON, as Ollama clients expect.
Unsupported Ollama lifecycle operations such as loading several models,
Modelfiles, create, copy, push, and embeddings are outside this project's scope.

## Frontend configuration

For a frontend running on the same machine, use:

```text
OpenAI base URL: http://127.0.0.1:8080/v1
Ollama base URL: http://127.0.0.1:8080
```

For a containerized frontend, `127.0.0.1` refers to that container, not the host.
Bind `llama` to an appropriate trusted interface and use the host or service name
reachable from the container network.

## Security

The proxy has permissive CORS and no authentication or rate limiting. Do not bind
it to an untrusted network or publish it directly to the internet. If remote
access is required, place it behind an authenticated reverse proxy and restrict
network access.
