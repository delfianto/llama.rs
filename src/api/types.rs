use serde::{Deserialize, Serialize};

// ─── OpenAI Types ────────────────────────────────────────────────────────────

/// Minimal wrapper to inspect the `stream` field without fully deserializing.
#[derive(Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub stream: bool,
    /// Everything else passed through as-is.
    #[serde(flatten)]
    pub rest: serde_json::Value,
}

/// OpenAI-compatible model list response.
#[derive(Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

/// Single model entry in the list.
#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

// ─── Ollama Types ────────────────────────────────────────────────────────────

fn default_true() -> bool {
    true
}

/// Ollama `/api/chat` request.
#[derive(Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama `/api/generate` request.
#[derive(Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

/// Ollama `/api/show` request.
#[derive(Deserialize)]
pub struct OllamaShowRequest {
    /// ollama CLI sends `name`, but API docs say `model` — accept both.
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub name: Option<String>,
}

impl OllamaShowRequest {
    /// Get the model identifier, preferring `name` over `model` (ollama CLI uses `name`).
    pub fn model_name(&self) -> &str {
        self.name
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(&self.model)
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub num_predict: Option<i32>,
    pub top_k: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub stop: Option<Vec<String>>,
}

/// Ollama `/api/chat` response (non-streaming and final streaming chunk).
#[derive(Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

/// Ollama `/api/generate` response (non-streaming).
#[derive(Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

/// Ollama `/api/tags` response.
#[derive(Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: serde_json::Value,
}

/// Ollama `/api/show` response.
#[derive(Serialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub model_info: serde_json::Value,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub details: serde_json::Value,
}

// ─── Translation Functions ───────────────────────────────────────────────────

/// Convert an Ollama chat request to an OpenAI-compatible JSON body.
pub fn ollama_chat_to_openai(req: &OllamaChatRequest, stream: bool) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": req.model,
        "messages": req.messages.iter().map(|m| {
            serde_json::json!({"role": &m.role, "content": &m.content})
        }).collect::<Vec<_>>(),
        "stream": stream,
    });

    if let Some(ref opts) = req.options {
        apply_options(&mut body, opts);
    }

    body
}

/// Convert an Ollama generate request to an OpenAI chat completion JSON body.
pub fn ollama_generate_to_openai(req: &OllamaGenerateRequest, stream: bool) -> serde_json::Value {
    let mut messages = Vec::new();
    if let Some(ref sys) = req.system {
        messages.push(serde_json::json!({"role": "system", "content": sys}));
    }
    messages.push(serde_json::json!({"role": "user", "content": &req.prompt}));

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": stream,
    });

    if let Some(ref opts) = req.options {
        apply_options(&mut body, opts);
    }

    body
}

fn apply_options(body: &mut serde_json::Value, opts: &OllamaOptions) {
    if let Some(t) = opts.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = opts.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(n) = opts.num_predict {
        body["max_tokens"] = serde_json::json!(n);
    }
    if let Some(k) = opts.top_k {
        body["top_k"] = serde_json::json!(k);
    }
    if let Some(rp) = opts.repeat_penalty {
        body["repeat_penalty"] = serde_json::json!(rp);
    }
    if let Some(s) = opts.seed {
        body["seed"] = serde_json::json!(s);
    }
    if let Some(ref stop) = opts.stop {
        body["stop"] = serde_json::json!(stop);
    }
}

/// Convert an OpenAI chat completion response to an Ollama chat response.
pub fn openai_to_ollama_chat(resp: &serde_json::Value, model: &str) -> OllamaChatResponse {
    let choice = &resp["choices"][0];
    let message = &choice["message"];

    // Handle both standard content and reasoning model content
    let content = message["content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .or_else(|| message["reasoning_content"].as_str())
        .unwrap_or("");

    OllamaChatResponse {
        model: model.to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        message: OllamaMessage {
            role: message["role"].as_str().unwrap_or("assistant").to_string(),
            content: content.to_string(),
        },
        done: true,
        done_reason: choice["finish_reason"].as_str().map(ToString::to_string),
        prompt_eval_count: resp["usage"]["prompt_tokens"].as_u64().map(|n| n as u32),
        eval_count: resp["usage"]["completion_tokens"]
            .as_u64()
            .map(|n| n as u32),
        total_duration: None,
    }
}

/// Convert an OpenAI chat completion response to an Ollama generate response.
pub fn openai_to_ollama_generate(resp: &serde_json::Value, model: &str) -> OllamaGenerateResponse {
    let choice = &resp["choices"][0];
    let content = choice["message"]["content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .or_else(|| choice["message"]["reasoning_content"].as_str())
        .unwrap_or("");

    OllamaGenerateResponse {
        model: model.to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        response: content.to_string(),
        done: true,
        done_reason: choice["finish_reason"].as_str().map(ToString::to_string),
        prompt_eval_count: resp["usage"]["prompt_tokens"].as_u64().map(|n| n as u32),
        eval_count: resp["usage"]["completion_tokens"]
            .as_u64()
            .map(|n| n as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_ollama_to_openai_chat_translation() {
        let req: OllamaChatRequest = serde_json::from_value(json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        }))
        .unwrap();

        let body = ollama_chat_to_openai(&req, false);
        assert_eq!(body["model"], "llama3");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hello");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_ollama_options_mapping() {
        let req: OllamaChatRequest = serde_json::from_value(json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 100
            }
        }))
        .unwrap();

        let body = ollama_chat_to_openai(&req, false);
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.01, "temperature: {temp}");
        let top_p = body["top_p"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 0.01, "top_p: {top_p}");
        assert_eq!(body["max_tokens"], 100);
    }

    #[test]
    fn test_ollama_generate_to_openai_translation() {
        let req: OllamaGenerateRequest = serde_json::from_value(json!({
            "model": "llama3",
            "prompt": "Why is the sky blue?",
            "system": "You are a scientist.",
            "stream": false
        }))
        .unwrap();

        let body = ollama_generate_to_openai(&req, false);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "You are a scientist.");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "Why is the sky blue?");
    }

    #[test]
    fn test_openai_to_ollama_response_translation() {
        let openai_resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        });

        let ollama = openai_to_ollama_chat(&openai_resp, "llama3");
        assert_eq!(ollama.model, "llama3");
        assert_eq!(ollama.message.role, "assistant");
        assert_eq!(ollama.message.content, "Hello!");
        assert!(ollama.done);
        assert_eq!(ollama.done_reason.as_deref(), Some("stop"));
        assert_eq!(ollama.prompt_eval_count, Some(5));
        assert_eq!(ollama.eval_count, Some(3));
    }

    #[test]
    fn test_openai_to_ollama_generate_translation() {
        let openai_resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Because of Rayleigh scattering."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });

        let ollama = openai_to_ollama_generate(&openai_resp, "llama3");
        assert_eq!(ollama.response, "Because of Rayleigh scattering.");
        assert!(ollama.done);
    }

    #[test]
    fn test_ollama_chat_stream_defaults_true() {
        let req: OllamaChatRequest = serde_json::from_value(json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .unwrap();
        assert!(req.stream);
    }
}
