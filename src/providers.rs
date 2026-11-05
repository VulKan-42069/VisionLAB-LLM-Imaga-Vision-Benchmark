//! Vision provider HTTP clients (OpenAI, Anthropic, OpenRouter, Ollama, custom).

use anyhow::{anyhow, Context, Result};
use reqwest::StatusCode;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::cost::normalize_per_m;
use crate::settings::{ModelEntry, ProviderConfig, ProviderKind, VisionStatus};
use crate::tasks::{system_prompt, user_prompt, TaskKind};

#[derive(Debug, Clone)]
pub struct RunRequest {
    pub model: String,
    pub task: TaskKind,
    pub image_jpeg: Vec<u8>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub focus: String,
}

#[derive(Debug, Clone)]
pub struct RunResponse {
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub hit_token_limit: bool,
}

pub struct ProviderClient {
    http: reqwest::Client,
}

impl ProviderClient {
    pub fn new(timeout: Duration) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("VisionLab/0.1")
            .build()?;
        Ok(Self { http })
    }

    pub async fn list_models(
        &self,
        provider: &ProviderConfig,
        api_key: &str,
    ) -> Result<Vec<ModelEntry>> {
        match provider.kind {
            ProviderKind::Anthropic => self.list_anthropic(provider, api_key).await,
            ProviderKind::OpenRouter => self.list_openrouter(provider, api_key).await,
            ProviderKind::Ollama => self.list_ollama(provider, api_key).await,
            ProviderKind::OpenAi | ProviderKind::Custom => self.list_openai_compat(provider, api_key).await,
        }
    }

    pub async fn run(
        &self,
        provider: &ProviderConfig,
        api_key: &str,
        req: &RunRequest,
        retries: u32,
    ) -> Result<RunResponse> {
        match provider.kind {
            ProviderKind::Anthropic => self.run_anthropic(provider, api_key, req, retries).await,
            ProviderKind::Ollama => self.run_ollama(provider, api_key, req, retries).await,
            ProviderKind::OpenAi | ProviderKind::OpenRouter | ProviderKind::Custom => {
                self.run_openai_compat(provider, api_key, req, retries).await
            }
        }
    }

    pub async fn test_connection(&self, provider: &ProviderConfig, api_key: &str) -> Result<String> {
        let models = self.list_models(provider, api_key).await?;
        if provider.kind == ProviderKind::Ollama && models.is_empty() {
            return Ok(
                "Connected to Ollama, but no models are pulled yet. Run ollama pull llama3.2-vision or another vision model.".into(),
            );
        }
        Ok(format!("Connected. {} models listed.", models.len()))
    }

    pub async fn probe_vision(
        &self,
        provider: &ProviderConfig,
        api_key: &str,
        model: &str,
        retries: u32,
    ) -> Result<bool> {
        let img = tiny_probe_jpeg();
        let req = RunRequest {
            model: model.to_string(),
            task: TaskKind::Classification,
            image_jpeg: img,
            temperature: 0.0,
            max_tokens: 128,
            focus: String::new(),
        };
        let resp = self.run(provider, api_key, &req, retries).await?;
        let lower = resp.text.to_ascii_lowercase();
        let bad = [
            "does not support image",
            "image inputs are not supported",
            "cannot process images",
            "vision is not",
            "no image",
            "didn't receive an image",
            "did not receive an image",
        ];
        Ok(!bad.iter().any(|s| lower.contains(s)))
    }

    async fn list_openai_compat(&self, provider: &ProviderConfig, api_key: &str) -> Result<Vec<ModelEntry>> {
        let url = format!("{}/models", provider.base_url.trim_end_matches('/'));
        let mut builder = self.http.get(&url).bearer_auth(api_key);
        builder = apply_headers(builder, &provider.extra_headers);
        if provider.kind == ProviderKind::OpenRouter {
            builder = builder
                .header("HTTP-Referer", "https://visionlab.local")
                .header("X-Title", "VisionLab");
        }
        let resp = builder.send().await.context("list models")?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_http(status, &body));
        }
        let v: Value = serde_json::from_str(&body).context("parse models")?;
        let arr = v.get("data").and_then(Value::as_array).cloned().unwrap_or_default();
        Ok(arr
            .iter()
            .filter_map(|m| {
                let id = m.get("id").and_then(Value::as_str)?.to_string();
                Some(ModelEntry {
                    vision: infer_openai_vision(&id, m),
                    name: id.clone(),
                    input_price_per_m: None,
                    output_price_per_m: None,
                    id,
                })
            })
            .collect())
    }

    async fn list_openrouter(&self, provider: &ProviderConfig, api_key: &str) -> Result<Vec<ModelEntry>> {
        let url = format!("{}/models", provider.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .bearer_auth(api_key)
            .header("HTTP-Referer", "https://visionlab.local")
            .header("X-Title", "VisionLab")
            .send()
            .await
            .context("list OpenRouter models")?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_http(status, &body));
        }
        let v: Value = serde_json::from_str(&body)?;
        let arr = v.get("data").and_then(Value::as_array).cloned().unwrap_or_default();
        Ok(arr
            .iter()
            .filter_map(|m| {
                let id = m.get("id").and_then(Value::as_str)?.to_string();
                let name = m
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(&id)
                    .to_string();
                let vision = infer_openrouter_vision(m);
                let (inp, out) = openrouter_prices(m);
                Some(ModelEntry {
                    id,
                    name,
                    vision,
                    input_price_per_m: inp,
                    output_price_per_m: out,
                })
            })
            .collect())
    }

    async fn list_anthropic(&self, provider: &ProviderConfig, api_key: &str) -> Result<Vec<ModelEntry>> {
        let url = format!("{}/v1/models", provider.base_url.trim_end_matches('/'));
        let mut builder = self
            .http
            .get(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01");
        builder = apply_headers(builder, &provider.extra_headers);
        let resp = builder.send().await.context("list Anthropic models")?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_http(status, &body));
        }
        let v: Value = serde_json::from_str(&body)?;
        let arr = v.get("data").and_then(Value::as_array).cloned().unwrap_or_default();
        Ok(arr
            .iter()
            .filter_map(|m| {
                let id = m.get("id").and_then(Value::as_str)?.to_string();
                Some(ModelEntry {
                    vision: VisionStatus::Confirmed,
                    name: id.clone(),
                    input_price_per_m: crate::cost::builtin_price(&id).map(|p| p.0),
                    output_price_per_m: crate::cost::builtin_price(&id).map(|p| p.1),
                    id,
                })
            })
            .collect())
    }

    async fn list_ollama(&self, provider: &ProviderConfig, api_key: &str) -> Result<Vec<ModelEntry>> {
        let root = ollama_root(&provider.base_url);
        let url = format!("{root}/api/tags");
        let mut builder = self.http.get(&url);
        if !api_key.is_empty() {
            builder = builder.bearer_auth(api_key);
        }
        builder = apply_headers(builder, &provider.extra_headers);
        let resp = builder.send().await.map_err(|e| ollama_connect_error(&e, &root))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_http(status, &body));
        }
        parse_ollama_tags(&body)
    }

    async fn run_ollama(
        &self,
        provider: &ProviderConfig,
        api_key: &str,
        req: &RunRequest,
        retries: u32,
    ) -> Result<RunResponse> {
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &req.image_jpeg);
        let root = ollama_root(&provider.base_url);
        let url = format!("{root}/api/chat");
        let body = json!({
            "model": req.model,
            "stream": false,
            "format": "json",
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt(req.task)},
                {
                    "role": "user",
                    "content": user_prompt(req.task, &req.focus),
                    "images": [b64]
                }
            ]
        });
        let started = Instant::now();
        let mut builder = self.http.post(&url).json(&body);
        if !api_key.is_empty() {
            builder = builder.bearer_auth(api_key);
        }
        builder = apply_headers(builder, &provider.extra_headers);
        let (status, text) = send_with_retry(&self.http, builder, retries).await.map_err(|e| {
            let msg = e.to_string();
            if msg.contains("Could not reach") || msg.contains("Network error") || msg.contains("timed out") {
                anyhow!(
                    "{msg}\n\nIf this is a first load, pull the model with ollama pull {} and raise the timeout in Settings > Run defaults.",
                    req.model
                )
            } else {
                e
            }
        })?;
        let latency_ms = started.elapsed().as_millis() as u64;
        if !status.is_success() {
            return Err(map_http(status, &text));
        }
        parse_ollama_chat(&text, latency_ms)
    }

    async fn run_openai_compat(
        &self,
        provider: &ProviderConfig,
        api_key: &str,
        req: &RunRequest,
        retries: u32,
    ) -> Result<RunResponse> {
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &req.image_jpeg);
        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));
        let mut body = json!({
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt(req.task)},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt(req.task, &req.focus)},
                    {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}}
                ]}
            ]
        });
        // Many OpenRouter / custom models reject json_object mode.
        if provider.kind == ProviderKind::OpenAi {
            body["response_format"] = json!({ "type": "json_object" });
        }
        let started = Instant::now();
        let mut builder = self.http.post(&url).bearer_auth(api_key).json(&body);
        builder = apply_headers(builder, &provider.extra_headers);
        if provider.kind == ProviderKind::OpenRouter {
            builder = builder
                .header("HTTP-Referer", "https://visionlab.local")
                .header("X-Title", "VisionLab");
        }
        let (status, text) = send_with_retry(&self.http, builder, retries).await?;
        let latency_ms = started.elapsed().as_millis() as u64;
        if !status.is_success() {
            return Err(map_http(status, &text));
        }
        let v: Value = serde_json::from_str(&text).context("parse completion")?;
        let content = v
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let input_tokens = v.pointer("/usage/prompt_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
        let output_tokens = v
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let finish = v
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("");
        if content.is_empty() {
            return Err(anyhow!("The provider returned an empty message."));
        }
        Ok(RunResponse {
            text: content,
            input_tokens,
            output_tokens,
            latency_ms,
            hit_token_limit: hit_output_limit(finish),
        })
    }

    async fn run_anthropic(
        &self,
        provider: &ProviderConfig,
        api_key: &str,
        req: &RunRequest,
        retries: u32,
    ) -> Result<RunResponse> {
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &req.image_jpeg);
        let url = format!("{}/v1/messages", provider.base_url.trim_end_matches('/'));
        let body = json!({
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": system_prompt(req.task),
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": user_prompt(req.task, &req.focus)}
                ]
            }]
        });
        let started = Instant::now();
        let mut builder = self
            .http
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body);
        builder = apply_headers(builder, &provider.extra_headers);
        let (status, text) = send_with_retry(&self.http, builder, retries).await?;
        let latency_ms = started.elapsed().as_millis() as u64;
        if !status.is_success() {
            return Err(map_http(status, &text));
        }
        let v: Value = serde_json::from_str(&text).context("parse Anthropic message")?;
        let content = v
            .get("content")
            .and_then(Value::as_array)
            .and_then(|a| a.iter().find(|b| b.get("type").and_then(Value::as_str) == Some("text")))
            .and_then(|b| b.get("text").and_then(Value::as_str))
            .unwrap_or("")
            .to_string();
        let input_tokens = v.pointer("/usage/input_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
        let output_tokens = v.pointer("/usage/output_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
        let stop = v.get("stop_reason").and_then(Value::as_str).unwrap_or("");
        if content.is_empty() {
            return Err(anyhow!("The provider returned an empty message."));
        }
        Ok(RunResponse {
            text: content,
            input_tokens,
            output_tokens,
            latency_ms,
            hit_token_limit: hit_output_limit(stop),
        })
    }
}

fn hit_output_limit(reason: &str) -> bool {
    matches!(
        reason.trim().to_ascii_lowercase().as_str(),
        "length" | "max_tokens" | "max_token"
    )
}

fn apply_headers(mut b: reqwest::RequestBuilder, extra: &HashMap<String, String>) -> reqwest::RequestBuilder {
    for (k, v) in extra {
        b = b.header(k, v);
    }
    b
}

async fn send_with_retry(
    _http: &reqwest::Client,
    builder: reqwest::RequestBuilder,
    retries: u32,
) -> Result<(StatusCode, String)> {
    let mut last_err = anyhow!("request failed");
    for attempt in 0..=retries {
        let req = builder.try_clone().ok_or_else(|| anyhow!("cannot retry this request"))?;
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                if status.as_u16() == 429 || status.is_server_error() {
                    last_err = map_http(status, &text);
                    if attempt < retries {
                        let wait = Duration::from_millis(400 * 2u64.pow(attempt));
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                    return Err(last_err);
                }
                return Ok((status, text));
            }
            Err(e) => {
                last_err = anyhow!(friendly_reqwest(&e));
                if attempt < retries {
                    let wait = Duration::from_millis(400 * 2u64.pow(attempt));
                    tokio::time::sleep(wait).await;
                    continue;
                }
            }
        }
    }
    Err(last_err)
}

fn friendly_reqwest(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "The request timed out. Check the network and the provider endpoint.".into()
    } else if e.is_connect() {
        "Could not reach the provider. Check the base URL and your connection.".into()
    } else {
        format!("Network error: {e}")
    }
}

pub fn map_http(status: StatusCode, body: &str) -> anyhow::Error {
    let snippet: String = body.chars().take(800).collect();
    let msg = match status.as_u16() {
        401 => "Invalid or expired API key. Update the key in Settings.".to_string(),
        403 => "The provider refused the request (permissions or region).".to_string(),
        404 => "Model or endpoint not found. Check the model id and base URL.".to_string(),
        413 => "The image was too large for this provider.".to_string(),
        429 => "Rate limited. Wait a moment, then retry.".to_string(),
        500..=599 => "The provider is having an outage. Try again shortly.".to_string(),
        _ => format!("Provider error HTTP {status}."),
    };
    if looks_like_non_vision(body) {
        anyhow!(
            "{msg}\n\nThis model does not appear to accept images. Pick a vision-capable model."
        )
    } else if snippet.is_empty() {
        anyhow!(msg)
    } else {
        anyhow!("{msg}\n\n{snippet}")
    }
}

pub fn looks_like_non_vision(body: &str) -> bool {
    let l = body.to_ascii_lowercase();
    l.contains("does not support image")
        || l.contains("image inputs are not supported")
        || l.contains("cannot process images")
        || l.contains("unsupported content type")
        || l.contains("invalid content") && l.contains("image")
}

fn infer_openai_vision(id: &str, meta: &Value) -> VisionStatus {
    if let Some(mods) = meta
        .pointer("/architecture/input_modalities")
        .or_else(|| meta.get("input_modalities"))
        .and_then(Value::as_array)
    {
        if mods.iter().any(|m| m.as_str() == Some("image")) {
            return VisionStatus::Confirmed;
        }
        return VisionStatus::Unlikely;
    }
    let id = id.to_ascii_lowercase();
    if id.contains("vision")
        || id.contains("gpt-4o")
        || id.contains("gpt-4.1")
        || id.contains("gpt-5")
        || id.contains("o1")
        || id.contains("o3")
        || id.contains("o4")
        || id.contains("claude")
        || id.contains("gemini")
        || id.contains("llama-4")
        || id.contains("pixtral")
        || id.contains("qwen") && id.contains("vl")
    {
        VisionStatus::Probable
    } else if id.contains("whisper") || id.contains("tts") || id.contains("embedding") {
        VisionStatus::Unlikely
    } else {
        VisionStatus::Unknown
    }
}

fn infer_openrouter_vision(meta: &Value) -> VisionStatus {
    if let Some(mods) = meta
        .pointer("/architecture/input_modalities")
        .and_then(Value::as_array)
    {
        if mods.iter().any(|m| m.as_str() == Some("image")) {
            return VisionStatus::Confirmed;
        }
        return VisionStatus::Unlikely;
    }
    let modality = meta
        .pointer("/architecture/modality")
        .and_then(Value::as_str)
        .unwrap_or("");
    if modality.contains("image") {
        VisionStatus::Confirmed
    } else if let Some(id) = meta.get("id").and_then(Value::as_str) {
        infer_openai_vision(id, meta)
    } else {
        VisionStatus::Unknown
    }
}

fn ollama_root(base: &str) -> String {
    let mut u = base.trim().trim_end_matches('/').to_string();
    for suffix in ["/v1", "/api"] {
        if u.len() > suffix.len() && u.to_ascii_lowercase().ends_with(suffix) {
            u.truncate(u.len() - suffix.len());
            u = u.trim_end_matches('/').to_string();
        }
    }
    u
}

fn ollama_connect_error(e: &reqwest::Error, root: &str) -> anyhow::Error {
    if e.is_connect() {
        anyhow!("Could not reach Ollama at {root}. Start it with ollama serve, then Test connection.")
    } else if e.is_timeout() {
        anyhow!("Ollama timed out at {root}. The model may still be loading. Raise the timeout in Settings > Run defaults.")
    } else {
        anyhow!(friendly_reqwest(e))
    }
}

fn parse_ollama_tags(body: &str) -> Result<Vec<ModelEntry>> {
    let v: Value = serde_json::from_str(body).context("parse Ollama model list")?;
    let arr = v.get("models").and_then(Value::as_array).cloned().unwrap_or_default();
    Ok(arr
        .iter()
        .filter_map(|m| {
            let id = m
                .get("name")
                .or_else(|| m.get("model"))
                .and_then(Value::as_str)?
                .to_string();
            Some(ModelEntry {
                vision: infer_ollama_vision(&id, m.get("details")),
                name: id.clone(),
                input_price_per_m: Some(0.0),
                output_price_per_m: Some(0.0),
                id,
            })
        })
        .collect())
}

fn parse_ollama_chat(text: &str, latency_ms: u64) -> Result<RunResponse> {
    let v: Value = serde_json::from_str(text).context("parse Ollama chat")?;
    let mut content = v
        .pointer("/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if content.is_empty() {
        if let Some(s) = v.pointer("/message/thinking").and_then(Value::as_str) {
            content = s.trim().to_string();
        }
    }
    let input_tokens = v
        .get("prompt_eval_count")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let output_tokens = v.get("eval_count").and_then(Value::as_u64).unwrap_or(0) as u32;
    let done = v
        .get("done_reason")
        .and_then(Value::as_str)
        .unwrap_or("");
    if content.is_empty() {
        return Err(anyhow!("Ollama returned an empty message."));
    }
    Ok(RunResponse {
        text: content,
        input_tokens,
        output_tokens,
        latency_ms,
        hit_token_limit: hit_output_limit(done),
    })
}

fn infer_ollama_vision(id: &str, details: Option<&Value>) -> VisionStatus {
    if let Some(d) = details {
        let families = d
            .get("families")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let family = d.get("family").and_then(Value::as_str).unwrap_or("");
        let hit = |s: &str| {
            let s = s.to_ascii_lowercase();
            s.contains("llava")
                || s.contains("mllama")
                || s.contains("vision")
                || s.contains("qwen2vl")
                || s.contains("qwen3vl")
                || s.contains("gemma3")
        };
        if hit(family) || families.iter().any(|f| f.as_str().is_some_and(hit)) {
            return VisionStatus::Probable;
        }
    }
    let idl = id.to_ascii_lowercase();
    if idl.contains("llava")
        || idl.contains("moondream")
        || idl.contains("minicpm")
        || idl.contains("pixtral")
        || idl.contains("gemma3")
        || idl.contains("gemma-3")
        || idl.contains("internvl")
        || idl.contains("vision")
        || (idl.contains("qwen") && idl.contains("vl"))
    {
        return VisionStatus::Probable;
    }
    infer_openai_vision(id, &Value::Null)
}

fn openrouter_prices(meta: &Value) -> (Option<f64>, Option<f64>) {
    let parse = |k: &str| {
        meta.pointer(&format!("/pricing/{k}"))
            .and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| v.as_f64().map(|f| f.to_string())))
            .and_then(|s| s.parse::<f64>().ok())
            .map(normalize_per_m)
    };
    (parse("prompt"), parse("completion"))
}

fn tiny_probe_jpeg() -> Vec<u8> {
    // Several providers reject images at or below 10px on a side.
    const S: u32 = 32;
    let mut img = image::RgbImage::new(S, S);
    for y in 0..S {
        for x in 0..S {
            let on = (x < S / 2) != (y < S / 2);
            img.put_pixel(
                x,
                y,
                image::Rgb(if on { [220, 40, 40] } else { [40, 90, 210] }),
            );
        }
    }
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, 80);
    let _ = encoder.encode(img.as_raw(), S, S, image::ExtendedColorType::Rgb8);
    buf
}

pub fn lookup_model<'a>(provider: &'a ProviderConfig, model_id: &str) -> Option<&'a ModelEntry> {
    provider
        .models
        .iter()
        .find(|m| m.id == model_id || m.name == model_id)
}

pub fn vision_status_for(provider: &ProviderConfig, model_id: &str) -> VisionStatus {
    if let Some(m) = lookup_model(provider, model_id) {
        return m.vision;
    }
    if provider.kind == ProviderKind::Ollama {
        return infer_ollama_vision(model_id, None);
    }
    infer_openai_vision(model_id, &Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_root_strips_compat_suffixes() {
        assert_eq!(ollama_root("http://localhost:11434"), "http://localhost:11434");
        assert_eq!(ollama_root("http://127.0.0.1:11434/"), "http://127.0.0.1:11434");
        assert_eq!(ollama_root("http://localhost:11434/v1"), "http://localhost:11434");
        assert_eq!(ollama_root("http://localhost:11434/api"), "http://localhost:11434");
    }

    #[test]
    fn ollama_vision_names() {
        assert!(infer_ollama_vision("llama3.2-vision:latest", None).is_ok());
        assert!(infer_ollama_vision("qwen2.5vl:7b", None).is_ok());
        assert!(infer_ollama_vision("llava:13b", None).is_ok());
        assert!(!infer_ollama_vision("llama3.2:latest", None).is_ok());
    }

    #[test]
    fn parse_ollama_tags_and_chat() {
        let tags = r#"{"models":[{"name":"llava:latest","details":{"family":"llava"}}]}"#;
        let models = parse_ollama_tags(tags).unwrap();
        assert_eq!(models[0].id, "llava:latest");
        assert!(models[0].vision.is_ok());

        let chat = r#"{
            "message": {"role": "assistant", "content": "{\"labels\":[\"car\"]}"},
            "prompt_eval_count": 12,
            "eval_count": 8
        }"#;
        let resp = parse_ollama_chat(chat, 100).unwrap();
        assert!(resp.text.contains("car"));
        assert_eq!(resp.input_tokens, 12);
        assert_eq!(resp.output_tokens, 8);
        assert_eq!(resp.latency_ms, 100);
        assert!(!resp.hit_token_limit);
    }

    #[test]
    fn output_limit_reasons() {
        assert!(hit_output_limit("length"));
        assert!(hit_output_limit("max_tokens"));
        assert!(!hit_output_limit("stop"));
        assert!(!hit_output_limit("end_turn"));
    }
}
