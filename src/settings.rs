//! App configuration and API-key storage.
//! Keys live in Windows Credential Manager; `config.json` only stores key ids.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::tasks::TaskKind;

pub fn data_dir() -> PathBuf {
    directories::BaseDirs::new()
        .map(|b| b.config_dir().join("VisionLab"))
        .unwrap_or_else(|| PathBuf::from("VisionLab-data"))
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThemeMode {
    Light,
    Dark,
    System,
}

impl Default for ThemeMode {
    fn default() -> Self {
        Self::Dark
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenAi,
    Anthropic,
    OpenRouter,
    Ollama,
    Custom,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI",
            Self::Anthropic => "Anthropic",
            Self::OpenRouter => "OpenRouter",
            Self::Ollama => "Ollama",
            Self::Custom => "Custom",
        }
    }

    pub fn from_index(i: i32) -> Self {
        match i {
            0 => Self::OpenAi,
            1 => Self::Anthropic,
            2 => Self::OpenRouter,
            3 => Self::Ollama,
            _ => Self::Custom,
        }
    }

    pub fn to_index(self) -> i32 {
        match self {
            Self::OpenAi => 0,
            Self::Anthropic => 1,
            Self::OpenRouter => 2,
            Self::Ollama => 3,
            Self::Custom => 4,
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com",
            Self::OpenRouter => "https://openrouter.ai/api/v1",
            Self::Ollama => "http://localhost:11434",
            Self::Custom => "http://localhost:8000/v1",
        }
    }

    /// Cloud providers need a stored key. Ollama talks to a local server.
    pub fn requires_api_key(self) -> bool {
        !matches!(self, Self::Ollama)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionStatus {
    Confirmed,
    Probable,
    Unlikely,
    #[default]
    Unknown,
}

impl VisionStatus {
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Confirmed | Self::Probable)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Confirmed => "VISION",
            Self::Probable => "VISION?",
            Self::Unlikely => "TEXT ONLY",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub vision: VisionStatus,
    pub input_price_per_m: Option<f64>,
    pub output_price_per_m: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub name: String,
    pub kind: ProviderKind,
    pub base_url: String,
    #[serde(default)]
    pub default_model: String,
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

impl ProviderConfig {
    pub fn new(kind: ProviderKind) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: kind.as_str().to_string(),
            kind,
            base_url: kind.default_base_url().to_string(),
            default_model: String::new(),
            extra_headers: HashMap::new(),
            models: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub theme: ThemeMode,
    #[serde(default = "default_true")]
    pub restore_last_session: bool,
    #[serde(default = "default_preview_fps")]
    pub preview_fps: u32,
    #[serde(default = "default_frame_step")]
    pub frame_step: u32,
    #[serde(default = "default_max_dim")]
    pub max_image_dim: u32,
    #[serde(default)]
    pub timeline_thumbnails: bool,
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_retries")]
    pub retries: u32,
    #[serde(default)]
    pub default_temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub default_max_tokens: u32,
    #[serde(default = "default_true")]
    pub warn_huge_images: bool,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    pub last_provider_id: Option<String>,
    pub last_model: Option<String>,
    #[serde(default)]
    pub last_task: TaskKind,
    #[serde(default)]
    pub last_focus: String,
    #[serde(default = "default_screen")]
    pub last_screen: i32,
}

fn default_true() -> bool {
    true
}
fn default_preview_fps() -> u32 {
    20
}
fn default_frame_step() -> u32 {
    1
}
fn default_max_dim() -> u32 {
    1536
}
fn default_timeout() -> u64 {
    90
}
fn default_retries() -> u32 {
    2
}
fn default_max_tokens() -> u32 {
    6144
}
fn default_screen() -> i32 {
    1
}

impl Default for Config {
    fn default() -> Self {
        Self {
            theme: ThemeMode::Dark,
            restore_last_session: true,
            preview_fps: 20,
            frame_step: 1,
            max_image_dim: 1536,
            timeline_thumbnails: false,
            request_timeout_secs: 90,
            retries: 2,
            default_temperature: 0.0,
            default_max_tokens: 6144,
            warn_huge_images: true,
            providers: vec![
                ProviderConfig::new(ProviderKind::OpenAi),
                ProviderConfig::new(ProviderKind::Anthropic),
                ProviderConfig::new(ProviderKind::OpenRouter),
                ProviderConfig::new(ProviderKind::Ollama),
            ],
            last_provider_id: None,
            last_model: None,
            last_task: TaskKind::ObjectDetection,
            last_focus: String::new(),
            last_screen: 1,
        }
    }
}

impl Config {
    pub fn path(dir: &Path) -> PathBuf {
        dir.join("config.json")
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let path = Self::path(dir);
        if !path.exists() {
            let cfg = Self::default();
            cfg.save(dir)?;
            return Ok(cfg);
        }
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let mut cfg: Self = serde_json::from_str(&text).context("parse config.json")?;
        let mut dirty = cfg.ensure_ollama();
        if cfg.default_max_tokens == 4096 {
            cfg.default_max_tokens = default_max_tokens();
            dirty = true;
        }
        if dirty {
            let _ = cfg.save(dir);
        }
        Ok(cfg)
    }

    fn ensure_ollama(&mut self) -> bool {
        if self.providers.iter().any(|p| p.kind == ProviderKind::Ollama) {
            return false;
        }
        self.providers.push(ProviderConfig::new(ProviderKind::Ollama));
        true
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        let path = Self::path(dir);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    pub fn provider(&self, id: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.id == id)
    }

    pub fn provider_mut(&mut self, id: &str) -> Option<&mut ProviderConfig> {
        self.providers.iter_mut().find(|p| p.id == id)
    }
}

const KEYRING_SERVICE: &str = "VisionLab";

pub fn secret_id(provider_id: &str) -> String {
    format!("VisionLab/{provider_id}")
}

/// True when the field is empty or only a masked placeholder, not a real key.
pub fn is_key_placeholder(s: &str) -> bool {
    let t = s.trim();
    t.is_empty() || t.chars().all(|c| matches!(c, '•' | '*' | '·'))
}

fn credential_entry(provider_id: &str) -> Result<keyring::Entry> {
    keyring::Entry::new(KEYRING_SERVICE, &secret_id(provider_id))
        .context("open Windows Credential Manager")
}

pub fn get_api_key(provider_id: &str) -> Option<String> {
    let entry = credential_entry(provider_id).ok()?;
    let key = entry.get_password().ok()?;
    let key = key.trim().to_string();
    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

pub fn has_api_key(provider_id: &str) -> bool {
    get_api_key(provider_id).is_some()
}

pub fn set_api_key(provider_id: &str, key: &str) -> Result<()> {
    let key = key.trim();
    anyhow::ensure!(!key.is_empty(), "API key is empty");
    let entry = credential_entry(provider_id)?;
    entry.set_password(key).context("store API key")?;
    match get_api_key(provider_id) {
        Some(stored) if stored == key => Ok(()),
        _ => Err(anyhow::anyhow!(
            "Windows Credential Manager did not keep the key. Try Save again."
        )),
    }
}

pub fn delete_api_key(provider_id: &str) {
    if let Ok(entry) = credential_entry(provider_id) {
        let _ = entry.delete_credential();
    }
}

pub fn provider_auth_ok(p: &ProviderConfig) -> bool {
    !p.kind.requires_api_key() || has_api_key(&p.id)
}

pub fn provider_status_label(p: &ProviderConfig) -> &'static str {
    if p.kind == ProviderKind::Ollama {
        "Local"
    } else if has_api_key(&p.id) {
        "Key saved"
    } else {
        "No key"
    }
}

/// Token for HTTP calls. Empty string for local Ollama when no key is stored.
pub fn request_api_key(p: &ProviderConfig) -> Result<String> {
    if let Some(k) = get_api_key(&p.id) {
        return Ok(k);
    }
    if p.kind.requires_api_key() {
        Err(anyhow::anyhow!(
            "This provider has no key in Windows Credential Manager. Open Settings, paste the key, and Save."
        ))
    } else {
        Ok(String::new())
    }
}

pub fn any_provider_ready(cfg: &Config) -> bool {
    cfg.providers.iter().any(|p| {
        if p.kind.requires_api_key() {
            has_api_key(&p.id)
        } else {
            !p.default_model.is_empty() || !p.models.is_empty()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_keys_are_detected() {
        assert!(is_key_placeholder(""));
        assert!(is_key_placeholder("   "));
        assert!(is_key_placeholder("••••••••"));
        assert!(is_key_placeholder("********"));
        assert!(!is_key_placeholder("sk-abc"));
        assert!(!is_key_placeholder("or-v1-real-key"));
    }

    #[test]
    fn ollama_kind_index_and_no_key() {
        assert_eq!(ProviderKind::from_index(3), ProviderKind::Ollama);
        assert_eq!(ProviderKind::Ollama.to_index(), 3);
        assert_eq!(ProviderKind::Custom.to_index(), 4);
        assert!(!ProviderKind::Ollama.requires_api_key());
        assert_eq!(ProviderKind::Ollama.default_base_url(), "http://localhost:11434");
        let p = ProviderConfig::new(ProviderKind::Ollama);
        assert!(provider_auth_ok(&p));
        assert_eq!(provider_status_label(&p), "Local");
        assert_eq!(request_api_key(&p).unwrap(), "");
    }

    #[test]
    fn credential_manager_roundtrip() {
        let id = "visionlab-keyring-selftest";
        delete_api_key(id);
        assert!(get_api_key(id).is_none());
        set_api_key(id, " sk-test-visionlab ").expect("store key");
        assert_eq!(get_api_key(id).as_deref(), Some("sk-test-visionlab"));
        delete_api_key(id);
        assert!(get_api_key(id).is_none());
    }
}
