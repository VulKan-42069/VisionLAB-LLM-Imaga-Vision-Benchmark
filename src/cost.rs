//! Token-cost estimates. OpenRouter supplies live prices; others use a family map.

use crate::settings::{ModelEntry, ProviderKind};

/// USD per 1M tokens for known OpenAI / Anthropic families.
pub fn builtin_price(model: &str) -> Option<(f64, f64)> {
    let m = model.to_ascii_lowercase();
    let pair = if m.contains("gpt-4o-mini") || m.contains("gpt-4.1-mini") {
        (0.15, 0.60)
    } else if m.contains("gpt-4o") {
        (2.50, 10.00)
    } else if m.contains("gpt-4.1-nano") {
        (0.10, 0.40)
    } else if m.contains("gpt-4.1") {
        (2.00, 8.00)
    } else if m.contains("o4-mini") || m.contains("o3-mini") {
        (1.10, 4.40)
    } else if m.contains("gpt-5-mini") {
        (0.25, 2.00)
    } else if m.contains("gpt-5") {
        (1.25, 10.00)
    } else if m.contains("haiku") {
        (0.80, 4.00)
    } else if m.contains("sonnet") {
        (3.00, 15.00)
    } else if m.contains("opus") {
        (15.00, 75.00)
    } else {
        return None;
    };
    Some(pair)
}

pub fn price_for(kind: ProviderKind, model: &str, entry: Option<&ModelEntry>) -> Option<(f64, f64)> {
    if let Some(e) = entry {
        if let (Some(i), Some(o)) = (e.input_price_per_m, e.output_price_per_m) {
            return Some((i, o));
        }
    }
    match kind {
        ProviderKind::OpenAi | ProviderKind::Anthropic => builtin_price(model),
        ProviderKind::Ollama => Some((0.0, 0.0)),
        ProviderKind::OpenRouter | ProviderKind::Custom => {
            entry.and_then(|e| Some((e.input_price_per_m?, e.output_price_per_m?)))
        }
    }
}

pub fn estimate_usd(input_tokens: u32, output_tokens: u32, per_m: Option<(f64, f64)>) -> Option<f64> {
    let (inp, out) = per_m?;
    Some((input_tokens as f64) * inp / 1_000_000.0 + (output_tokens as f64) * out / 1_000_000.0)
}

pub fn format_usd(v: Option<f64>) -> String {
    match v {
        Some(x) if x < 0.0001 && x > 0.0 => format!("${x:.6}"),
        Some(x) => format!("${x:.4}"),
        None => "n/a".into(),
    }
}

/// OpenRouter lists USD per token. Convert to per-1M when the value looks per-token.
pub fn normalize_per_m(raw: f64) -> f64 {
    if raw > 0.0 && raw < 0.01 {
        raw * 1_000_000.0
    } else {
        raw
    }
}
