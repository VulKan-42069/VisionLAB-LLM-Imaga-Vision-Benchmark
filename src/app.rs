//! Shared application state.

use image::RgbaImage;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use uuid::Uuid;

use crate::providers::ProviderClient;
use crate::scoreboard::Scoreboard;
use crate::settings::Config;
use crate::tasks::TaskResult;
use crate::video::VideoInfo;

#[derive(Clone)]
pub enum PendingConfirm {
    None,
    ClearScoreboard,
    DeleteProvider,
    LeaveSettings { target: i32 },
    SendLargeImage,
}

pub struct LoadedMedia {
    pub path: PathBuf,
    #[allow(dead_code)]
    pub is_video: bool,
    pub source: RgbaImage,
    pub display: RgbaImage,
    pub width: u32,
    pub height: u32,
    pub notice: String,
}

pub struct VideoSession {
    pub info: VideoInfo,
    pub position: f64,
    pub playing: Arc<AtomicBool>,
    pub epoch: Arc<AtomicU64>,
    /// First-frame preview used by Stop so rewind does not call ffmpeg on the UI thread.
    pub poster: Option<RgbaImage>,
}

pub struct AppState {
    pub data_dir: PathBuf,
    pub config: Config,
    pub scoreboard: Scoreboard,
    #[allow(dead_code)]
    pub http: ProviderClient,
    pub media: Option<LoadedMedia>,
    pub video: Option<VideoSession>,
    pub last_run: Option<Uuid>,
    pub last_raw: String,
    pub last_result: Option<TaskResult>,
    pub pending_confirm: PendingConfirm,
    pub ffmpeg_msg: String,
    pub ffmpeg_ok: bool,
    pub filter: String,
    pub task_filter: i32,
    pub rt: tokio::runtime::Handle,
    pub running: bool,
    pub skip_huge_warn: bool,
}

#[derive(Clone)]
pub struct PickerItem {
    pub provider_id: String,
    pub model_id: String,
    pub label: String,
    pub vision_ok: bool,
    pub vision_label: String,
}

pub fn picker_items(cfg: &Config) -> Vec<PickerItem> {
    let mut out = Vec::new();
    for p in &cfg.providers {
        if !p.models.is_empty() {
            for m in &p.models {
                out.push(PickerItem {
                    provider_id: p.id.clone(),
                    model_id: m.id.clone(),
                    label: format!("{} / {}", p.name, m.name),
                    vision_ok: m.vision.is_ok(),
                    vision_label: m.vision.label().to_string(),
                });
            }
        } else if !p.default_model.is_empty() {
            let vs = crate::providers::vision_status_for(p, &p.default_model);
            out.push(PickerItem {
                provider_id: p.id.clone(),
                model_id: p.default_model.clone(),
                label: format!("{} / {}", p.name, p.default_model),
                vision_ok: vs.is_ok() || vs == crate::settings::VisionStatus::Unknown,
                vision_label: vs.label().to_string(),
            });
        }
    }
    out
}

pub fn current_picker_index(cfg: &Config, items: &[PickerItem]) -> i32 {
    let Some(pid) = &cfg.last_provider_id else {
        return if items.is_empty() { -1 } else { 0 };
    };
    let model = cfg.last_model.clone().unwrap_or_default();
    items
        .iter()
        .position(|i| &i.provider_id == pid && (i.model_id == model || model.is_empty()))
        .or_else(|| items.iter().position(|i| &i.provider_id == pid))
        .map(|i| i as i32)
        .unwrap_or(if items.is_empty() { -1 } else { 0 })
}
