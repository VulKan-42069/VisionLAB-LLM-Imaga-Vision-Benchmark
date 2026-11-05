//! Persistent run history and scoreboard stats.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::cost::format_usd;
use crate::settings::ProviderKind;
use crate::tasks::{TaskKind, TaskResult};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
    Pending,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Pending => "pending",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MediaRef {
    Image {
        path: String,
        width: u32,
        height: u32,
    },
    Video {
        path: String,
        frame_ts: f64,
        frame_index: u64,
        width: u32,
        height: u32,
    },
}

impl MediaRef {
    pub fn display(&self) -> String {
        match self {
            Self::Image { path, .. } => Path::new(path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(path)
                .to_string(),
            Self::Video {
                path,
                frame_ts,
                frame_index,
                ..
            } => {
                let name = Path::new(path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(path);
                format!("{name}  f{frame_index} ({frame_ts:.2}s)")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub provider: ProviderKind,
    pub provider_name: String,
    pub model: String,
    pub task: TaskKind,
    pub media: MediaRef,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub est_cost_usd: Option<f64>,
    pub result: TaskResult,
    pub raw_response: String,
    pub annotated_path: Option<String>,
    pub verdict: Verdict,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub focus: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Scoreboard {
    pub runs: Vec<RunRecord>,
}

impl Scoreboard {
    pub fn path(dir: &Path) -> PathBuf {
        dir.join("scoreboard.json")
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let path = Self::path(dir);
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        Ok(serde_json::from_str(&text).context("parse scoreboard.json")?)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        fs::write(Self::path(dir), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn push(&mut self, rec: RunRecord) {
        self.runs.insert(0, rec);
    }

    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut RunRecord> {
        self.runs.iter_mut().find(|r| r.id == id)
    }

    pub fn filtered(&self, query: &str, task_idx: i32) -> Vec<&RunRecord> {
        let q = query.trim().to_ascii_lowercase();
        self.runs
            .iter()
            .filter(|r| {
                if task_idx > 0 && r.task.to_index() != task_idx - 1 {
                    return false;
                }
                if q.is_empty() {
                    return true;
                }
                r.model.to_ascii_lowercase().contains(&q)
                    || r.provider_name.to_ascii_lowercase().contains(&q)
                    || r.notes.to_ascii_lowercase().contains(&q)
                    || r.focus.to_ascii_lowercase().contains(&q)
                    || r.media.display().to_ascii_lowercase().contains(&q)
            })
            .collect()
    }

    pub fn stats(&self, rows: &[&RunRecord]) -> (u32, String, String, String) {
        let n = rows.len() as u32;
        let passes = rows.iter().filter(|r| r.verdict == Verdict::Pass).count();
        let graded = rows
            .iter()
            .filter(|r| r.verdict != Verdict::Pending)
            .count();
        let pass_rate = if graded == 0 {
            "n/a".into()
        } else {
            format!("{:.0}%", (passes as f64) * 100.0 / graded as f64)
        };
        let avg = if n == 0 {
            "n/a".into()
        } else {
            let ms = rows.iter().map(|r| r.latency_ms).sum::<u64>() as f64 / n as f64;
            format!("{:.2}s", ms / 1000.0)
        };
        let cost: Option<f64> = if n == 0 {
            None
        } else {
            rows.iter().try_fold(0.0, |acc, r| Some(acc + r.est_cost_usd?))
        };
        (n, pass_rate, avg, format_usd(cost))
    }

    pub fn to_csv(&self, rows: &[&RunRecord]) -> String {
        let mut out = String::from(
            "id,timestamp,provider,model,task,focus,media,latency_ms,input_tokens,output_tokens,cost_usd,verdict,notes\n",
        );
        for r in rows {
            let cost = r.est_cost_usd.map(|c| format!("{c:.6}")).unwrap_or_default();
            let notes = r.notes.replace('"', "'").replace('\n', " ");
            let focus = r.focus.replace('"', "'").replace('\n', " ");
            out.push_str(&format!(
                "{},{},{},{},{},\"{}\",{},{},{},{},{},{},\"{}\"\n",
                r.id,
                r.timestamp.to_rfc3339(),
                r.provider_name,
                r.model,
                r.task.label(),
                focus,
                r.media.display().replace(',', " "),
                r.latency_ms,
                r.input_tokens,
                r.output_tokens,
                cost,
                r.verdict.as_str(),
                notes
            ));
        }
        out
    }
}

pub fn runs_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("runs")
}
