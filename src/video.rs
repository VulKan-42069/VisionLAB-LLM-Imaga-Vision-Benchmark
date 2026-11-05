//! Video open, metadata, frame extract, and preview playback via ffmpeg-sidecar.

use anyhow::{anyhow, Context, Result};
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use image::RgbaImage;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::media::rgb24_to_rgba;

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub duration: f64,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub codec: String,
    pub frame_count: u64,
}

impl VideoInfo {
    pub fn frame_from_time(&self, seconds: f64) -> u64 {
        if self.fps <= 0.0 {
            return 0;
        }
        ((seconds * self.fps).floor() as u64).min(self.frame_count.saturating_sub(1))
    }

    pub fn time_from_frame(&self, frame: u64) -> f64 {
        if self.fps <= 0.0 {
            return 0.0;
        }
        (frame as f64 / self.fps).min(self.duration)
    }
}

fn tool_filename(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

fn is_bin(path: &Path) -> bool {
    path.is_file()
}

fn sidecar_tool(stem: &str) -> Option<PathBuf> {
    let mut path = std::env::current_exe().ok()?.parent()?.join(stem);
    if cfg!(windows) {
        path.set_extension("exe");
    }
    is_bin(&path).then_some(path)
}

fn path_tool(stem: &str) -> Option<PathBuf> {
    let name = tool_filename(stem);
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(&name);
        if is_bin(&candidate) {
            return Some(std::fs::canonicalize(&candidate).unwrap_or(candidate));
        }
    }
    None
}

fn winget_gyan_tool(stem: &str) -> Option<PathBuf> {
    let local = directories::BaseDirs::new()?.data_local_dir().to_path_buf();
    let packages = local.join("Microsoft").join("WinGet").join("Packages");
    let entries = std::fs::read_dir(&packages).ok()?;
    let name = tool_filename(stem);
    for entry in entries.flatten() {
        let folder = entry.file_name();
        if !folder.to_string_lossy().to_ascii_lowercase().starts_with("gyan.ffmpeg") {
            continue;
        }
        let inner = std::fs::read_dir(entry.path()).ok()?;
        for child in inner.flatten() {
            let candidate = child.path().join("bin").join(&name);
            if is_bin(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn sibling_tool(stem: &str, ffmpeg: &Path) -> Option<PathBuf> {
    let candidate = ffmpeg.with_file_name(tool_filename(stem));
    is_bin(&candidate).then_some(candidate)
}

/// Absolute path to ffmpeg or ffprobe. Never a bare name like `ffprobe`.
pub fn resolve_tool(stem: &str) -> Option<PathBuf> {
    if let Some(path) = sidecar_tool(stem) {
        return Some(path);
    }
    if let Some(ffmpeg) = sidecar_tool("ffmpeg").or_else(|| path_tool("ffmpeg")).or_else(|| winget_gyan_tool("ffmpeg")) {
        if let Some(path) = sibling_tool(stem, &ffmpeg) {
            return Some(path);
        }
    }
    path_tool(stem).or_else(|| winget_gyan_tool(stem))
}

fn install_sidecar_bundle() -> Result<()> {
    let dest = ffmpeg_sidecar::paths::sidecar_dir().context("locate app folder for ffmpeg")?;
    std::fs::create_dir_all(&dest)?;
    let url = ffmpeg_sidecar::download::ffmpeg_download_url().context("ffmpeg download URL")?;
    let archive = ffmpeg_sidecar::download::download_ffmpeg_package(url, &dest)
        .context("download ffmpeg zip")?;
    ffmpeg_sidecar::download::unpack_ffmpeg(&archive, &dest).context("unpack ffmpeg")?;
    Ok(())
}

pub fn ensure_ffmpeg() -> Result<String> {
    if let (Some(ffmpeg), Some(ffprobe)) = (resolve_tool("ffmpeg"), resolve_tool("ffprobe")) {
        return Ok(format!(
            "Video engine: {} and {}",
            ffmpeg.display(),
            ffprobe.display()
        ));
    }
    install_sidecar_bundle()?;
    match (resolve_tool("ffmpeg"), resolve_tool("ffprobe")) {
        (Some(ffmpeg), Some(ffprobe)) => Ok(format!(
            "Video engine: downloaded {} and {}",
            ffmpeg.display(),
            ffprobe.display()
        )),
        _ => Err(anyhow!(
            "ffmpeg/ffprobe are still missing after download. Install a full ffmpeg package (Gyan full build) and restart VisionLab."
        )),
    }
}

fn command_for(bin: &Path) -> Command {
    let mut cmd = Command::new(bin);
    if let Some(dir) = bin.parent() {
        let _ = cmd.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

fn ffmpeg_command() -> Result<FfmpegCommand> {
    let ffmpeg = resolve_tool("ffmpeg").ok_or_else(|| {
        anyhow!("ffmpeg was not found. Restart VisionLab so it can download a local copy, or install ffmpeg.")
    })?;
    let mut cmd = FfmpegCommand::new_with_path(&ffmpeg);
    if let Some(dir) = ffmpeg.parent() {
        cmd.as_inner_mut().current_dir(dir);
    }
    Ok(cmd)
}

fn kill_process(pid: u32) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut cmd = Command::new(
            Path::new(&std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into()))
                .join("System32")
                .join("taskkill.exe"),
        );
        cmd.args(["/PID", &pid.to_string(), "/F"]);
        cmd.creation_flags(0x0800_0000);
        let _ = cmd.stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

fn spawn_play_watchdog(
    pid: u32,
    playing: Arc<AtomicBool>,
    got_frame: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            if finished.load(Ordering::SeqCst) {
                return;
            }
            if !playing.load(Ordering::SeqCst) {
                kill_process(pid);
                return;
            }
            if !got_frame.load(Ordering::SeqCst) && start.elapsed() > Duration::from_secs(10) {
                timed_out.store(true, Ordering::SeqCst);
                playing.store(false, Ordering::SeqCst);
                kill_process(pid);
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let probe_bin = resolve_tool("ffprobe").ok_or_else(|| {
        anyhow!(
            "ffprobe was not found. VisionLab needs it to read video duration, size, and codec. Install a full ffmpeg package or restart the app so it can download a local copy."
        )
    })?;
    let output = command_for(&probe_bin)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(path)
        .output()
        .with_context(|| format!("Could not start ffprobe ({})", probe_bin.display()))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Unsupported or corrupt video.\n{err}");
    }
    let v: Value = serde_json::from_slice(&output.stdout).context("parse ffprobe")?;
    let streams = v.get("streams").and_then(Value::as_array).cloned().unwrap_or_default();
    let video = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(Value::as_str) == Some("video"))
        .ok_or_else(|| anyhow!("No video stream in this file."))?;
    let width = video.get("width").and_then(Value::as_u64).unwrap_or(0) as u32;
    let height = video.get("height").and_then(Value::as_u64).unwrap_or(0) as u32;
    let codec = video
        .get("codec_name")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let fps = parse_fps(video.get("avg_frame_rate").and_then(Value::as_str).unwrap_or("0/1"));
    let duration = v
        .pointer("/format/duration")
        .and_then(Value::as_str)
        .and_then(|s| s.parse().ok())
        .or_else(|| video.get("duration").and_then(Value::as_str).and_then(|s| s.parse().ok()))
        .unwrap_or(0.0);
    let frame_count = video
        .get("nb_frames")
        .and_then(Value::as_str)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            if fps > 0.0 {
                (duration * fps).round() as u64
            } else {
                0
            }
        });
    if width == 0 || height == 0 {
        anyhow::bail!("ffprobe did not report a video size for this file.");
    }
    Ok(VideoInfo {
        path: path.to_path_buf(),
        duration,
        fps: if fps > 0.0 { fps } else { 30.0 },
        width,
        height,
        codec,
        frame_count: frame_count.max(1),
    })
}

fn parse_fps(s: &str) -> f64 {
    if let Some((a, b)) = s.split_once('/') {
        let n: f64 = a.parse().unwrap_or(0.0);
        let d: f64 = b.parse().unwrap_or(1.0);
        if d.abs() < f64::EPSILON {
            0.0
        } else {
            n / d
        }
    } else {
        s.parse().unwrap_or(0.0)
    }
}

pub fn extract_frame(path: &Path, seconds: f64) -> Result<RgbaImage> {
    let ffmpeg = resolve_tool("ffmpeg").ok_or_else(|| {
        anyhow!("ffmpeg was not found. Restart VisionLab so it can download a local copy, or install ffmpeg.")
    })?;
    let child = command_for(&ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-loglevel")
        .arg("error")
        .arg("-ss")
        .arg(format!("{seconds:.3}"))
        .arg("-i")
        .arg(path)
        .arg("-an")
        .arg("-sn")
        .arg("-frames:v")
        .arg("1")
        .arg("-f")
        .arg("image2pipe")
        .arg("-vcodec")
        .arg("png")
        .arg("pipe:1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn ffmpeg for frame extract")?;
    let pid = child.id();
    let finished = Arc::new(AtomicBool::new(false));
    let finished_watch = finished.clone();
    std::thread::spawn(move || {
        let start = Instant::now();
        while !finished_watch.load(Ordering::SeqCst) {
            if start.elapsed() > Duration::from_secs(8) {
                kill_process(pid);
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    let output = child
        .wait_with_output()
        .context("wait ffmpeg frame extract")?;
    finished.store(true, Ordering::SeqCst);
    if output.stdout.is_empty() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg produced no frame at {seconds:.2}s. {err}");
    }
    Ok(image::load_from_memory(&output.stdout)
        .context("decode extracted frame")?
        .to_rgba8())
}

pub fn play_loop(
    path: PathBuf,
    start: f64,
    preview_fps: u32,
    playing: Arc<AtomicBool>,
    mut on_frame: impl FnMut(RgbaImage, f64),
) -> Result<()> {
    let rate = preview_fps.clamp(5, 24);
    // fps filter sets the preview cadence. scale uses an escaped comma so the
    // graph stays valid on Windows ffmpeg (quoted min() expressions often hang).
    let vf = format!("fps={rate},scale=w=min(iw\\,960):h=-2");
    let mut child = ffmpeg_command()?
        .hide_banner()
        .arg("-nostdin")
        .seek(format!("{start:.3}"))
        .input(path.to_string_lossy())
        .arg("-an")
        .arg("-sn")
        .args(["-vf", &vf])
        .rawvideo()
        .spawn()
        .context("Could not start ffmpeg for playback")?;
    let pid = child.as_inner().id();
    let got_frame = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    spawn_play_watchdog(
        pid,
        playing.clone(),
        got_frame.clone(),
        timed_out.clone(),
        finished.clone(),
    );
    let iter = match child.iter() {
        Ok(i) => i,
        Err(e) => {
            finished.store(true, Ordering::SeqCst);
            playing.store(false, Ordering::SeqCst);
            let _ = child.kill();
            return Err(e).context("Could not read ffmpeg preview output");
        }
    };
    let dt = 1.0 / rate as f64;
    let mut t = start;
    let mut frames = 0u32;
    let mut last_err = String::new();
    for event in iter {
        if !playing.load(Ordering::SeqCst) {
            break;
        }
        match event {
            FfmpegEvent::OutputFrame(frame) => {
                got_frame.store(true, Ordering::SeqCst);
                if let Ok(img) = rgb24_to_rgba(&frame.data, frame.width, frame.height) {
                    on_frame(img, t);
                    frames += 1;
                    t += dt;
                }
            }
            FfmpegEvent::Error(e) | FfmpegEvent::Log(LogLevel::Error, e) | FfmpegEvent::Log(LogLevel::Fatal, e) => {
                tracing::warn!("ffmpeg play: {e}");
                last_err = e;
            }
            FfmpegEvent::Done => break,
            _ => {}
        }
    }
    let stopped_by_user =
        !playing.load(Ordering::SeqCst) && !timed_out.load(Ordering::SeqCst);
    finished.store(true, Ordering::SeqCst);
    let _ = child.kill();
    let _ = child.wait();
    playing.store(false, Ordering::SeqCst);
    if timed_out.load(Ordering::SeqCst) {
        anyhow::bail!(
            "Playback timed out before the first preview frame. Try seeking to a later point, or open the file again."
        );
    }
    if frames == 0 && !stopped_by_user {
        if last_err.is_empty() {
            anyhow::bail!("ffmpeg produced no preview frames for this video.");
        }
        anyhow::bail!("Playback failed: {last_err}");
    }
    Ok(())
}

pub fn format_time(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "0:00".into();
    }
    let s = seconds as u64;
    let m = s / 60;
    let r = s % 60;
    if m >= 60 {
        format!("{}:{:02}:{:02}", m / 60, m % 60, r)
    } else {
        format!("{m}:{r:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_ffprobe_to_a_real_file() {
        let Some(path) = resolve_tool("ffprobe") else {
            eprintln!("skip: no ffprobe on this machine");
            return;
        };
        assert!(path.is_absolute(), "{path:?}");
        assert!(path.is_file(), "{path:?}");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        assert!(name.starts_with("ffprobe"), "{name}");
    }
}
