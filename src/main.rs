#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! VisionLab - LLM Vision Benchmark Studio

mod annotate;
mod app;
mod cost;
mod media;
mod providers;
mod scoreboard;
mod settings;
mod tasks;
mod ui;
mod video;

use anyhow::Result;
use app::AppState;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

slint::include_modules!();

fn main() -> Result<()> {
    let data_dir = settings::data_dir();
    std::fs::create_dir_all(data_dir.join("logs"))?;
    std::fs::create_dir_all(data_dir.join("runs"))?;

    let file_appender = tracing_appender::rolling::daily(data_dir.join("logs"), "visionlab.log");
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::fmt::layer().with_ansi(false).with_writer(file_appender))
        .init();

    tracing::info!("Starting VisionLab v{}", env!("CARGO_PKG_VERSION"));

    if let Err(err) = slint::BackendSelector::new()
        .backend_name("winit".into())
        .renderer_name("skia".into())
        .select()
    {
        tracing::warn!("Skia renderer unavailable ({err}), using the default backend");
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("visionlab-async")
        .build()?;
    let rt_handle = rt.handle().clone();
    std::thread::spawn(move || {
        rt.block_on(std::future::pending::<()>());
    });

    let skip_splash = std::env::var("VISIONLAB_SKIP_SPLASH").ok().as_deref() == Some("1");

    let ui = MainWindow::new()?;
    let api = ui.global::<Api>();
    api.set_splash_visible(!skip_splash);
    api.set_splash_opacity(if skip_splash { 0.0 } else { 1.0 });
    api.set_splash_status("Loading settings…".into());
    api.set_screen(1);

    if skip_splash {
        let state = init_state(data_dir, rt_handle)?;
        let app = Arc::new(Mutex::new(state));
        ui::bind(&ui, app);
        ui.run()?;
        return Ok(());
    }

    let ui_weak = ui.as_weak();
    std::thread::spawn(move || {
        let set_status = |status: &str, progress: f32| {
            let status = status.to_string();
            let _ = slint::invoke_from_event_loop({
                let ui_weak = ui_weak.clone();
                move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        let api = ui.global::<Api>();
                        api.set_splash_status(status.into());
                        api.set_splash_progress(progress);
                    }
                }
            });
        };

        set_status("Loading settings…", 0.2);
        let result = init_state(data_dir, rt_handle);

        match result {
            Ok(state) => {
                set_status("Checking video engine…", 0.55);
                std::thread::sleep(Duration::from_millis(180));
                set_status("Ready", 1.0);
                std::thread::sleep(Duration::from_millis(220));
                let _ = slint::invoke_from_event_loop({
                    let ui_weak = ui_weak.clone();
                    move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let app = Arc::new(Mutex::new(state));
                            ui::bind(&ui, app);
                            ui.global::<Api>().set_splash_opacity(0.0);
                        }
                    }
                });
                std::thread::sleep(Duration::from_millis(280));
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.global::<Api>().set_splash_visible(false);
                    }
                });
            }
            Err(e) => {
                tracing::error!("init failed: {e:#}");
                let msg = e.to_string();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        ui.global::<Api>().set_splash_visible(false);
                        ui.global::<Api>().set_error_title("Startup failed".into());
                        ui.global::<Api>().set_error_message(msg.into());
                        ui.global::<Api>().set_error_open(true);
                    }
                });
            }
        }
    });

    ui.run()?;
    Ok(())
}

fn init_state(data_dir: std::path::PathBuf, rt: tokio::runtime::Handle) -> Result<AppState> {
    let config = settings::Config::load(&data_dir)?;
    let scoreboard = scoreboard::Scoreboard::load(&data_dir)?;
    let (ffmpeg_ok, ffmpeg_msg) = match video::ensure_ffmpeg() {
        Ok(msg) => (true, msg),
        Err(e) => (false, e.to_string()),
    };
    let http = providers::ProviderClient::new(Duration::from_secs(config.request_timeout_secs.max(10)))?;
    let _ = ffmpeg_ok;
    Ok(AppState {
        data_dir,
        config,
        scoreboard,
        http,
        media: None,
        video: None,
        last_run: None,
        last_raw: String::new(),
        last_result: None,
        pending_confirm: app::PendingConfirm::None,
        ffmpeg_msg,
        ffmpeg_ok,
        filter: String::new(),
        task_filter: 0,
        rt,
        running: false,
        skip_huge_warn: false,
    })
}
