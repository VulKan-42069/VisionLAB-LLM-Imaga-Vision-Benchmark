//! Slint callback wiring and UI refresh helpers.

use anyhow::Result;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel, Weak};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use uuid::Uuid;

use crate::annotate::{self, slint_color};
use crate::app::{current_picker_index, picker_items, AppState, LoadedMedia, PendingConfirm, PickerItem, VideoSession};
use crate::cost::{estimate_usd, format_usd, price_for};
use crate::media;
use crate::providers::{self, RunRequest};
use crate::scoreboard::{MediaRef, RunRecord, Scoreboard, Verdict};
use crate::settings::{
    self, Config, ModelEntry, ProviderConfig, ProviderKind, ThemeMode, VisionStatus,
};
use crate::tasks::{parse_task, ParsedTask, TaskKind, TaskResult};
use crate::video;
use crate::{Api, LegendItem, MainWindow, ModelRow, ProviderRow, ScoreboardRow, Theme};

pub type SharedApp = Arc<Mutex<AppState>>;

fn provider_http_timeout_secs(kind: ProviderKind, configured: u64) -> u64 {
    if kind == ProviderKind::Ollama {
        configured.max(180)
    } else {
        configured.max(10)
    }
}

pub fn bind(ui: &MainWindow, app: SharedApp) {
    refresh_all(ui, &app);
    if ui.global::<Api>().get_selected_provider() < 0
        && !app.lock().unwrap().config.providers.is_empty()
    {
        load_provider_form(ui, &app, 0);
    }
    if app.lock().unwrap().config.restore_last_session {
        let screen = app.lock().unwrap().config.last_screen;
        if (1..=4).contains(&screen) {
            ui.global::<Api>().set_screen(screen);
        }
    }

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_toggle_theme(move || {
        if let Some(ui) = w.upgrade() {
            let mut g = a.lock().unwrap();
            g.config.theme = match g.config.theme {
                ThemeMode::Dark => ThemeMode::Light,
                _ => ThemeMode::Dark,
            };
            let _ = g.config.save(&g.data_dir);
            drop(g);
            apply_theme(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_goto_screen(move |screen| {
        if let Some(ui) = w.upgrade() {
            maybe_goto(&ui, &a, screen);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_open_image(move || {
        if let Some(ui) = w.upgrade() {
            open_image(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_open_video(move || {
        if let Some(ui) = w.upgrade() {
            open_video(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_run_test(move || {
        if let Some(ui) = w.upgrade() {
            start_run(&ui, &a, false);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_retry_test(move || {
        if let Some(ui) = w.upgrade() {
            start_run(&ui, &a, true);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_export_result(move || {
        if let Some(ui) = w.upgrade() {
            export_result(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_copy_json(move || {
        if let Some(ui) = w.upgrade() {
            let raw = a.lock().unwrap().last_raw.clone();
            if raw.is_empty() {
                show_error(&ui, "Nothing to copy", "Run a test first.", "");
                return;
            }
            match arboard::Clipboard::new().and_then(|mut c| c.set_text(raw)) {
                Ok(()) => show_toast(&ui, "Copied JSON to the clipboard."),
                Err(e) => show_error(&ui, "Clipboard failed", &e.to_string(), ""),
            }
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_clear_result(move || {
        if let Some(ui) = w.upgrade() {
            stop_playing(&a);
            {
                let mut g = a.lock().unwrap();
                g.media = None;
                g.video = None;
            }
            clear_current_result(&ui, &a);
            drop_refresh(&ui, &a);
            show_toast(&ui, "Image and results cleared.");
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_grade_pass(move || grade(&w, &a, Verdict::Pass));

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_grade_fail(move || grade(&w, &a, Verdict::Fail));

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_select_task(move |idx| {
        if let Some(ui) = w.upgrade() {
            let mut g = a.lock().unwrap();
            g.config.last_task = TaskKind::from_index(idx);
            let _ = g.config.save(&g.data_dir);
            ui.global::<Api>().set_task_index(idx);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_select_model(move |idx| {
        if let Some(ui) = w.upgrade() {
            if idx < 0 {
                return;
            }
            let items = picker_items(&a.lock().unwrap().config);
            if let Some(item) = items.get(idx as usize) {
                let mut g = a.lock().unwrap();
                g.config.last_provider_id = Some(item.provider_id.clone());
                g.config.last_model = Some(item.model_id.clone());
                let _ = g.config.save(&g.data_dir);
            }
            drop_refresh(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_notes_changed(move |text| {
        if let Some(ui) = w.upgrade() {
            ui.global::<Api>().set_notes_text(text.clone());
            let mut g = a.lock().unwrap();
            if let Some(id) = g.last_run {
                if let Some(rec) = g.scoreboard.runs.iter_mut().find(|r| r.id == id) {
                    rec.notes = text.to_string();
                }
                let dir = g.data_dir.clone();
                let _ = g.scoreboard.save(&dir);
            }
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_video_play_toggle(move || {
        if let Some(ui) = w.upgrade() {
            toggle_play(&ui, &a);
        }
    });
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_video_stop(move || {
        if let Some(ui) = w.upgrade() {
            stop_video(&ui, &a);
        }
    });
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_video_seek(move |pos| {
        if let Some(ui) = w.upgrade() {
            seek_video(&ui, &a, pos);
        }
    });
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_video_step(move |delta| {
        if let Some(ui) = w.upgrade() {
            step_video(&ui, &a, delta);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_confirm_ok(move || {
        if let Some(ui) = w.upgrade() {
            confirm_ok(&ui, &a);
        }
    });
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_confirm_cancel(move || {
        if let Some(ui) = w.upgrade() {
            a.lock().unwrap().pending_confirm = PendingConfirm::None;
            ui.global::<Api>().set_confirm_open(false);
        }
    });
    let w = ui.as_weak();
    ui.global::<Api>().on_error_dismiss(move || {
        if let Some(ui) = w.upgrade() {
            ui.global::<Api>().set_error_open(false);
        }
    });
    let w = ui.as_weak();
    ui.global::<Api>().on_toast_dismiss(move || {
        if let Some(ui) = w.upgrade() {
            ui.global::<Api>().set_toast_visible(false);
        }
    });

    bind_settings(ui, app.clone());
    bind_scoreboard(ui, app);
}

fn bind_settings(ui: &MainWindow, app: SharedApp) {
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_section_changed(move |s| {
        if let Some(ui) = w.upgrade() {
            ui.global::<Api>().set_settings_section(s);
            if s == 1 {
                refresh_catalog_view(&ui, &a);
            }
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_add_provider(move || {
        if let Some(ui) = w.upgrade() {
            let mut g = a.lock().unwrap();
            g.config.providers.push(ProviderConfig::new(ProviderKind::Custom));
            let idx = g.config.providers.len() as i32 - 1;
            let _ = g.config.save(&g.data_dir);
            drop(g);
            refresh_settings(&ui, &a);
            ui.global::<Api>().set_selected_provider(idx);
            load_provider_form(&ui, &a, idx);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_select_provider(move |idx| {
        if let Some(ui) = w.upgrade() {
            load_provider_form(&ui, &a, idx);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_kind_changed(move |kind| {
        if let Some(ui) = w.upgrade() {
            let idx = ui.global::<Api>().get_selected_provider();
            let saved_kind = a
                .lock()
                .unwrap()
                .config
                .providers
                .get(idx as usize)
                .map(|p| p.kind.to_index());
            ui.global::<Api>().set_edit_kind(kind);
            if saved_kind != Some(kind) {
                let k = ProviderKind::from_index(kind);
                ui.global::<Api>().set_edit_base_url(k.default_base_url().into());
            }
            ui.global::<Api>().set_settings_dirty(true);
        }
    });

    let w = ui.as_weak();
    ui.global::<Api>().on_settings_field_changed(move || {
        if let Some(ui) = w.upgrade() {
            ui.global::<Api>().set_settings_dirty(true);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_save(move || {
        if let Some(ui) = w.upgrade() {
            let _ = save_settings(&ui, &a, true);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_delete_provider(move || {
        if let Some(ui) = w.upgrade() {
            a.lock().unwrap().pending_confirm = PendingConfirm::DeleteProvider;
            show_confirm(
                &ui,
                "Delete provider",
                "Remove this provider from the list? The stored API key is also deleted.",
                "",
                "Delete",
                true,
            );
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_test_connection(move || {
        if let Some(ui) = w.upgrade() {
            test_connection(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_refresh_models(move || {
        if let Some(ui) = w.upgrade() {
            refresh_models_async(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_probe_vision(move || {
        if let Some(ui) = w.upgrade() {
            probe_vision(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_select_model(move |idx| {
        if let Some(ui) = w.upgrade() {
            use_catalog_model(&ui, &a, idx);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_theme_mode(move |mode| {
        if let Some(ui) = w.upgrade() {
            let mut g = a.lock().unwrap();
            g.config.theme = match mode {
                0 => ThemeMode::Light,
                1 => ThemeMode::Dark,
                _ => ThemeMode::System,
            };
            let _ = g.config.save(&g.data_dir);
            drop(g);
            ui.global::<Api>().set_theme_mode(mode);
            apply_theme(&ui, &a);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_settings_restore_session(move |on| {
        if let Some(ui) = w.upgrade() {
            let mut g = a.lock().unwrap();
            g.config.restore_last_session = on;
            let _ = g.config.save(&g.data_dir);
            drop(g);
            ui.global::<Api>().set_restore_session(on);
        }
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_open_data_dir(move || {
        let dir = a.lock().unwrap().data_dir.clone();
        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
        let _ = w;
    });

    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_export_config(move || export_config(&w, &a));
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_import_config(move || import_config(&w, &a));
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_backup_scoreboard(move || backup_scoreboard(&w, &a));
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_restore_scoreboard(move || restore_scoreboard(&w, &a));
}

fn bind_scoreboard(ui: &MainWindow, app: SharedApp) {
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_scoreboard_filter_changed(move |q| {
        if let Some(ui) = w.upgrade() {
            a.lock().unwrap().filter = q.to_string();
            refresh_scoreboard(&ui, &a);
        }
    });
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_scoreboard_task_filter_changed(move |idx| {
        if let Some(ui) = w.upgrade() {
            a.lock().unwrap().task_filter = idx;
            ui.global::<Api>().set_scoreboard_task_filter(idx);
            refresh_scoreboard(&ui, &a);
        }
    });
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_scoreboard_clear(move || {
        if let Some(ui) = w.upgrade() {
            a.lock().unwrap().pending_confirm = PendingConfirm::ClearScoreboard;
            show_confirm(
                &ui,
                "Clear scoreboard",
                "This deletes every stored run. Annotated images in the data folder stay on disk.",
                "",
                "Clear",
                true,
            );
        }
    });
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_scoreboard_export_csv(move || export_board(&w, &a, true));
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_scoreboard_export_json(move || export_board(&w, &a, false));
    let w = ui.as_weak();
    let a = app.clone();
    ui.global::<Api>().on_scoreboard_open_row(move |idx| {
        if let Some(ui) = w.upgrade() {
            open_row(&ui, &a, idx);
        }
    });
}

pub fn refresh_all(ui: &MainWindow, app: &SharedApp) {
    apply_theme(ui, app);
    drop_refresh(ui, app);
    refresh_settings(ui, app);
    refresh_scoreboard(ui, app);
    let g = app.lock().unwrap();
    let api = ui.global::<Api>();
    api.set_app_version(env!("CARGO_PKG_VERSION").into());
    api.set_data_dir(g.data_dir.to_string_lossy().to_string().into());
    api.set_ffmpeg_status(g.ffmpeg_msg.clone().into());
    api.set_about_text(
        "Keys stay in Windows Credential Manager.\nLogs: the data directory /logs folder.\nNo local inference in v1."
            .into(),
    );
    api.set_preview_fps(g.config.preview_fps as i32);
    api.set_frame_step(g.config.frame_step as i32);
    api.set_max_dim(g.config.max_image_dim as i32);
    api.set_timeout_secs(g.config.request_timeout_secs as i32);
    api.set_retries(g.config.retries as i32);
    api.set_max_tokens(g.config.default_max_tokens as i32);
    api.set_temperature(g.config.default_temperature);
    api.set_warn_huge(g.config.warn_huge_images);
    api.set_restore_session(g.config.restore_last_session);
    api.set_theme_mode(match g.config.theme {
        ThemeMode::Light => 0,
        ThemeMode::Dark => 1,
        ThemeMode::System => 2,
    });
    api.set_task_index(g.config.last_task.to_index());
    api.set_focus_prompt(g.config.last_focus.clone().into());
}

fn drop_refresh(ui: &MainWindow, app: &SharedApp) {
    let g = app.lock().unwrap();
    let api = ui.global::<Api>();
    let items = picker_items(&g.config);
    let mut names: Vec<SharedString> = items.iter().map(|i| i.label.clone().into()).collect();
    let idx = current_picker_index(&g.config, &items);
    if names.is_empty() {
        names.push("Select a model in Settings".into());
        api.set_model_index(0);
    } else {
        api.set_model_index(idx);
    }
    api.set_model_names(ModelRc::new(VecModel::from(names)));
    if let Some(item) = items.get(idx.max(0) as usize) {
        api.set_selected_model(item.label.clone().into());
        api.set_model_vision_ok(item.vision_ok);
        api.set_model_vision_label(item.vision_label.clone().into());
    } else {
        api.set_selected_model("Select a model in Settings".into());
        api.set_model_vision_ok(false);
        api.set_model_vision_label("".into());
    }
    api.set_show_connect_banner(!settings::any_provider_ready(&g.config));
    let has_media = g.media.is_some();
    api.set_has_media(has_media);
    api.set_is_video(g.video.is_some());
    api.set_run_enabled(has_media && idx >= 0 && !g.running);
    api.set_is_running(g.running);
    if let Some(m) = &g.media {
        api.set_original_image(media::to_slint(&m.source));
        api.set_canvas_image(media::to_slint(&m.display));
        api.set_media_label(media::file_stem(&m.path).into());
        api.set_downscale_notice(m.notice.clone().into());
        api.set_media_meta(format!("{} × {}", m.width, m.height).into());
    } else {
        api.set_media_label("".into());
        api.set_downscale_notice("".into());
        api.set_media_meta("".into());
    }
    if let Some(v) = &g.video {
        api.set_video_position((v.position / v.info.duration.max(0.001)) as f32);
        api.set_video_time(
            format!(
                "{} / {}",
                video::format_time(v.position),
                video::format_time(v.info.duration)
            )
            .into(),
        );
        let f = v.info.frame_from_time(v.position);
        api.set_video_frame(format!("Frame {f} / {}", v.info.frame_count).into());
        api.set_video_playing(v.playing.load(Ordering::SeqCst));
        api.set_media_meta(
            format!(
                "{} × {}  {:.2} fps  {}  {}",
                v.info.width,
                v.info.height,
                v.info.fps,
                v.info.codec,
                video::format_time(v.info.duration)
            )
            .into(),
        );
    } else {
        api.set_video_playing(false);
        api.set_video_position(0.0);
        api.set_video_time("".into());
        api.set_video_frame("".into());
    }
    api.set_has_result(g.last_run.is_some());
    api.set_grade_enabled(g.last_run.is_some());
    push_json_preview(&api, &g.last_raw);
}

fn refresh_playback(ui: &MainWindow, app: &SharedApp) {
    let g = app.lock().unwrap();
    let api = ui.global::<Api>();
    if let Some(m) = &g.media {
        api.set_canvas_image(media::to_slint(&m.display));
    }
    if let Some(v) = &g.video {
        api.set_video_position((v.position / v.info.duration.max(0.001)) as f32);
        api.set_video_time(
            format!(
                "{} / {}",
                video::format_time(v.position),
                video::format_time(v.info.duration)
            )
            .into(),
        );
        let f = v.info.frame_from_time(v.position);
        api.set_video_frame(format!("Frame {f} / {}", v.info.frame_count).into());
        api.set_video_playing(v.playing.load(Ordering::SeqCst));
    }
}

fn clear_current_result(ui: &MainWindow, app: &SharedApp) {
    let mut g = app.lock().unwrap();
    g.last_run = None;
    g.last_raw.clear();
    g.last_result = None;
    drop(g);
    let api = ui.global::<Api>();
    api.set_has_result(false);
    api.set_grade_enabled(false);
    api.set_raw_json("".into());
    api.set_raw_json_clipped(false);
    api.set_notes_text("".into());
    api.set_last_verdict("".into());
    api.set_latency_text("".into());
    api.set_tokens_text("".into());
    api.set_cost_text("".into());
    api.set_result_summary("Run a test to see results".into());
    api.set_legend(ModelRc::new(VecModel::<LegendItem>::from(vec![])));
}

fn apply_theme(ui: &MainWindow, app: &SharedApp) {
    let mode = app.lock().unwrap().config.theme;
    let dark = match mode {
        ThemeMode::Light => false,
        ThemeMode::Dark => true,
        ThemeMode::System => system_is_dark(),
    };
    ui.global::<Api>().set_is_dark(dark);
    ui.global::<Theme>().set_is_dark(dark);
}

fn system_is_dark() -> bool {
    let out = std::process::Command::new("reg")
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            "/v",
            "AppsUseLightTheme",
        ])
        .output();
    match out {
        Ok(o) => !String::from_utf8_lossy(&o.stdout).to_ascii_lowercase().contains("0x1"),
        Err(_) => true,
    }
}

fn maybe_goto(ui: &MainWindow, app: &SharedApp, screen: i32) {
    let dirty = ui.global::<Api>().get_settings_dirty();
    let current = ui.global::<Api>().get_screen();
    if current == 3 && screen != 3 && dirty {
        app.lock().unwrap().pending_confirm = PendingConfirm::LeaveSettings { target: screen };
        show_confirm(
            ui,
            "Unsaved settings",
            "Leave Settings without saving the current form?",
            "Use Save on the provider form to keep API keys and URLs.",
            "Leave",
            false,
        );
        return;
    }
    ui.global::<Api>().set_screen(screen);
    if (1..=4).contains(&screen) {
        let mut g = app.lock().unwrap();
        if g.config.last_screen != screen {
            g.config.last_screen = screen;
            let _ = g.config.save(&g.data_dir);
        }
    }
}

fn open_image(ui: &MainWindow, app: &SharedApp) {
    let path = rfd::FileDialog::new()
        .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"])
        .pick_file();
    let Some(path) = path else { return };
    stop_playing(app);
    match load_still(&path, app) {
        Ok(()) => {
            app.lock().unwrap().video = None;
            clear_current_result(ui, app);
            drop_refresh(ui, app);
            ui.global::<Api>().set_is_video(false);
        }
        Err(e) => show_error(ui, "Could not open image", &format!("{e:#}"), ""),
    }
}

fn load_still(path: &PathBuf, app: &SharedApp) -> Result<()> {
    let img = media::load_image(path)?;
    let mut g = app.lock().unwrap();
    let max = g.config.max_image_dim;
    let (img, scaled) = media::downscale(img, max);
    let notice = if scaled {
        format!("downscaled to {}px", img.width().max(img.height()))
    } else {
        String::new()
    };
    g.media = Some(LoadedMedia {
        path: path.clone(),
        is_video: false,
        width: img.width(),
        height: img.height(),
        notice,
        source: img.clone(),
        display: img,
    });
    Ok(())
}

fn open_video(ui: &MainWindow, app: &SharedApp) {
    if !app.lock().unwrap().ffmpeg_ok {
        show_error(
            ui,
            "Video engine unavailable",
            "Install ffmpeg and add it to PATH, or allow the app to download a local binary.",
            &app.lock().unwrap().ffmpeg_msg,
        );
        return;
    }
    let path = rfd::FileDialog::new()
        .add_filter("Video", &["mp4", "webm", "mov", "mkv", "avi"])
        .pick_file();
    let Some(path) = path else { return };
    stop_playing(app);
    match video::probe(&path) {
        Ok(info) => match video::extract_frame(&path, 0.0) {
            Ok(frame) => {
                let mut g = app.lock().unwrap();
                let max = g.config.max_image_dim;
                let (frame, scaled) = media::downscale(frame, max);
                g.video = Some(VideoSession {
                    info: info.clone(),
                    position: 0.0,
                    playing: Arc::new(AtomicBool::new(false)),
                    epoch: Arc::new(AtomicU64::new(0)),
                    poster: Some(frame.clone()),
                });
                g.media = Some(LoadedMedia {
                    path: path.clone(),
                    is_video: true,
                    width: frame.width(),
                    height: frame.height(),
                    notice: if scaled {
                        format!("preview downscaled to {}px", frame.width().max(frame.height()))
                    } else {
                        String::new()
                    },
                    source: frame.clone(),
                    display: frame,
                });
                drop(g);
                clear_current_result(ui, app);
                drop_refresh(ui, app);
            }
            Err(e) => show_error(ui, "Could not decode frame", &format!("{e:#}"), ""),
        },
        Err(e) => show_error(ui, "Could not open video", &format!("{e:#}"), ""),
    }
}

fn bump_video_epoch(app: &SharedApp) -> u64 {
    match app.lock().unwrap().video.as_ref() {
        Some(v) => v.epoch.fetch_add(1, Ordering::SeqCst) + 1,
        None => 0,
    }
}

fn current_video_epoch(app: &SharedApp) -> u64 {
    app.lock()
        .unwrap()
        .video
        .as_ref()
        .map(|v| v.epoch.load(Ordering::SeqCst))
        .unwrap_or(0)
}

fn stop_playing(app: &SharedApp) {
    if let Some(v) = app.lock().unwrap().video.as_ref() {
        v.playing.store(false, Ordering::SeqCst);
    }
}

fn stop_video(ui: &MainWindow, app: &SharedApp) {
    stop_playing(app);
    bump_video_epoch(app);
    ui.global::<Api>().set_video_playing(false);
    let poster = {
        let mut g = app.lock().unwrap();
        let Some(v) = g.video.as_mut() else { return };
        v.position = 0.0;
        v.poster.clone()
    };
    if let Some(frame) = poster {
        apply_frame(app, frame, 0.0);
        refresh_playback(ui, app);
        return;
    }
    request_video_frame(ui, app, 0.0, false);
}

fn apply_frame(app: &SharedApp, frame: image::RgbaImage, seconds: f64) {
    let mut g = app.lock().unwrap();
    if let Some(v) = g.video.as_mut() {
        v.position = seconds;
    }
    let max = g.config.max_image_dim;
    let (frame, scaled) = media::downscale(frame, max);
    if let Some(m) = g.media.as_mut() {
        m.source = frame.clone();
        m.display = frame;
        m.width = m.source.width();
        m.height = m.source.height();
        m.notice = if scaled {
            format!("downscaled to {}px", m.width.max(m.height))
        } else {
            String::new()
        };
    }
}

fn request_video_frame(ui: &MainWindow, app: &SharedApp, seconds: f64, bump: bool) {
    stop_playing(app);
    ui.global::<Api>().set_video_playing(false);
    let epoch = if bump {
        bump_video_epoch(app)
    } else {
        current_video_epoch(app)
    };
    let path = {
        let mut g = app.lock().unwrap();
        let Some(v) = g.video.as_mut() else { return };
        v.position = seconds;
        v.info.path.clone()
    };
    refresh_playback(ui, app);
    let weak = ui.as_weak();
    let app2 = app.clone();
    std::thread::spawn(move || {
        let result = video::extract_frame(&path, seconds);
        let _ = slint::invoke_from_event_loop(move || {
            let Some(ui) = weak.upgrade() else { return };
            if current_video_epoch(&app2) != epoch {
                return;
            }
            match result {
                Ok(frame) => {
                    apply_frame(&app2, frame, seconds);
                    refresh_playback(&ui, &app2);
                }
                Err(e) => show_error(&ui, "Could not show that frame", &e.to_string(), ""),
            }
        });
    });
}

fn seek_video(ui: &MainWindow, app: &SharedApp, pos: f32) {
    let duration = {
        let g = app.lock().unwrap();
        match &g.video {
            Some(v) => v.info.duration,
            None => return,
        }
    };
    request_video_frame(ui, app, (pos as f64) * duration, true);
}

fn step_video(ui: &MainWindow, app: &SharedApp, delta: i32) {
    let next_t = {
        let g = app.lock().unwrap();
        let Some(v) = &g.video else { return };
        let step = g.config.frame_step.max(1) as i64 * delta as i64;
        let frame = v.info.frame_from_time(v.position) as i64 + step;
        let frame = frame.clamp(0, v.info.frame_count.saturating_sub(1) as i64) as u64;
        v.info.time_from_frame(frame)
    };
    request_video_frame(ui, app, next_t, true);
}

fn toggle_play(ui: &MainWindow, app: &SharedApp) {
    let playing_now = app
        .lock()
        .unwrap()
        .video
        .as_ref()
        .map(|v| v.playing.load(Ordering::SeqCst))
        .unwrap_or(false);
    if playing_now {
        stop_playing(app);
        bump_video_epoch(app);
        ui.global::<Api>().set_video_playing(false);
        return;
    }
    let (path, start, fps, flag, epoch) = {
        let g = app.lock().unwrap();
        let Some(v) = &g.video else { return };
        v.playing.store(true, Ordering::SeqCst);
        let epoch = v.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        (
            v.info.path.clone(),
            v.position,
            g.config.preview_fps,
            v.playing.clone(),
            epoch,
        )
    };
    ui.global::<Api>().set_video_playing(true);
    let weak = ui.as_weak();
    let app2 = app.clone();
    let weak_end = ui.as_weak();
    let pending = Arc::new(AtomicBool::new(false));
    std::thread::spawn(move || {
        let result = video::play_loop(path, start, fps, flag.clone(), {
            let pending = pending.clone();
            move |img, t| {
                if pending.swap(true, Ordering::SeqCst) {
                    return;
                }
                let app = app2.clone();
                let weak = weak.clone();
                let pending = pending.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if current_video_epoch(&app) != epoch {
                        pending.store(false, Ordering::SeqCst);
                        return;
                    }
                    apply_frame(&app, img, t);
                    if let Some(ui) = weak.upgrade() {
                        refresh_playback(&ui, &app);
                    }
                    pending.store(false, Ordering::SeqCst);
                });
            }
        });
        flag.store(false, Ordering::SeqCst);
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = weak_end.upgrade() {
                ui.global::<Api>().set_video_playing(false);
                if let Err(e) = result {
                    show_error(&ui, "Playback failed", &e.to_string(), "");
                }
            }
        });
    });
}

fn start_run(ui: &MainWindow, app: &SharedApp, _retry: bool) {
    stop_playing(app);
    let ui_model = ui.global::<Api>().get_model_index();
    let ui_task = ui.global::<Api>().get_task_index();
    let focus = ui.global::<Api>().get_focus_prompt().trim().to_string();
    let mut g = app.lock().unwrap();
    if g.running {
        return;
    }
    let skip_huge = g.skip_huge_warn;
    g.skip_huge_warn = false;
    if g.media.is_none() {
        drop(g);
        show_error(ui, "No media", "Open an image or video frame first.", "");
        return;
    }
    if ui_task >= 0 {
        g.config.last_task = TaskKind::from_index(ui_task);
    }
    g.config.last_focus = focus.clone();
    let items = picker_items(&g.config);
    let idx = if ui_model >= 0 && (ui_model as usize) < items.len() {
        ui_model
    } else {
        current_picker_index(&g.config, &items)
    };
    let Some(item) = items.get(idx.max(0) as usize).cloned() else {
        drop(g);
        show_error(
            ui,
            "No model selected",
            "Pick a vision model in Settings. For Ollama, start ollama serve, refresh the catalog, then choose a vision model.",
            "",
        );
        return;
    };
    g.config.last_provider_id = Some(item.provider_id.clone());
    g.config.last_model = Some(item.model_id.clone());
    let _ = g.config.save(&g.data_dir);
    if !item.vision_ok && item.vision_label == VisionStatus::Unlikely.label() {
        drop(g);
        show_error(
            ui,
            "This model does not support image inputs",
            "Pick a vision-capable model or change provider.",
            "Open the model picker or Settings to choose another model.",
        );
        return;
    }
    let Some(provider) = g.config.provider(&item.provider_id).cloned() else {
        drop(g);
        return;
    };
    let key = match settings::request_api_key(&provider) {
        Ok(k) => k,
        Err(e) => {
            drop(g);
            show_error(ui, "Missing API key", &e.to_string(), "Open Settings, paste the key, and Save.");
            return;
        }
    };
    let media = g.media.as_ref().unwrap();
    let jpeg = match media::encode_jpeg(&media.source, 90) {
        Ok(j) => j,
        Err(e) => {
            drop(g);
            show_error(ui, "Could not encode image", &e.to_string(), "");
            return;
        }
    };
    let pixels = media.width.saturating_mul(media.height);
    let huge = pixels > 3_000_000 || jpeg.len() > 2_500_000;
    if g.config.warn_huge_images && huge && !skip_huge {
        let w = media.width;
        let h = media.height;
        let kb = jpeg.len() / 1024;
        drop(g);
        app.lock().unwrap().pending_confirm = PendingConfirm::SendLargeImage;
        show_confirm(
            ui,
            "Large image",
            &format!("This frame is {w} × {h} ({kb} KB JPEG). Send it to the model?"),
            "You can raise the downscale limit in Settings > Media, or turn off this warning in Run defaults.",
            "Send",
            false,
        );
        return;
    }
    let req = RunRequest {
        model: item.model_id.clone(),
        task: g.config.last_task,
        image_jpeg: jpeg,
        temperature: g.config.default_temperature,
        max_tokens: g.config.default_max_tokens,
        focus,
    };
    let retries = g.config.retries;
    let timeout_secs = provider_http_timeout_secs(provider.kind, g.config.request_timeout_secs);
    let source = media.source.clone();
    let media_ref = if let Some(v) = &g.video {
        MediaRef::Video {
            path: media.path.to_string_lossy().to_string(),
            frame_ts: v.position,
            frame_index: v.info.frame_from_time(v.position),
            width: media.width,
            height: media.height,
        }
    } else {
        MediaRef::Image {
            path: media.path.to_string_lossy().to_string(),
            width: media.width,
            height: media.height,
        }
    };
    let rt = g.rt.clone();
    g.running = true;
    drop(g);

    ui.global::<Api>().set_is_running(true);
    ui.global::<Api>().set_run_enabled(false);

    let weak = ui.as_weak();
    let app2 = app.clone();
    let provider2 = provider.clone();
    rt.spawn(async move {
        let client = providers::ProviderClient::new(Duration::from_secs(timeout_secs));
        let result = match client {
            Ok(c) => c.run(&provider2, &key, &req, retries).await,
            Err(e) => Err(e),
        };
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = weak.upgrade() {
                finish_run(
                    &ui,
                    &app2,
                    result,
                    provider2,
                    item,
                    req.task,
                    source,
                    media_ref,
                    req.focus.clone(),
                    req.max_tokens,
                );
            }
        });
    });
}

fn finish_run(
    ui: &MainWindow,
    app: &SharedApp,
    result: Result<providers::RunResponse>,
    provider: ProviderConfig,
    item: PickerItem,
    task: TaskKind,
    source: image::RgbaImage,
    media_ref: MediaRef,
    focus: String,
    max_tokens: u32,
) {
    app.lock().unwrap().running = false;
    ui.global::<Api>().set_is_running(false);
    match result {
        Err(e) => {
            let detail = e.to_string();
            if providers::looks_like_non_vision(&detail) {
                {
                    let mut g = app.lock().unwrap();
                    if let Some(p) = g.config.provider_mut(&provider.id) {
                        if let Some(m) = p.models.iter_mut().find(|m| m.id == item.model_id) {
                            m.vision = VisionStatus::Unlikely;
                        }
                    }
                    let dir = g.data_dir.clone();
                    let _ = g.config.save(&dir);
                }
                drop_refresh(ui, app);
                show_error(
                    ui,
                    "This model does not support image inputs",
                    "The provider rejected the image. The model is now marked text-only.",
                    &detail,
                );
            } else {
                drop_refresh(ui, app);
                show_error(ui, "Run failed", &detail, "");
            }
        }
        Ok(resp) => {
            let parsed = match parse_task(task, &resp.text) {
                Ok(v) => v,
                Err(e) => ParsedTask {
                    result: TaskResult::Invalid {
                        message: format!("Response was not valid JSON ({e})"),
                    },
                    value: serde_json::json!({"raw": resp.text}),
                    truncated: resp.hit_token_limit || json_looks_incomplete(&resp.text),
                },
            };
            let pretty = serde_json::to_string_pretty(&parsed.value).unwrap_or(resp.text.clone());
            let annotated = annotate::annotate(&source, &parsed.result);
            let entry = providers::lookup_model(&provider, &item.model_id);
            let prices = price_for(provider.kind, &item.model_id, entry);
            let cost = estimate_usd(resp.input_tokens, resp.output_tokens, prices);
            let id = Uuid::new_v4();
            let runs = crate::scoreboard::runs_dir(&app.lock().unwrap().data_dir);
            let _ = std::fs::create_dir_all(&runs);
            let ann_path = runs.join(format!("{id}.png"));
            let _ = image::DynamicImage::ImageRgba8(annotated.clone()).save(&ann_path);
            let rec = RunRecord {
                id,
                timestamp: chrono::Utc::now(),
                provider: provider.kind,
                provider_name: provider.name.clone(),
                model: item.model_id.clone(),
                task,
                media: media_ref,
                input_tokens: resp.input_tokens,
                output_tokens: resp.output_tokens,
                latency_ms: resp.latency_ms,
                est_cost_usd: cost,
                result: parsed.result.clone(),
                raw_response: pretty.clone(),
                annotated_path: Some(ann_path.to_string_lossy().to_string()),
                verdict: Verdict::Pending,
                notes: String::new(),
                focus,
            };
            {
                let mut g = app.lock().unwrap();
                g.last_run = Some(id);
                g.last_raw = pretty.clone();
                g.last_result = Some(parsed.result.clone());
                if let Some(m) = g.media.as_mut() {
                    m.display = annotated.clone();
                }
                g.scoreboard.push(rec.clone());
                let _ = g.scoreboard.save(&g.data_dir);
            }
            apply_run_widgets(ui, &rec);
            drop_refresh(ui, app);
            refresh_scoreboard(ui, app);
            let cutoff = parsed.truncated || resp.hit_token_limit;
            if cutoff {
                let drew_boxes = !matches!(parsed.result, TaskResult::Invalid { .. });
                let message = if drew_boxes {
                    "The model stopped before it finished the JSON. VisionLab drew every complete detection and dropped the truncated tail."
                } else {
                    "The model stopped before it finished the JSON, so the overlay could not be drawn."
                };
                show_error(
                    ui,
                    "Response was cut off",
                    message,
                    &format!(
                        "This run allowed {max_tokens} output tokens (Settings > Run defaults > Max tokens). The provider reported {} output tokens. Raise Max tokens if crowded scenes keep getting cut off.",
                        resp.output_tokens
                    ),
                );
            } else {
                show_toast(ui, "Run recorded. Grade PASS or FAIL when you have reviewed it.");
            }
        }
    }
}

fn grade(weak: &Weak<MainWindow>, app: &SharedApp, verdict: Verdict) {
    let Some(ui) = weak.upgrade() else { return };
    let notes = ui.global::<Api>().get_notes_text().to_string();
    let mut g = app.lock().unwrap();
    let Some(id) = g.last_run else {
        drop(g);
        show_error(&ui, "Nothing to grade", "Run a test first.", "");
        return;
    };
    if let Some(rec) = g.scoreboard.get_mut(id) {
        rec.verdict = verdict;
        rec.notes = notes;
    }
    let _ = g.scoreboard.save(&g.data_dir);
    drop(g);
    ui.global::<Api>().set_last_verdict(verdict.as_str().into());
    refresh_scoreboard(&ui, app);
    show_toast(
        &ui,
        if verdict == Verdict::Pass {
            "PASS recorded."
        } else {
            "FAIL recorded."
        },
    );
}

fn export_result(ui: &MainWindow, app: &SharedApp) {
    let g = app.lock().unwrap();
    let Some(id) = g.last_run else {
        drop(g);
        show_error(ui, "Nothing to export", "Run a test first.", "");
        return;
    };
    let Some(rec) = g.scoreboard.runs.iter().find(|r| r.id == id).cloned() else {
        drop(g);
        return;
    };
    let annotated = g.media.as_ref().map(|m| m.display.clone());
    drop(g);
    let Some(dir) = rfd::FileDialog::new().set_title("Export result").pick_folder() else {
        return;
    };
    if let Some(img) = annotated {
        let _ = image::DynamicImage::ImageRgba8(img).save(dir.join("annotated.png"));
    }
    let _ = std::fs::write(dir.join("result.json"), serde_json::to_string_pretty(&rec).unwrap_or_default());
    show_toast(ui, "Exported annotated.png and result.json.");
}

fn refresh_settings(ui: &MainWindow, app: &SharedApp) {
    let g = app.lock().unwrap();
    let rows: Vec<ProviderRow> = g
        .config
        .providers
        .iter()
        .map(|p| ProviderRow {
            id: p.id.clone().into(),
            name: p.name.clone().into(),
            kind: p.kind.as_str().into(),
            model: p.default_model.clone().into(),
            has_key: settings::provider_auth_ok(p),
            status: settings::provider_status_label(p).into(),
        })
        .collect();
    ui.global::<Api>().set_providers(ModelRc::new(VecModel::from(rows)));
    ui.global::<Api>().set_show_connect_banner(!settings::any_provider_ready(&g.config));
}

fn load_provider_form(ui: &MainWindow, app: &SharedApp, idx: i32) {
    let g = app.lock().unwrap();
    let Some(p) = g.config.providers.get(idx as usize) else {
        ui.global::<Api>().set_selected_provider(-1);
        return;
    };
    let api = ui.global::<Api>();
    api.set_selected_provider(idx);
    api.set_edit_name(p.name.clone().into());
    api.set_edit_kind(p.kind.to_index());
    api.set_edit_base_url(p.base_url.clone().into());
    api.set_edit_default_model(p.default_model.clone().into());
    let headers_text = if p.extra_headers.is_empty() {
        "{}".to_string()
    } else {
        serde_json::to_string_pretty(&p.extra_headers).unwrap_or_else(|_| "{}".into())
    };
    api.set_edit_headers(headers_text.into());
    api.set_edit_api_key(if settings::has_api_key(&p.id) {
        "••••••••".into()
    } else {
        "".into()
    });
    api.set_connection_status("".into());
    api.set_settings_dirty(false);
    drop(g);
    refresh_catalog_view(ui, app);
}

fn save_settings(ui: &MainWindow, app: &SharedApp, announce: bool) -> bool {
    let api = ui.global::<Api>();
    let idx = api.get_selected_provider();
    let mut g = app.lock().unwrap();
    g.config.preview_fps = api.get_preview_fps().max(5) as u32;
    g.config.frame_step = api.get_frame_step().max(1) as u32;
    g.config.max_image_dim = api.get_max_dim().max(256) as u32;
    g.config.request_timeout_secs = api.get_timeout_secs().max(10) as u64;
    g.config.retries = api.get_retries().max(0) as u32;
    g.config.default_max_tokens = api.get_max_tokens().max(256) as u32;
    g.config.default_temperature = api.get_temperature().clamp(0.0, 2.0);
    g.config.warn_huge_images = api.get_warn_huge();
    g.config.restore_last_session = api.get_restore_session();
    let mut pending_key: Option<(String, String)> = None;
    if idx >= 0 {
        if let Some(p) = g.config.providers.get_mut(idx as usize) {
            p.name = api.get_edit_name().to_string();
            p.kind = ProviderKind::from_index(api.get_edit_kind());
            p.base_url = api.get_edit_base_url().to_string();
            p.default_model = api.get_edit_default_model().to_string();
            let headers = api.get_edit_headers().to_string();
            let headers_trim = headers.trim();
            if headers_trim.is_empty() {
                p.extra_headers.clear();
            } else {
                match serde_json::from_str::<std::collections::HashMap<String, String>>(headers_trim)
                {
                    Ok(map) => p.extra_headers = map,
                    Err(e) => {
                        drop(g);
                        show_error(
                            ui,
                            "Invalid extra headers",
                            "Extra headers must be a JSON object of string keys and values.",
                            &e.to_string(),
                        );
                        return false;
                    }
                }
            }
            let key = api.get_edit_api_key().to_string();
            let pid = p.id.clone();
            if !settings::is_key_placeholder(&key) {
                pending_key = Some((pid, key));
            }
        }
    }
    if let Some((pid, key)) = pending_key {
        if let Err(e) = settings::set_api_key(&pid, &key) {
            drop(g);
            tracing::error!("API key store failed: {e}");
            show_error(ui, "Could not store API key", &e.to_string(), "");
            return false;
        }
        tracing::info!("API key stored for provider {pid}");
    }
    if let Err(e) = g.config.save(&g.data_dir) {
        drop(g);
        show_error(ui, "Could not save settings", &e.to_string(), "");
        return false;
    }
    drop(g);
    api.set_settings_dirty(false);
    refresh_settings(ui, app);
    drop_refresh(ui, app);
    if idx >= 0 && api.get_settings_section() == 0 {
        load_provider_form(ui, app, idx);
    }
    if announce {
        show_toast(ui, "Settings saved.");
    }
    true
}

fn test_connection(ui: &MainWindow, app: &SharedApp) {
    if !save_settings(ui, app, false) {
        return;
    }
    let idx = ui.global::<Api>().get_selected_provider();
    let g = app.lock().unwrap();
    let Some(p) = g.config.providers.get(idx as usize).cloned() else {
        drop(g);
        show_error(ui, "No provider selected", "Select a provider first.", "");
        return;
    };
    let key = match settings::request_api_key(&p) {
        Ok(k) => k,
        Err(e) => {
            drop(g);
            ui.global::<Api>().set_connection_ok(false);
            ui.global::<Api>().set_connection_status(e.to_string().into());
            return;
        }
    };
    let timeout_secs = provider_http_timeout_secs(p.kind, 30);
    let rt = g.rt.clone();
    drop(g);
    ui.global::<Api>().set_connection_status("Testing…".into());
    let weak = ui.as_weak();
    let app2 = app.clone();
    rt.spawn(async move {
        let client = providers::ProviderClient::new(Duration::from_secs(timeout_secs));
        let res = match client {
            Ok(c) => c.test_connection(&p, &key).await,
            Err(e) => Err(e),
        };
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = weak.upgrade() {
                match res {
                    Ok(msg) => {
                        ui.global::<Api>().set_connection_ok(true);
                        ui.global::<Api>().set_connection_status(msg.into());
                        refresh_models_async(&ui, &app2);
                    }
                    Err(e) => {
                        ui.global::<Api>().set_connection_ok(false);
                        ui.global::<Api>().set_connection_status(e.to_string().into());
                    }
                }
            }
        });
    });
}

fn refresh_models_async(ui: &MainWindow, app: &SharedApp) {
    let idx = ui.global::<Api>().get_selected_provider();
    let g = app.lock().unwrap();
    let Some(p) = g.config.providers.get(idx as usize).cloned() else {
        drop(g);
        show_error(ui, "No provider selected", "Select a provider first.", "");
        return;
    };
    let key = match settings::request_api_key(&p) {
        Ok(k) => k,
        Err(e) => {
            drop(g);
            show_error(ui, "Missing API key", &e.to_string(), "For Ollama, start the server and refresh. Cloud providers need a saved key.");
            return;
        }
    };
    let timeout_secs = provider_http_timeout_secs(p.kind, 30);
    let rt = g.rt.clone();
    drop(g);
    show_toast(ui, "Refreshing model catalog…");
    let weak = ui.as_weak();
    let app2 = app.clone();
    rt.spawn(async move {
        let client = providers::ProviderClient::new(Duration::from_secs(timeout_secs));
        let res = match client {
            Ok(c) => c.list_models(&p, &key).await,
            Err(e) => Err(e),
        };
        let pid = p.id.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = weak.upgrade() {
                match res {
                    Ok(models) => {
                        {
                            let mut g = app2.lock().unwrap();
                            if let Some(pr) = g.config.provider_mut(&pid) {
                                pr.models = models;
                            }
                            let dir = g.data_dir.clone();
                            let _ = g.config.save(&dir);
                        }
                        refresh_catalog_view(&ui, &app2);
                        drop_refresh(&ui, &app2);
                        show_toast(&ui, "Model list refreshed.");
                    }
                    Err(e) => show_error(&ui, "Catalog failed", &e.to_string(), ""),
                }
            }
        });
    });
}

fn refresh_catalog_view(ui: &MainWindow, app: &SharedApp) {
    let idx = ui.global::<Api>().get_selected_provider();
    let g = app.lock().unwrap();
    let models = g
        .config
        .providers
        .get(idx as usize)
        .map(|p| p.models.clone())
        .unwrap_or_default();
    let rows: Vec<ModelRow> = models
        .iter()
        .map(|m| ModelRow {
            id: m.id.clone().into(),
            name: m.name.clone().into(),
            vision_label: m.vision.label().into(),
            vision_ok: m.vision.is_ok(),
        })
        .collect();
    ui.global::<Api>().set_models(ModelRc::new(VecModel::from(rows)));
}

fn use_catalog_model(ui: &MainWindow, app: &SharedApp, idx: i32) {
    let pidx = ui.global::<Api>().get_selected_provider();
    let mut g = app.lock().unwrap();
    let Some(p) = g.config.providers.get(pidx as usize) else {
        drop(g);
        show_error(ui, "No provider selected", "Select a provider first.", "");
        return;
    };
    let Some(m) = p.models.get(idx as usize).cloned() else {
        drop(g);
        show_error(ui, "No model selected", "Refresh the catalog, then pick a model.", "");
        return;
    };
    let pid = p.id.clone();
    g.config.last_provider_id = Some(pid.clone());
    g.config.last_model = Some(m.id.clone());
    if let Some(p) = g.config.provider_mut(&pid) {
        p.default_model = m.id.clone();
    }
    let _ = g.config.save(&g.data_dir);
    drop(g);
    ui.global::<Api>().set_edit_default_model(m.id.into());
    drop_refresh(ui, app);
    refresh_settings(ui, app);
    show_toast(ui, "Default model updated.");
}

fn probe_vision(ui: &MainWindow, app: &SharedApp) {
    let idx = ui.global::<Api>().get_selected_provider();
    let g = app.lock().unwrap();
    let Some(p) = g.config.providers.get(idx as usize).cloned() else {
        drop(g);
        show_error(ui, "No provider selected", "Select a provider first.", "");
        return;
    };
    let model = ui.global::<Api>().get_edit_default_model().to_string();
    if model.is_empty() {
        drop(g);
        show_error(ui, "No model", "Set a default model, then probe.", "");
        return;
    }
    let key = match settings::request_api_key(&p) {
        Ok(k) => k,
        Err(e) => {
            drop(g);
            show_error(ui, "Missing API key", &e.to_string(), "");
            return;
        }
    };
    let rt = g.rt.clone();
    let retries = g.config.retries;
    let timeout_secs = provider_http_timeout_secs(p.kind, 60);
    drop(g);
    show_toast(ui, "Probing vision support…");
    let weak = ui.as_weak();
    let app2 = app.clone();
    rt.spawn(async move {
        let client = providers::ProviderClient::new(Duration::from_secs(timeout_secs));
        let res = match client {
            Ok(c) => c.probe_vision(&p, &key, &model, retries).await,
            Err(e) => Err(e),
        };
        let pid = p.id.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = weak.upgrade() {
                match res {
                    Ok(ok) => {
                        let status = if ok {
                            VisionStatus::Confirmed
                        } else {
                            VisionStatus::Unlikely
                        };
                        {
                            let mut g = app2.lock().unwrap();
                            if let Some(pr) = g.config.provider_mut(&pid) {
                                if let Some(m) = pr.models.iter_mut().find(|m| m.id == model) {
                                    m.vision = status;
                                } else {
                                    pr.models.push(ModelEntry {
                                        id: model.clone(),
                                        name: model.clone(),
                                        vision: status,
                                        input_price_per_m: None,
                                        output_price_per_m: None,
                                    });
                                }
                            }
                            let dir = g.data_dir.clone();
                            let _ = g.config.save(&dir);
                        }
                        refresh_catalog_view(&ui, &app2);
                        drop_refresh(&ui, &app2);
                        show_toast(
                            &ui,
                            if ok {
                                "Probe passed: this model accepted an image."
                            } else {
                                "Probe failed: this model does not appear to accept images."
                            },
                        );
                    }
                    Err(e) => show_error(&ui, "Probe failed", &e.to_string(), ""),
                }
            }
        });
    });
}

fn refresh_scoreboard(ui: &MainWindow, app: &SharedApp) {
    let g = app.lock().unwrap();
    let rows = g.scoreboard.filtered(&g.filter, g.task_filter);
    let (n, pass, lat, cost) = g.scoreboard.stats(&rows);
    let view: Vec<ScoreboardRow> = rows
        .iter()
        .map(|r| ScoreboardRow {
            id: r.id.to_string().into(),
            time: r.timestamp.format("%Y-%m-%d %H:%M").to_string().into(),
            provider: r.provider_name.clone().into(),
            model: r.model.clone().into(),
            task: r.task.label().into(),
            media: r.media.display().into(),
            latency: format!("{:.2}s", r.latency_ms as f64 / 1000.0).into(),
            tokens: format!("{}/{}", r.input_tokens, r.output_tokens).into(),
            cost: format_usd(r.est_cost_usd).into(),
            verdict: r.verdict.as_str().into(),
            notes: r.notes.clone().into(),
        })
        .collect();
    drop(g);
    let api = ui.global::<Api>();
    api.set_scoreboard_rows(ModelRc::new(VecModel::from(view)));
    api.set_stat_runs(n as i32);
    api.set_stat_pass_rate(pass.into());
    api.set_stat_avg_latency(lat.into());
    api.set_stat_total_cost(cost.into());
}

fn push_json_preview(api: &Api, full: &str) {
    // TextEdit layouts every glyph. Segmentation polygons easily reach tens of
    // thousands of characters and freeze scrolling and tab switches.
    const LIMIT: usize = 8_192;
    if full.len() <= LIMIT {
        api.set_raw_json(full.into());
        api.set_raw_json_clipped(false);
        return;
    }
    let mut end = LIMIT;
    while end > 0 && !full.is_char_boundary(end) {
        end -= 1;
    }
    api.set_raw_json(
        format!(
            "{}\n\n... truncated for the UI ({} characters total). Use Copy for the full JSON.",
            &full[..end],
            full.len()
        )
        .into(),
    );
    api.set_raw_json_clipped(true);
}

fn apply_run_widgets(ui: &MainWindow, rec: &RunRecord) {
    let api = ui.global::<Api>();
    push_json_preview(&api, &rec.raw_response);
    api.set_notes_text(rec.notes.clone().into());
    api.set_last_verdict(rec.verdict.as_str().into());
    api.set_latency_text(format!("{:.2}s", rec.latency_ms as f64 / 1000.0).into());
    api.set_tokens_text(format!("{} / {}", rec.input_tokens, rec.output_tokens).into());
    api.set_cost_text(format_usd(rec.est_cost_usd).into());
    api.set_result_summary(rec.result.summary().into());
    let legend: Vec<LegendItem> = rec
        .result
        .legend()
        .into_iter()
        .map(|(label, count)| LegendItem {
            color: slint_color(annotate::color_for(&label)),
            label: label.into(),
            count,
        })
        .collect();
    api.set_legend(ModelRc::new(VecModel::from(legend)));
    api.set_has_result(true);
    api.set_grade_enabled(true);
}

fn open_row(ui: &MainWindow, app: &SharedApp, idx: i32) {
    let g = app.lock().unwrap();
    let rows = g.scoreboard.filtered(&g.filter, g.task_filter);
    let Some(rec) = rows.get(idx as usize).map(|r| (*r).clone()) else {
        return;
    };
    let max_dim = g.config.max_image_dim;
    drop(g);

    let annotated = rec
        .annotated_path
        .as_ref()
        .and_then(|p| image::open(p).ok())
        .map(|img| img.to_rgba8());

    stop_playing(app);
    app.lock().unwrap().video = None;

    let mut source_img = None;
    match &rec.media {
        MediaRef::Image { path, .. } => {
            if let Ok(img) = media::load_image(Path::new(path)) {
                source_img = Some(media::downscale(img, max_dim).0);
            }
        }
        MediaRef::Video {
            path, frame_ts, ..
        } => {
            let p = PathBuf::from(path);
            if let Ok(info) = video::probe(&p) {
                if let Ok(frame) = video::extract_frame(&p, *frame_ts) {
                    let frame = media::downscale(frame, max_dim).0;
                    source_img = Some(frame.clone());
                    let mut g = app.lock().unwrap();
                    g.video = Some(VideoSession {
                        info,
                        position: *frame_ts,
                        playing: Arc::new(AtomicBool::new(false)),
                        epoch: Arc::new(AtomicU64::new(0)),
                        poster: (*frame_ts <= 0.05).then_some(frame),
                    });
                }
            }
        }
    }

    let source = source_img.clone().or_else(|| annotated.clone());
    let display = source_img
        .as_ref()
        .map(|src| annotate::annotate(src, &rec.result))
        .or(annotated);
    if let (Some(source), Some(display)) = (source, display) {
        let path = match &rec.media {
            MediaRef::Image { path, .. } | MediaRef::Video { path, .. } => PathBuf::from(path),
        };
        let mut g = app.lock().unwrap();
        g.media = Some(LoadedMedia {
            path,
            is_video: matches!(rec.media, MediaRef::Video { .. }),
            width: source.width(),
            height: source.height(),
            notice: String::new(),
            source,
            display,
        });
        g.last_run = Some(rec.id);
        g.last_raw = rec.raw_response.clone();
        g.last_result = Some(rec.result.clone());
    } else {
        let mut g = app.lock().unwrap();
        g.last_run = Some(rec.id);
        g.last_raw = rec.raw_response.clone();
        g.last_result = Some(rec.result.clone());
        drop(g);
        show_error(
            ui,
            "Media missing",
            "The original file and annotated image could not be loaded. JSON is still available.",
            "",
        );
    }

    apply_run_widgets(ui, &rec);
    ui.global::<Api>().set_screen(1);
    drop_refresh(ui, app);
}

fn export_board(weak: &Weak<MainWindow>, app: &SharedApp, csv: bool) {
    let Some(ui) = weak.upgrade() else { return };
    let g = app.lock().unwrap();
    let rows = g.scoreboard.filtered(&g.filter, g.task_filter);
    let payload = if csv {
        g.scoreboard.to_csv(&rows)
    } else {
        serde_json::to_string_pretty(&rows.iter().map(|r| (*r).clone()).collect::<Vec<_>>()).unwrap_or_default()
    };
    drop(g);
    let filter = if csv { ("CSV", "csv") } else { ("JSON", "json") };
    if let Some(path) = rfd::FileDialog::new()
        .add_filter(filter.0, &[filter.1])
        .set_file_name(if csv { "scoreboard.csv" } else { "scoreboard.json" })
        .save_file()
    {
        if let Err(e) = std::fs::write(&path, payload) {
            show_error(&ui, "Export failed", &e.to_string(), "");
        } else {
            show_toast(&ui, "Scoreboard exported.");
        }
    }
}

fn export_config(weak: &Weak<MainWindow>, app: &SharedApp) {
    let Some(ui) = weak.upgrade() else { return };
    let cfg = app.lock().unwrap().config.clone();
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("JSON", &["json"])
        .set_file_name("visionlab-config.json")
        .save_file()
    {
        match serde_json::to_string_pretty(&cfg) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    show_error(&ui, "Export failed", &e.to_string(), "");
                } else {
                    show_toast(&ui, "Config exported (keys are not included).");
                }
            }
            Err(e) => show_error(&ui, "Export failed", &e.to_string(), ""),
        }
    }
}

fn import_config(weak: &Weak<MainWindow>, app: &SharedApp) {
    let Some(ui) = weak.upgrade() else { return };
    let Some(path) = rfd::FileDialog::new().add_filter("JSON", &["json"]).pick_file() else {
        return;
    };
    match std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Config>(&s).ok()) {
        Some(cfg) => {
            let mut g = app.lock().unwrap();
            g.config = cfg;
            let _ = g.config.save(&g.data_dir);
            drop(g);
            refresh_all(&ui, app);
            let n = app.lock().unwrap().config.providers.len();
            if n > 0 {
                load_provider_form(&ui, app, 0);
            }
            show_toast(&ui, "Config imported. Re-enter API keys if needed.");
        }
        None => show_error(&ui, "Import failed", "The file is not a valid VisionLab config.", ""),
    }
}

fn backup_scoreboard(weak: &Weak<MainWindow>, app: &SharedApp) {
    let Some(ui) = weak.upgrade() else { return };
    let g = app.lock().unwrap();
    let json = serde_json::to_string_pretty(&g.scoreboard).unwrap_or_default();
    drop(g);
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("JSON", &["json"])
        .set_file_name("scoreboard-backup.json")
        .save_file()
    {
        if std::fs::write(&path, json).is_ok() {
            show_toast(&ui, "Scoreboard backup saved.");
        }
    }
}

fn restore_scoreboard(weak: &Weak<MainWindow>, app: &SharedApp) {
    let Some(ui) = weak.upgrade() else { return };
    let Some(path) = rfd::FileDialog::new().add_filter("JSON", &["json"]).pick_file() else {
        return;
    };
    match std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Scoreboard>(&s).ok()) {
        Some(sb) => {
            let mut g = app.lock().unwrap();
            g.scoreboard = sb;
            let _ = g.scoreboard.save(&g.data_dir);
            drop(g);
            refresh_scoreboard(&ui, app);
            show_toast(&ui, "Scoreboard restored.");
        }
        None => show_error(&ui, "Restore failed", "The file is not a valid scoreboard backup.", ""),
    }
}

fn confirm_ok(ui: &MainWindow, app: &SharedApp) {
    ui.global::<Api>().set_confirm_open(false);
    let pending = app.lock().unwrap().pending_confirm.clone();
    app.lock().unwrap().pending_confirm = PendingConfirm::None;
    match pending {
        PendingConfirm::None => {}
        PendingConfirm::ClearScoreboard => {
            let mut g = app.lock().unwrap();
            g.scoreboard.runs.clear();
            let _ = g.scoreboard.save(&g.data_dir);
            drop(g);
            refresh_scoreboard(ui, app);
            show_toast(ui, "Scoreboard cleared.");
        }
        PendingConfirm::DeleteProvider => {
            let idx = ui.global::<Api>().get_selected_provider();
            let mut g = app.lock().unwrap();
            if idx >= 0 && (idx as usize) < g.config.providers.len() {
                let p = g.config.providers.remove(idx as usize);
                settings::delete_api_key(&p.id);
            }
            let remaining = g.config.providers.len();
            let _ = g.config.save(&g.data_dir);
            drop(g);
            refresh_settings(ui, app);
            drop_refresh(ui, app);
            if remaining > 0 {
                let next = (idx as usize).min(remaining - 1) as i32;
                load_provider_form(ui, app, next);
            } else {
                ui.global::<Api>().set_selected_provider(-1);
            }
            show_toast(ui, "Provider deleted.");
        }
        PendingConfirm::LeaveSettings { target } => {
            ui.global::<Api>().set_settings_dirty(false);
            ui.global::<Api>().set_screen(target);
        }
        PendingConfirm::SendLargeImage => {
            app.lock().unwrap().skip_huge_warn = true;
            start_run(ui, app, false);
        }
    }
}

fn json_looks_incomplete(raw: &str) -> bool {
    let t = raw.trim();
    if t.is_empty() {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(t).is_err() && (t.contains('{') || t.contains('['))
}

fn show_error(ui: &MainWindow, title: &str, message: &str, detail: &str) {
    let api = ui.global::<Api>();
    api.set_error_title(title.into());
    api.set_error_message(message.into());
    api.set_error_detail(detail.into());
    api.set_error_action("OK".into());
    api.set_error_open(true);
}

fn show_confirm(ui: &MainWindow, title: &str, message: &str, detail: &str, action: &str, danger: bool) {
    let api = ui.global::<Api>();
    api.set_confirm_title(title.into());
    api.set_confirm_message(message.into());
    api.set_confirm_detail(detail.into());
    api.set_confirm_action(action.into());
    api.set_confirm_danger(danger);
    api.set_confirm_open(true);
}

fn show_toast(ui: &MainWindow, text: &str) {
    let api = ui.global::<Api>();
    api.set_toast_text(text.into());
    api.set_toast_visible(true);
    let weak = ui.as_weak();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2800));
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = weak.upgrade() {
                ui.global::<Api>().set_toast_visible(false);
            }
        });
    });
}
