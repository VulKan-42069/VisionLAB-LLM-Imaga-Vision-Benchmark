fn main() {
    embed_windows_icon();
    let config = slint_build::CompilerConfiguration::new();
    slint_build::compile_with_config("ui/main.slint", config).unwrap();
}

fn embed_windows_icon() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "windows" {
        return;
    }
    let mut res = winresource::WindowsResource::new();
    res.set_icon("assets/app.ico");
    let ver = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.1.0".into());
    res.set("ProductName", "VisionLab");
    res.set("FileDescription", "VisionLab - LLM Vision Benchmark Studio");
    res.set("LegalCopyright", "VisionLab");
    res.set("ProductVersion", &ver);
    res.set("FileVersion", &ver);
    res.compile().expect("embed Windows app icon");
}
