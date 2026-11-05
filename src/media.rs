//! Image load, downscale, JPEG encode, and Slint pixel conversion.

use anyhow::{Context, Result};
use image::{DynamicImage, RgbaImage};
use std::io::Cursor;
use std::path::Path;

pub fn load_image(path: &Path) -> Result<RgbaImage> {
    // Guess from magic bytes, not the extension. Annotation sets often store
    // WebP or PNG data under a .jpg name, and image::open() would decode those
    // with the JPEG decoder and fail.
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("read {}", path.display()))?;
    let detected = reader.format();
    let img = reader.decode().with_context(|| {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("unknown");
        match detected {
            Some(fmt) => format!("decode {} (file is {fmt:?}, extension .{ext})", path.display()),
            None => format!("decode {} (unrecognized image, extension .{ext})", path.display()),
        }
    })?;
    Ok(img.to_rgba8())
}

pub fn downscale(img: RgbaImage, max_dim: u32) -> (RgbaImage, bool) {
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    if max_dim == 0 || longest <= max_dim {
        return (img, false);
    }
    let scale = max_dim as f32 / longest as f32;
    let nw = ((w as f32) * scale).max(1.0).round() as u32;
    let nh = ((h as f32) * scale).max(1.0).round() as u32;
    let resized = DynamicImage::ImageRgba8(img).resize_exact(nw, nh, image::imageops::FilterType::Triangle);
    (resized.to_rgba8(), true)
}

pub fn encode_jpeg(img: &RgbaImage, quality: u8) -> Result<Vec<u8>> {
    let rgb = DynamicImage::ImageRgba8(img.clone()).to_rgb8();
    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
        encoder
            .encode(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)
            .context("encode jpeg")?;
    }
    Ok(buf)
}

pub fn rgb24_to_rgba(data: &[u8], width: u32, height: u32) -> Result<RgbaImage> {
    let expected = width as usize * height as usize * 3;
    if data.len() < expected {
        anyhow::bail!("frame buffer is smaller than {width}x{height}");
    }
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for chunk in data[..expected].chunks_exact(3) {
        rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
    }
    RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| anyhow::anyhow!("could not build RGBA frame {width}x{height}"))
}

pub fn to_slint(img: &RgbaImage) -> slint::Image {
    let (w, h) = (img.width(), img.height());
    let mut buf = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(w, h);
    buf.make_mut_bytes().copy_from_slice(img.as_raw());
    slint::Image::from_rgba8(buf)
}

pub fn file_stem(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("media")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_webp_bytes_saved_as_jpg() {
        let path = Path::new(r"L:\AI_ANNOTATION\not label\car\car_interection.jpg");
        if !path.exists() {
            return;
        }
        let img = load_image(path).expect("WebP data under a .jpg name should still decode");
        assert!(img.width() > 0 && img.height() > 0);
    }
}
