//! Overlay compositor: boxes, polygons, keypoints, OCR regions, classification badge.

use ab_glyph::{FontRef, PxScale};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut};
use imageproc::rect::Rect;
use once_cell::sync::Lazy;

use crate::tasks::{BBox, TaskResult};

static FONT: Lazy<Option<Vec<u8>>> = Lazy::new(|| {
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ];
    for p in CANDIDATES {
        if let Ok(bytes) = std::fs::read(p) {
            return Some(bytes);
        }
    }
    None
});

pub const PALETTE: [[u8; 4]; 8] = [
    [167, 139, 250, 255],
    [96, 165, 250, 255],
    [52, 211, 153, 255],
    [251, 191, 36, 255],
    [248, 113, 113, 255],
    [244, 114, 182, 255],
    [34, 211, 238, 255],
    [163, 230, 53, 255],
];

pub fn color_for(label: &str) -> [u8; 4] {
    let mut h = 0u64;
    for b in label.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u64);
    }
    PALETTE[(h as usize) % PALETTE.len()]
}

pub fn slint_color(c: [u8; 4]) -> slint::Color {
    slint::Color::from_argb_u8(c[3], c[0], c[1], c[2])
}

pub fn annotate(src: &RgbaImage, result: &TaskResult) -> RgbaImage {
    let mut img = src.clone();
    let map = CoordMap::from_result(&img, result);
    match result {
        TaskResult::Classification { label, confidence, .. } => {
            let c = color_for(label);
            let text = format!("{label}  {:.0}%", confidence * 100.0);
            draw_banner(&mut img, &text, c);
        }
        TaskResult::Detection { objects } => {
            for o in objects {
                draw_box(
                    &mut img,
                    &map,
                    &o.bbox,
                    color_for(&o.label),
                    &format!("{} {:.0}%", o.label, o.confidence * 100.0),
                );
            }
        }
        TaskResult::Segmentation { instances } => {
            for inst in instances {
                let c = color_for(&inst.label);
                for ring in &inst.polygons {
                    fill_polygon(&mut img, &map, ring, c, 0.28);
                    stroke_polygon(&mut img, &map, ring, c);
                }
                if let Some(ring) = inst.polygons.first() {
                    if let Some([x, y]) = ring.first() {
                        draw_label_at(&mut img, &map, *x, *y, &inst.label, c);
                    }
                }
            }
        }
        TaskResult::Keypoints { keypoints, skeleton } => {
            for edge in skeleton {
                let a = keypoints.get(edge[0] as usize);
                let b = keypoints.get(edge[1] as usize);
                if let (Some(a), Some(b)) = (a, b) {
                    let (x1, y1) = map.xy(&img, a.point[0], a.point[1]);
                    let (x2, y2) = map.xy(&img, b.point[0], b.point[1]);
                    draw_line_segment_mut(
                        &mut img,
                        (x1 as f32, y1 as f32),
                        (x2 as f32, y2 as f32),
                        Rgba(color_for("skeleton")),
                    );
                }
            }
            for kp in keypoints {
                let c = color_for(&kp.label);
                let (x, y) = map.xy(&img, kp.point[0], kp.point[1]);
                draw_filled_circle_mut(&mut img, (x as i32, y as i32), 5, Rgba(c));
                draw_label_px(&mut img, x.saturating_add(8), y.saturating_sub(8), &kp.label, c);
            }
        }
        TaskResult::Ocr { lines, .. } => {
            for line in lines {
                if let Some(bbox) = &line.bbox {
                    draw_box(&mut img, &map, bbox, color_for("text"), &line.text);
                }
            }
        }
        TaskResult::Invalid { message } => {
            let text: String = message.chars().take(64).collect();
            draw_banner(&mut img, &text, color_for("error"));
        }
    }
    img
}

struct CoordMap {
    sx: f32,
    sy: f32,
}

impl CoordMap {
    fn from_result(img: &RgbaImage, result: &TaskResult) -> Self {
        let extent = coord_extent(result);
        Self::from_extent(extent.max_x, extent.max_y, img.width(), img.height())
    }

    fn from_extent(max_x: f32, max_y: f32, img_w: u32, img_h: u32) -> Self {
        let w = img_w as f32;
        let h = img_h as f32;
        let m = max_x.max(max_y);
        // One scale for the whole result. Qwen-style models use a 0..1000 canvas
        // on each axis, including on portrait images, and they often overflow 1000
        // a little. Treating that overflow as native pixels of the long side was
        // drawing every box too small and toward the top-left.
        if !m.is_finite() || m <= 2.0 {
            return Self { sx: w, sy: h };
        }
        if m <= 100.5 {
            return Self { sx: w / 100.0, sy: h / 100.0 };
        }

        let native_x = max_x >= w * 0.82 && max_x <= w * 1.12;
        let native_y = max_y >= h * 0.82 && max_y <= h * 1.12;
        let qwen_x = near_thousand(max_x) && (max_x - 1000.0).abs() <= (max_x - w).abs();
        let qwen_y = near_thousand(max_y) && (max_y - 1000.0).abs() <= (max_y - h).abs();
        if (qwen_x || qwen_y) && !(native_x && native_y) {
            return Self {
                sx: w / 1000.0,
                sy: h / 1000.0,
            };
        }
        if native_x && native_y {
            return Self { sx: 1.0, sy: 1.0 };
        }
        if let Some(space) = snap_space(m) {
            return Self {
                sx: w / space,
                sy: h / space,
            };
        }
        Self { sx: 1.0, sy: 1.0 }
    }

    fn xy(&self, img: &RgbaImage, x: f32, y: f32) -> (u32, u32) {
        let max_x = img.width().saturating_sub(1) as f32;
        let max_y = img.height().saturating_sub(1) as f32;
        let px = (x * self.sx).round().clamp(0.0, max_x) as u32;
        let py = (y * self.sy).round().clamp(0.0, max_y) as u32;
        (px, py)
    }
}

fn near_thousand(v: f32) -> bool {
    (700.0..=1400.0).contains(&v)
}

fn snap_space(max_abs: f32) -> Option<f32> {
    const SPACES: [f32; 11] = [
        224.0, 256.0, 336.0, 384.0, 448.0, 512.0, 640.0, 768.0, 1024.0, 1280.0, 2048.0,
    ];
    SPACES
        .iter()
        .copied()
        .filter(|s| max_abs >= *s * 0.82 && max_abs <= *s * 1.05)
        .min_by(|a, b| (a - max_abs).abs().total_cmp(&(b - max_abs).abs()))
}

struct CoordExtent {
    max_x: f32,
    max_y: f32,
}

fn coord_extent(result: &TaskResult) -> CoordExtent {
    let mut e = CoordExtent { max_x: 0.0, max_y: 0.0 };
    let bump = |e: &mut CoordExtent, x: f32, y: f32| {
        if x.is_finite() {
            e.max_x = e.max_x.max(x.abs());
        }
        if y.is_finite() {
            e.max_y = e.max_y.max(y.abs());
        }
    };
    match result {
        TaskResult::Detection { objects } => {
            for o in objects {
                bump(&mut e, o.bbox.x, o.bbox.y);
                bump(&mut e, o.bbox.x + o.bbox.w, o.bbox.y + o.bbox.h);
            }
        }
        TaskResult::Segmentation { instances } => {
            for inst in instances {
                for ring in &inst.polygons {
                    for p in ring {
                        bump(&mut e, p[0], p[1]);
                    }
                }
            }
        }
        TaskResult::Keypoints { keypoints, .. } => {
            for kp in keypoints {
                bump(&mut e, kp.point[0], kp.point[1]);
            }
        }
        TaskResult::Ocr { lines, .. } => {
            for line in lines {
                if let Some(b) = &line.bbox {
                    bump(&mut e, b.x, b.y);
                    bump(&mut e, b.x + b.w, b.y + b.h);
                }
            }
        }
        TaskResult::Classification { .. } | TaskResult::Invalid { .. } => {}
    }
    e
}

fn draw_box(img: &mut RgbaImage, map: &CoordMap, bbox: &BBox, color: [u8; 4], label: &str) {
    let (x, y) = map.xy(img, bbox.x, bbox.y);
    let (x2, y2) = map.xy(img, bbox.x + bbox.w, bbox.y + bbox.h);
    let left = x.min(x2);
    let top = y.min(y2);
    let w = x.abs_diff(x2).max(1);
    let h = y.abs_diff(y2).max(1);
    let rect = Rect::at(left as i32, top as i32).of_size(w, h);
    draw_hollow_rect_mut(img, rect, Rgba(color));
    if w > 2 && h > 2 {
        let inner = Rect::at(left as i32 + 1, top as i32 + 1).of_size(w - 2, h - 2);
        draw_hollow_rect_mut(img, inner, Rgba(color));
    }
    draw_label_px(img, left, top.saturating_sub(16), label, color);
}

fn draw_banner(img: &mut RgbaImage, text: &str, color: [u8; 4]) {
    let w = img.width().min(420);
    let h = 28u32;
    for y in 8..8 + h {
        for x in 8..8 + w {
            if x < img.width() && y < img.height() {
                blend(img, x, y, color, 0.75);
            }
        }
    }
    draw_label_px(img, 16, 12, text, [255, 255, 255, 255]);
}

fn draw_label_at(img: &mut RgbaImage, map: &CoordMap, nx: f32, ny: f32, text: &str, color: [u8; 4]) {
    let (x, y) = map.xy(img, nx, ny);
    draw_label_px(img, x, y, text, color);
}

fn draw_label_px(img: &mut RgbaImage, x: u32, y: u32, text: &str, color: [u8; 4]) {
    let Some(bytes) = FONT.as_ref() else {
        return;
    };
    let Ok(font) = FontRef::try_from_slice(bytes) else {
        return;
    };
    let scale = PxScale::from(16.0);
    let clipped: String = text.chars().take(48).collect();
    let bg_w = (clipped.len() as u32 * 8 + 8).min(img.width().saturating_sub(x).max(1));
    let bg_h = 18u32;
    for yy in y..y.saturating_add(bg_h).min(img.height()) {
        for xx in x..x.saturating_add(bg_w).min(img.width()) {
            blend(img, xx, yy, [20, 16, 28, 255], 0.72);
        }
    }
    draw_text_mut(img, Rgba(color), x as i32 + 2, y as i32 + 1, scale, &font, &clipped);
}

fn stroke_polygon(img: &mut RgbaImage, map: &CoordMap, ring: &[[f32; 2]], color: [u8; 4]) {
    if ring.len() < 2 {
        return;
    }
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        let (x1, y1) = map.xy(img, a[0], a[1]);
        let (x2, y2) = map.xy(img, b[0], b[1]);
        draw_line_segment_mut(img, (x1 as f32, y1 as f32), (x2 as f32, y2 as f32), Rgba(color));
    }
}

fn fill_polygon(img: &mut RgbaImage, map: &CoordMap, ring: &[[f32; 2]], color: [u8; 4], alpha: f32) {
    if ring.len() < 3 {
        return;
    }
    let pts: Vec<(u32, u32)> = ring.iter().map(|p| map.xy(img, p[0], p[1])).collect();
    let min_x = pts.iter().map(|p| p.0).min().unwrap_or(0);
    let max_x = pts.iter().map(|p| p.0).max().unwrap_or(0);
    let min_y = pts.iter().map(|p| p.1).min().unwrap_or(0);
    let max_y = pts.iter().map(|p| p.1).max().unwrap_or(0);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if point_in_poly(x as f32, y as f32, &pts) {
                blend(img, x, y, color, alpha);
            }
        }
    }
}

fn point_in_poly(x: f32, y: f32, pts: &[(u32, u32)]) -> bool {
    let mut inside = false;
    let n = pts.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (pts[i].0 as f32, pts[i].1 as f32);
        let (xj, yj) = (pts[j].0 as f32, pts[j].1 as f32);
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + f32::EPSILON) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn blend(img: &mut RgbaImage, x: u32, y: u32, color: [u8; 4], a: f32) {
    if x >= img.width() || y >= img.height() {
        return;
    }
    let p = img.get_pixel(x, y).0;
    let a = a.clamp(0.0, 1.0);
    let ia = 1.0 - a;
    img.put_pixel(
        x,
        y,
        Rgba([
            (p[0] as f32 * ia + color[0] as f32 * a) as u8,
            (p[1] as f32 * ia + color[1] as f32 * a) as u8,
            (p[2] as f32 * ia + color[2] as f32 * a) as u8,
            255,
        ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::DetObject;

    fn map_xy(max_x: f32, max_y: f32, img_w: u32, img_h: u32, x: f32, y: f32) -> (u32, u32) {
        let img = RgbaImage::new(img_w, img_h);
        CoordMap::from_extent(max_x, max_y, img_w, img_h).xy(&img, x, y)
    }

    #[test]
    fn normalized_fractions_fill_the_image() {
        let (x, y) = map_xy(1.0, 1.0, 1536, 1536, 0.5, 1.0);
        assert_eq!((x, y), (768, 1535));
    }

    #[test]
    fn qwen_1000_canvas_scales_onto_the_display_image() {
        let (x, y) = map_xy(1000.0, 500.0, 1536, 1536, 1000.0, 500.0);
        assert_eq!((x, y), (1535, 768));
    }

    #[test]
    fn native_image_pixels_stay_one_to_one() {
        let (x, y) = map_xy(1530.0, 1530.0, 1536, 1536, 200.0, 400.0);
        assert_eq!((x, y), (200, 400));
    }

    #[test]
    fn percent_coords_scale_from_100() {
        let (x, y) = map_xy(100.0, 100.0, 200, 100, 50.0, 100.0);
        assert_eq!((x, y), (100, 99));
    }

    #[test]
    fn extent_uses_box_far_corner() {
        let result = TaskResult::Detection {
            objects: vec![DetObject {
                label: "guitar".into(),
                confidence: 0.9,
                bbox: BBox {
                    x: 142.0,
                    y: 516.0,
                    w: 858.0,
                    h: 484.0,
                },
            }],
        };
        let e = coord_extent(&result);
        assert!((e.max_x - 1000.0).abs() < f32::EPSILON);
        assert!((e.max_y - 1000.0).abs() < f32::EPSILON);
        let map = CoordMap::from_extent(e.max_x, e.max_y, 1536, 1536);
        assert!((map.sx - 1.536).abs() < 0.001);
        assert!((map.sy - 1.536).abs() < 0.001);
    }

    #[test]
    fn qwen_1000_on_portrait_does_not_use_native_pixels() {
        // Hat overflow (x+w = 1239) used to trip "native pixels of the 1440 side".
        let map = CoordMap::from_extent(1239.0, 980.0, 1080, 1440);
        assert!((map.sx - 1.08).abs() < 0.001);
        assert!((map.sy - 1.44).abs() < 0.001);
        let img = RgbaImage::new(1080, 1440);
        let (x, y) = map.xy(&img, 998.0, 980.0);
        assert_eq!((x, y), (1078, 1411));
    }
}
