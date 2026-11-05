//! Prompt templates, JSON contracts, and parsers for the five vision tasks.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Classification,
    #[default]
    ObjectDetection,
    InstanceSegmentation,
    KeypointDetection,
    Ocr,
}

impl TaskKind {
    pub fn from_index(i: i32) -> Self {
        match i {
            0 => Self::Classification,
            1 => Self::ObjectDetection,
            2 => Self::InstanceSegmentation,
            3 => Self::KeypointDetection,
            _ => Self::Ocr,
        }
    }

    pub fn to_index(self) -> i32 {
        match self {
            Self::Classification => 0,
            Self::ObjectDetection => 1,
            Self::InstanceSegmentation => 2,
            Self::KeypointDetection => 3,
            Self::Ocr => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Classification => "Classification",
            Self::ObjectDetection => "Object detection",
            Self::InstanceSegmentation => "Instance segmentation",
            Self::KeypointDetection => "Keypoint detection",
            Self::Ocr => "OCR",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetObject {
    pub label: String,
    #[serde(default)]
    pub confidence: f32,
    pub bbox: BBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub label: String,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub polygons: Vec<Vec<[f32; 2]>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keypoint {
    pub label: String,
    #[serde(default)]
    pub confidence: f32,
    pub point: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    #[serde(default)]
    pub confidence: f32,
    pub bbox: Option<BBox>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskResult {
    Classification {
        label: String,
        #[serde(default)]
        confidence: f32,
        #[serde(default)]
        reasoning: String,
    },
    Detection {
        objects: Vec<DetObject>,
    },
    Segmentation {
        instances: Vec<Instance>,
    },
    Keypoints {
        keypoints: Vec<Keypoint>,
        #[serde(default)]
        skeleton: Vec<[u32; 2]>,
    },
    Ocr {
        text: String,
        #[serde(default)]
        lines: Vec<OcrLine>,
    },
    Invalid {
        message: String,
    },
}

impl TaskResult {
    pub fn summary(&self) -> String {
        match self {
            Self::Classification { label, confidence, .. } => {
                format!("{label}  ({:.0}%)", confidence * 100.0)
            }
            Self::Detection { objects } => format!("{} objects detected", objects.len()),
            Self::Segmentation { instances } => format!("{} instances", instances.len()),
            Self::Keypoints { keypoints, .. } => format!("{} keypoints", keypoints.len()),
            Self::Ocr { text, .. } => {
                let preview: String = text.chars().take(80).collect();
                if text.chars().count() > 80 {
                    format!("{preview}…")
                } else if preview.is_empty() {
                    "No text detected".into()
                } else {
                    preview
                }
            }
            Self::Invalid { message } => message.clone(),
        }
    }

    pub fn legend(&self) -> Vec<(String, i32)> {
        let mut map = Vec::<(String, i32)>::new();
        let bump = |map: &mut Vec<(String, i32)>, label: &str| {
            if let Some((_, n)) = map.iter_mut().find(|(l, _)| l == label) {
                *n += 1;
            } else {
                map.push((label.to_string(), 1));
            }
        };
        match self {
            Self::Classification { label, .. } => bump(&mut map, label),
            Self::Detection { objects } => {
                for o in objects {
                    bump(&mut map, &o.label);
                }
            }
            Self::Segmentation { instances } => {
                for o in instances {
                    bump(&mut map, &o.label);
                }
            }
            Self::Keypoints { keypoints, .. } => {
                for o in keypoints {
                    bump(&mut map, &o.label);
                }
            }
            Self::Ocr { lines, .. } => {
                if !lines.is_empty() {
                    bump(&mut map, "text");
                    if let Some((_, n)) = map.last_mut() {
                        *n = lines.len() as i32;
                    }
                }
            }
            Self::Invalid { .. } => {}
        }
        map
    }
}

pub fn system_prompt(task: TaskKind) -> &'static str {
    match task {
        TaskKind::Classification => {
            "You are a vision classifier. Reply with a single JSON object only, no markdown. \
             Schema: {\"label\": string, \"confidence\": number 0-1, \"reasoning\": string}. \
             Pick the single most likely class."
        }
        TaskKind::ObjectDetection => {
            "You are an object detector. Reply with a single JSON object only, no markdown. \
             Schema: {\"objects\": [{\"label\": string, \"confidence\": number 0-1, \
             \"bbox\": {\"x\": number, \"y\": number, \"w\": number, \"h\": number}}]}. \
             Bounding boxes are normalized 0..1 with origin at the top-left of the image. \
             x,y is the top-left of the box. w,h are width and height. Detect every distinct object."
        }
        TaskKind::InstanceSegmentation => {
            "You are an instance segmentation model. Reply with a single JSON object only, no markdown. \
             Schema: {\"instances\": [{\"label\": string, \"confidence\": number 0-1, \
             \"polygons\": [[[x,y], ...]]}]}. Coordinates are normalized 0..1, origin top-left. \
             Each instance has one or more polygons."
        }
        TaskKind::KeypointDetection => {
            "You are a keypoint detector. Reply with a single JSON object only, no markdown. \
             Schema: {\"keypoints\": [{\"label\": string, \"confidence\": number 0-1, \
             \"point\": {\"x\": number, \"y\": number}}], \"skeleton\": [[i,j], ...]}. \
             Coordinates are normalized 0..1, origin top-left. skeleton indexes into keypoints."
        }
        TaskKind::Ocr => {
            "You are an OCR engine. Reply with a single JSON object only, no markdown. \
             Schema: {\"text\": string, \"lines\": [{\"text\": string, \"confidence\": number 0-1, \
             \"bbox\": {\"x\": number, \"y\": number, \"w\": number, \"h\": number}}]}. \
             Bounding boxes are normalized 0..1, origin top-left. text is the full transcript."
        }
    }
}

pub fn user_prompt(task: TaskKind, focus: &str) -> String {
    let base = match task {
        TaskKind::Classification => "Classify the main subject of this image. JSON only.",
        TaskKind::ObjectDetection => "Detect every object in this image. JSON only.",
        TaskKind::InstanceSegmentation => "Segment every instance in this image. JSON only.",
        TaskKind::KeypointDetection => "Detect keypoints in this image. JSON only.",
        TaskKind::Ocr => "Read all text in this image. JSON only.",
    };
    let focus = focus.trim();
    if focus.is_empty() {
        return base.to_string();
    }
    let restrict = match task {
        TaskKind::Classification => {
            "Only choose from the classes described below. If none fit, pick the closest and lower the confidence."
        }
        TaskKind::ObjectDetection | TaskKind::InstanceSegmentation => {
            "Only include the objects described below. If none are present, return an empty list."
        }
        TaskKind::KeypointDetection => {
            "Only include keypoints described below. If none are present, return an empty list."
        }
        TaskKind::Ocr => {
            "Only read text that matches the request below. If none, return empty text and lines."
        }
    };
    format!("{base} {restrict}\nUser request: {focus}")
}

pub fn extract_json(s: &str) -> &str {
    let t = s.trim();
    let t = t
        .strip_prefix("```json")
        .or_else(|| t.strip_prefix("```JSON"))
        .or_else(|| t.strip_prefix("```"))
        .unwrap_or(t);
    let t = t.strip_suffix("```").unwrap_or(t).trim();
    match (t.find('{'), t.rfind('}')) {
        (Some(a), Some(b)) if b > a => &t[a..=b],
        _ => t,
    }
}

fn parse_json_value(raw: &str) -> Result<(Value, bool)> {
    let snippet = extract_json(raw);
    if let Ok(v) = serde_json::from_str(snippet) {
        return Ok((v, false));
    }
    if let Ok(v) = serde_json::from_str(raw) {
        return Ok((v, false));
    }
    if let Some(v) = close_truncated_json(raw).or_else(|| close_truncated_json(snippet)) {
        return Ok((v, true));
    }
    serde_json::from_str::<Value>(snippet)
        .context("parse JSON")
        .map(|v| (v, false))
}

/// Models often hit max tokens mid-object. Keep every complete value, then close
/// the open `[` / `{` so the detections we already have still parse.
fn close_truncated_json(s: &str) -> Option<Value> {
    let start = s.find('{').or_else(|| s.find('['))?;
    let body = &s[start..];
    let cut = last_complete_prefix(body)?;
    let mut out = body[..cut].trim_end().to_string();
    while out.ends_with(',') {
        out.pop();
        let trimmed = out.trim_end();
        out.truncate(trimmed.len());
    }
    let closers = unclosed_brackets(&out)?;
    out.extend(closers.chars());
    serde_json::from_str(&out).ok()
}

fn last_complete_prefix(s: &str) -> Option<usize> {
    let mut in_string = false;
    let mut escape = false;
    let mut stack: Vec<char> = Vec::new();
    let mut last = 0usize;
    for (i, c) in s.char_indices() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
                escape = true;
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                let want = if c == '}' { '}' } else { ']' };
                if stack.last().copied() != Some(want) {
                    break;
                }
                stack.pop();
                let end = i + c.len_utf8();
                if stack.is_empty() {
                    return Some(end);
                }
                last = end;
            }
            _ => {}
        }
    }
    (last > 0).then_some(last)
}

fn unclosed_brackets(s: &str) -> Option<String> {
    let mut in_string = false;
    let mut escape = false;
    let mut stack = Vec::new();
    for c in s.chars() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
                escape = true;
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                let want = if c == '}' { '}' } else { ']' };
                if stack.pop() != Some(want) {
                    return None;
                }
            }
            _ => {}
        }
    }
    if in_string {
        return None;
    }
    Some(stack.into_iter().rev().collect())
}

fn f32_field(v: &Value, keys: &[&str]) -> f32 {
    for k in keys {
        if let Some(x) = v.get(*k).and_then(|n| n.as_f64()) {
            return x as f32;
        }
    }
    0.0
}

fn bbox_from(v: &Value) -> Option<BBox> {
    if let Some(arr) = v.as_array() {
        if arr.len() >= 4 {
            let a = arr[0].as_f64()? as f32;
            let b = arr[1].as_f64()? as f32;
            let c = arr[2].as_f64()? as f32;
            let d = arr[3].as_f64()? as f32;
            // Pixel xyxy: far corner is > 1.5 and greater than origin.
            if c > 1.5 && d > 1.5 && c > a && d > b {
                return Some(BBox {
                    x: a,
                    y: b,
                    w: c - a,
                    h: d - b,
                });
            }
            return Some(BBox {
                x: a,
                y: b,
                w: c,
                h: d,
            });
        }
    }
    let obj = v.as_object()?;
    if let (Some(x), Some(y), Some(w), Some(h)) = (
        obj.get("x")
            .or_else(|| obj.get("left"))
            .and_then(Value::as_f64),
        obj.get("y")
            .or_else(|| obj.get("top"))
            .and_then(Value::as_f64),
        obj.get("w")
            .or_else(|| obj.get("width"))
            .and_then(Value::as_f64),
        obj.get("h")
            .or_else(|| obj.get("height"))
            .and_then(Value::as_f64),
    ) {
        return Some(BBox {
            x: x as f32,
            y: y as f32,
            w: w as f32,
            h: h as f32,
        });
    }
    if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (
        obj.get("x1")
            .or_else(|| obj.get("x_min"))
            .or_else(|| obj.get("xmin"))
            .and_then(Value::as_f64),
        obj.get("y1")
            .or_else(|| obj.get("y_min"))
            .or_else(|| obj.get("ymin"))
            .and_then(Value::as_f64),
        obj.get("x2")
            .or_else(|| obj.get("x_max"))
            .or_else(|| obj.get("xmax"))
            .and_then(Value::as_f64),
        obj.get("y2")
            .or_else(|| obj.get("y_max"))
            .or_else(|| obj.get("ymax"))
            .and_then(Value::as_f64),
    ) {
        return Some(BBox {
            x: x1 as f32,
            y: y1 as f32,
            w: (x2 - x1) as f32,
            h: (y2 - y1) as f32,
        });
    }
    None
}

fn point_from(v: &Value) -> Option<[f32; 2]> {
    if let Some(arr) = v.as_array() {
        if arr.len() >= 2 {
            return Some([arr[0].as_f64().unwrap_or(0.0) as f32, arr[1].as_f64().unwrap_or(0.0) as f32]);
        }
    }
    if let Some(obj) = v.as_object() {
        let x = obj.get("x").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let y = obj.get("y").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        return Some([x, y]);
    }
    None
}

pub struct ParsedTask {
    pub result: TaskResult,
    pub value: Value,
    pub truncated: bool,
}

pub fn parse_task(task: TaskKind, raw: &str) -> Result<ParsedTask> {
    let (value, truncated) = parse_json_value(raw)?;
    let parsed = match task {
        TaskKind::Classification => {
            let label = value
                .get("label")
                .or_else(|| value.get("class"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            TaskResult::Classification {
                confidence: f32_field(&value, &["confidence", "score"]),
                reasoning: value
                    .get("reasoning")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                label,
            }
        }
        TaskKind::ObjectDetection => {
            let objects = value
                .get("objects")
                .or_else(|| value.get("detections"))
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|o| {
                            Some(DetObject {
                                label: o
                                    .get("label")
                                    .or_else(|| o.get("class"))
                                    .and_then(Value::as_str)?
                                    .to_string(),
                                confidence: f32_field(o, &["confidence", "score"]),
                                bbox: bbox_from(o.get("bbox").unwrap_or(o))?,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            TaskResult::Detection { objects }
        }
        TaskKind::InstanceSegmentation => {
            let instances = value
                .get("instances")
                .or_else(|| value.get("objects"))
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|o| {
                            let polygons = o
                                .get("polygons")
                                .or_else(|| o.get("polygon").map(|p| {
                                    // wrap a single polygon as a 1-element array via leaked Value; handled below
                                    p
                                }))
                                .cloned()
                                .unwrap_or(Value::Array(vec![]));
                            let polys = parse_polygons(&polygons);
                            Some(Instance {
                                label: o
                                    .get("label")
                                    .or_else(|| o.get("class"))
                                    .and_then(Value::as_str)?
                                    .to_string(),
                                confidence: f32_field(o, &["confidence", "score"]),
                                polygons: polys,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            TaskResult::Segmentation { instances }
        }
        TaskKind::KeypointDetection => {
            let keypoints = value
                .get("keypoints")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|o| {
                            let pt = o.get("point").and_then(point_from).or_else(|| point_from(o))?;
                            Some(Keypoint {
                                label: o
                                    .get("label")
                                    .or_else(|| o.get("name"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("point")
                                    .to_string(),
                                confidence: f32_field(o, &["confidence", "score"]),
                                point: pt,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let skeleton = value
                .get("skeleton")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| {
                            let a = e.as_array()?;
                            Some([a.first()?.as_u64()? as u32, a.get(1)?.as_u64()? as u32])
                        })
                        .collect()
                })
                .unwrap_or_default();
            TaskResult::Keypoints { keypoints, skeleton }
        }
        TaskKind::Ocr => {
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let lines = value
                .get("lines")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|o| {
                            Some(OcrLine {
                                text: o.get("text").and_then(Value::as_str)?.to_string(),
                                confidence: f32_field(o, &["confidence", "score"]),
                                bbox: o.get("bbox").and_then(bbox_from),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            TaskResult::Ocr { text, lines }
        }
    };
    Ok(ParsedTask {
        result: parsed,
        value,
        truncated,
    })
}

fn parse_polygons(v: &Value) -> Vec<Vec<[f32; 2]>> {
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    // Either [[[x,y],...]] or [[x,y],...]
    if arr.first().and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_array).is_some()
        || arr
            .first()
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_object)
            .is_some()
    {
        return arr.iter().filter_map(parse_ring).collect();
    }
    parse_ring(v).into_iter().collect()
}

fn parse_ring(v: &Value) -> Option<Vec<[f32; 2]>> {
    let arr = v.as_array()?;
    let ring: Vec<[f32; 2]> = arr.iter().filter_map(point_from).collect();
    if ring.len() >= 3 { Some(ring) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_fenced_json() {
        let raw = "```json\n{\"label\":\"cat\",\"confidence\":0.9}\n```";
        let parsed = parse_task(TaskKind::Classification, raw).unwrap();
        match parsed.result {
            TaskResult::Classification { label, .. } => assert_eq!(label, "cat"),
            _ => panic!("wrong variant"),
        }
        assert!(!parsed.truncated);
    }

    #[test]
    fn parses_detection() {
        let raw = r#"{"objects":[{"label":"car","confidence":0.8,"bbox":{"x":0.1,"y":0.2,"w":0.3,"h":0.4}}]}"#;
        let parsed = parse_task(TaskKind::ObjectDetection, raw).unwrap();
        match parsed.result {
            TaskResult::Detection { objects } => assert_eq!(objects[0].label, "car"),
            _ => panic!("wrong variant"),
        }
        assert!(!parsed.truncated);
    }

    #[test]
    fn salvages_truncated_detection_array() {
        let raw = r#"{"objects": [
            {"label": "car", "confidence": 0.85, "bbox": {"x": 0.0, "y": 0.12, "w": 0.05, "h": 0.08}},
            {"label": "car", "confidence": 0.9, "bbox": {"x": 0.04, "y": 0.22, "w": 0.075, "h": 0.067}},
            {"label": "car", "confidence": 0.9, "bbox": {"x": 0.375, "y": 0.31, "w": 0.075, ""#;
        let parsed = parse_task(TaskKind::ObjectDetection, raw).unwrap();
        match parsed.result {
            TaskResult::Detection { objects } => {
                assert_eq!(objects.len(), 2);
                assert!((objects[0].bbox.x - 0.0).abs() < f32::EPSILON);
                assert!((objects[1].bbox.y - 0.22).abs() < f32::EPSILON);
            }
            other => panic!("expected detections, got {other:?}"),
        }
        assert!(parsed.truncated);
        assert_eq!(parsed.value["objects"].as_array().map(|a| a.len()), Some(2));
    }
}
