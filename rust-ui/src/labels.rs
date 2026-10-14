//! Event labels for one video, candidate events to review, and timeline geometry. Pure logic, no UI.
//!
//! The labels file is `.jsonl` with `{"video","kind","frame","time_s","source"}` rows, read by
//! `scripts/eval_events.py`. Frames are one-based and `time_s = frame / fps`, like the detection pipeline.
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const FAILURE: &str = "suspected_failure_to_yield";
pub const YIELDED: &str = "yielded";
/// What the image rule calls its candidates; they are reviewed as possible failures to yield.
const CANDIDATE_ALIAS: &str = "vehicle_on_crosswalk_with_pedestrian";

pub fn time_of(frame: u64, fps: f64) -> f64 {
    (frame as f64 / fps * 1000.0).round() / 1000.0
}

pub fn clock(seconds: f64) -> String {
    let total = (seconds * 1000.0).round().max(0.0) as u64;
    format!("{:02}:{:02}.{:03}", total / 60_000, total / 1000 % 60, total % 1000)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub kind: String,
    pub frame: u64,
    pub time_s: f64,
    pub source: String,
}

/// Labels for ONE video, saved after every change; resumes from an existing file.
pub struct Store {
    pub path: PathBuf,
    video: String,
    fps: f64,
    /// Insertion order, so undo removes the most recent mark rather than the latest in time.
    pub labels: Vec<Label>,
    pub declared_none: bool,
}

impl Store {
    pub fn open(path: &Path, video: &str, fps: f64) -> Result<Self, String> {
        let mut store = Self {
            path: path.to_path_buf(),
            video: video.to_string(),
            fps,
            labels: Vec::new(),
            declared_none: false,
        };
        if path.is_file() {
            let text = fs::read_to_string(path).map_err(|e| format!("無法讀取 {}：{e}", path.display()))?;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let row: Value = serde_json::from_str(line).map_err(|e| format!("{} 格式錯誤：{e}", path.display()))?;
                if row["video"].as_str() != Some(video) {
                    return Err(format!("{} 已經存放另一支影片的標註，請每支影片使用不同的標註檔。", path.display()));
                }
                store.labels.push(Label {
                    kind: row["kind"].as_str().unwrap_or(FAILURE).to_string(),
                    frame: row["frame"].as_u64().unwrap_or(0),
                    time_s: row["time_s"].as_f64().unwrap_or(0.0),
                    source: row["source"].as_str().unwrap_or("manual").to_string(),
                });
            }
        }
        Ok(store)
    }

    pub fn add(&mut self, kind: &str, frame: u64, source: &str) -> Result<(), String> {
        if self.labels.iter().any(|l| l.kind == kind && l.frame == frame) {
            return Ok(()); // the same mark twice on one frame is a slip, not a second event
        }
        self.declared_none = false;
        self.labels.push(Label { kind: kind.to_string(), frame, time_s: time_of(frame, self.fps), source: source.to_string() });
        self.save()
    }

    pub fn undo(&mut self) -> Result<Option<Label>, String> {
        let removed = self.labels.pop();
        self.save()?;
        Ok(removed)
    }

    /// `index` refers to `sorted()`.
    pub fn remove_sorted(&mut self, index: usize) -> Result<Option<Label>, String> {
        let target = self.sorted().get(index).cloned();
        if let Some(target) = &target {
            if let Some(position) = self.labels.iter().position(|l| l == target) {
                self.labels.remove(position);
            }
        }
        self.save()?;
        Ok(target)
    }

    /// The human watched the whole video and saw no event. Refused while marks exist.
    pub fn declare_none(&mut self) -> Result<bool, String> {
        if !self.labels.is_empty() {
            return Ok(false);
        }
        self.declared_none = true;
        self.save()?;
        Ok(true)
    }

    pub fn sorted(&self) -> Vec<Label> {
        let mut rows = self.labels.clone();
        rows.sort_by_key(|l| l.frame);
        rows
    }

    /// Atomic write. Nothing is written when there are no marks, none was declared, and no file exists yet.
    pub fn save(&self) -> Result<(), String> {
        if self.labels.is_empty() && !self.declared_none && !self.path.is_file() {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("無法建立資料夾 {}：{e}", parent.display()))?;
        }
        let mut text = String::new();
        for l in self.sorted() {
            text.push_str(&json!({"video": self.video, "kind": l.kind, "frame": l.frame, "time_s": l.time_s, "source": l.source}).to_string());
            text.push('\n');
        }
        let temporary = self.path.with_extension("jsonl.tmp");
        fs::write(&temporary, text).map_err(|e| format!("無法寫入 {}：{e}", temporary.display()))?;
        fs::rename(&temporary, &self.path).map_err(|e| format!("無法儲存 {}：{e}", self.path.display()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    Pending,
    Confirmed,
    Rejected,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub time_s: f64,
    #[allow(dead_code)] // kept for the alias test and future per-kind review
    pub kind: String,
    pub decision: Decision,
}

/// Detected events (`events.jsonl` from the pipeline) to confirm or reject.
pub fn load_candidates(path: &Path) -> Result<Vec<Candidate>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("無法讀取 {}：{e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: Value = serde_json::from_str(line).map_err(|e| format!("{} 格式錯誤：{e}", path.display()))?;
        let time_s = row["time_s"].as_f64().ok_or_else(|| format!("{} 的事件缺少 time_s", path.display()))?;
        let kind = match row["kind"].as_str().unwrap_or(FAILURE) {
            CANDIDATE_ALIAS => FAILURE,
            other => other,
        };
        rows.push(Candidate { time_s, kind: kind.to_string(), decision: Decision::Pending });
    }
    rows.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
    Ok(rows)
}

/// Decisions on candidates live beside the labels (`<labels>.review.json`); rejected ones are NOT labels.
pub fn review_path(labels: &Path) -> PathBuf {
    labels.with_extension("review.json")
}

pub fn save_review(path: &Path, candidates: &[Candidate]) -> Result<(), String> {
    let rows: Vec<Value> = candidates
        .iter()
        .filter(|c| c.decision != Decision::Pending)
        .map(|c| json!({"time_s": c.time_s, "decision": if c.decision == Decision::Confirmed { "confirmed" } else { "rejected" }}))
        .collect();
    if rows.is_empty() && !path.is_file() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(path, serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?)
        .map_err(|e| format!("無法儲存 {}：{e}", path.display()))
}

pub fn apply_review(path: &Path, candidates: &mut [Candidate]) {
    let Ok(text) = fs::read_to_string(path) else { return };
    let Ok(Value::Array(rows)) = serde_json::from_str::<Value>(&text) else { return };
    for row in rows {
        let (Some(time), Some(decision)) = (row["time_s"].as_f64(), row["decision"].as_str()) else { continue };
        if let Some(c) = candidates.iter_mut().find(|c| (c.time_s - time).abs() < 1e-6) {
            c.decision = if decision == "confirmed" { Decision::Confirmed } else { Decision::Rejected };
        }
    }
}

/// Timeline geometry: x is a pixel position, `left`/`width` describe the bar, frames are zero-based indices.
pub fn x_to_frame(x: f32, left: f32, width: f32, count: u64) -> u64 {
    if count == 0 || width <= 0.0 {
        return 0;
    }
    let fraction = ((x - left) / width).clamp(0.0, 1.0) as f64;
    ((fraction * (count - 1) as f64).round() as u64).min(count - 1)
}

pub fn frame_to_x(frame: u64, left: f32, width: f32, count: u64) -> f32 {
    if count <= 1 {
        return left;
    }
    left + width * (frame.min(count - 1) as f64 / (count - 1) as f64) as f32
}

/// The frame window shown by the zoomed bar: `seconds` on each side of the playhead, kept inside the video.
pub fn zoom_window(current: u64, count: u64, fps: f64, seconds: f64) -> (u64, u64) {
    let half = (seconds * fps).round().max(1.0) as u64;
    let span = (2 * half).min(count.saturating_sub(1));
    let start = current.saturating_sub(half).min(count.saturating_sub(1).saturating_sub(span));
    (start, start + span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp_dir() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!("labels-test-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn rows(path: &Path) -> Vec<Value> {
        fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    #[test]
    fn time_uses_one_based_frame_over_fps_like_the_pipeline() {
        assert_eq!(time_of(660, 60.0), 11.0);
        assert_eq!(time_of(1, 30.0), 0.033);
        assert_eq!(clock(11.0), "00:11.000");
        assert_eq!(clock(125.5), "02:05.500");
    }

    #[test]
    fn marks_are_saved_immediately_sorted_and_undo_removes_the_latest_made() {
        let dir = temp_dir();
        let path = dir.join("a.jsonl");
        let mut store = Store::open(&path, "a.mp4", 60.0).unwrap();
        store.add(FAILURE, 660, "manual").unwrap();
        store.add(YIELDED, 300, "manual").unwrap();
        assert_eq!(rows(&path).iter().map(|r| r["frame"].as_u64().unwrap()).collect::<Vec<_>>(), vec![300, 660]);
        assert_eq!(store.undo().unwrap().unwrap().frame, 300);
        let remaining = rows(&path);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0]["time_s"].as_f64().unwrap(), 11.0);
        assert_eq!(remaining[0]["video"], "a.mp4");
    }

    #[test]
    fn the_same_mark_twice_on_one_frame_is_ignored() {
        let dir = temp_dir();
        let mut store = Store::open(&dir.join("a.jsonl"), "a.mp4", 60.0).unwrap();
        store.add(FAILURE, 10, "manual").unwrap();
        store.add(FAILURE, 10, "manual").unwrap();
        assert_eq!(store.labels.len(), 1);
    }

    #[test]
    fn resumes_and_keeps_one_file_per_video() {
        let dir = temp_dir();
        let path = dir.join("a.jsonl");
        Store::open(&path, "a.mp4", 60.0).unwrap().add(FAILURE, 660, "manual").unwrap();
        let mut again = Store::open(&path, "a.mp4", 60.0).unwrap();
        assert_eq!(again.labels.len(), 1);
        again.add(FAILURE, 900, "manual").unwrap();
        assert_eq!(rows(&path).len(), 2);
        assert!(Store::open(&path, "b.mp4", 30.0).is_err());
    }

    #[test]
    fn nothing_is_written_until_something_is_decided_and_none_cannot_wipe_marks() {
        let dir = temp_dir();
        let path = dir.join("a.jsonl");
        let mut store = Store::open(&path, "a.mp4", 60.0).unwrap();
        store.save().unwrap();
        assert!(!path.exists());
        assert!(store.declare_none().unwrap());
        assert_eq!(fs::read_to_string(&path).unwrap(), "");
        store.add(FAILURE, 5, "manual").unwrap();
        assert!(!store.declare_none().unwrap());
        assert_eq!(rows(&path).len(), 1);
    }

    #[test]
    fn remove_by_sorted_position_removes_that_label() {
        let dir = temp_dir();
        let mut store = Store::open(&dir.join("a.jsonl"), "a.mp4", 60.0).unwrap();
        store.add(FAILURE, 900, "manual").unwrap();
        store.add(FAILURE, 100, "manual").unwrap();
        assert_eq!(store.remove_sorted(0).unwrap().unwrap().frame, 100);
        assert_eq!(store.sorted().iter().map(|l| l.frame).collect::<Vec<_>>(), vec![900]);
    }

    #[test]
    fn candidates_load_with_alias_and_review_round_trips_without_becoming_labels() {
        let dir = temp_dir();
        let events = dir.join("events.jsonl");
        fs::write(&events, "{\"kind\":\"vehicle_on_crosswalk_with_pedestrian\",\"time_s\":11.0}\n{\"kind\":\"yielded\",\"time_s\":5.5}\n").unwrap();
        let mut candidates = load_candidates(&events).unwrap();
        assert_eq!(candidates[0].time_s, 5.5);
        assert_eq!(candidates[1].kind, FAILURE);
        candidates[1].decision = Decision::Confirmed;
        candidates[0].decision = Decision::Rejected;
        let review = review_path(&dir.join("a.jsonl"));
        save_review(&review, &candidates).unwrap();
        let mut fresh = load_candidates(&events).unwrap();
        apply_review(&review, &mut fresh);
        assert_eq!(fresh[0].decision, Decision::Rejected);
        assert_eq!(fresh[1].decision, Decision::Confirmed);
        assert!(!dir.join("a.jsonl").exists(), "review decisions must not create a labels file");
    }

    #[test]
    fn timeline_maps_pixels_to_frames_and_back() {
        assert_eq!(x_to_frame(100.0, 100.0, 800.0, 900), 0);
        assert_eq!(x_to_frame(900.0, 100.0, 800.0, 900), 899);
        assert_eq!(x_to_frame(500.0, 100.0, 800.0, 901), 450);
        assert_eq!(x_to_frame(-50.0, 100.0, 800.0, 900), 0);
        assert_eq!(x_to_frame(5000.0, 100.0, 800.0, 900), 899);
        assert!((frame_to_x(450, 100.0, 800.0, 901) - 500.0).abs() < 0.01);
        assert_eq!(frame_to_x(5, 100.0, 800.0, 1), 100.0);
        assert_eq!(x_to_frame(10.0, 0.0, 100.0, 0), 0);
    }

    #[test]
    fn zoom_window_follows_the_playhead_and_stays_inside_the_video() {
        assert_eq!(zoom_window(450, 900, 60.0, 2.0), (330, 570));
        assert_eq!(zoom_window(10, 900, 60.0, 2.0), (0, 240));
        assert_eq!(zoom_window(895, 900, 60.0, 2.0), (659, 899));
        assert_eq!(zoom_window(5, 100, 60.0, 5.0), (0, 99));
    }
}
