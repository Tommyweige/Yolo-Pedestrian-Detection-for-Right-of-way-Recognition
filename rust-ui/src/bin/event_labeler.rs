//! 事件標註工具: mark yield-to-pedestrian events on a video, or confirm/reject detected candidates.
//!
//! Writes `labels\<video>.jsonl` (read by scripts/eval_events.py). Reuses the desktop app's frame-accurate
//! Media Foundation decoder. Run: `event_labeler.exe [--video V] [--labels L.jsonl] [--events E.jsonl]`.
#[path = "../labels.rs"]
mod labels;
#[path = "../video.rs"]
mod video;

use eframe::egui;
use labels::{Candidate, Decision, FAILURE, Label, Store, YIELDED, clock, time_of};
use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    time::{Duration, Instant},
};

const RED: egui::Color32 = egui::Color32::from_rgb(214, 69, 65);
const GREEN: egui::Color32 = egui::Color32::from_rgb(52, 168, 83);
const AMBER: egui::Color32 = egui::Color32::from_rgb(240, 173, 40);
const GREY: egui::Color32 = egui::Color32::from_rgb(130, 130, 130);

#[derive(Clone, Copy, PartialEq)]
enum PickKind {
    Video,
    Events,
}

enum Action {
    Jump(u64),
    DeleteLabel(usize),
    Confirm(usize),
    Reject(usize),
    ResetCandidate(usize),
}

struct App {
    video_path: Option<PathBuf>,
    video_name: String,
    preview: Option<video::Preview>,
    texture: Option<egui::TextureHandle>,
    frame: u64,
    count: u64,
    fps: f64,
    playing: bool,
    play_start: u64,
    last_tick: Instant,
    speed: f32,
    requested: Option<u64>,
    displayed: Option<u64>,
    dirty: bool,
    scrubbing: bool,
    last_seek: Instant,
    store: Option<Store>,
    labels_path: Option<PathBuf>,
    candidates: Vec<Candidate>,
    status: String,
    error: Option<String>,
    picker: Option<(PickKind, Receiver<Option<PathBuf>>)>,
    started: Instant,
    screenshot: Option<PathBuf>,
    screenshot_requested: bool,
    start_frame: Option<u64>,
    pending_events: Option<PathBuf>,
}

fn default_labels_path(video: &Path) -> PathBuf {
    let stem = video.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "video".into());
    PathBuf::from("labels").join(format!("{stem}.jsonl"))
}

impl App {
    fn new() -> Self {
        Self {
            video_path: None,
            video_name: String::new(),
            preview: None,
            texture: None,
            frame: 0,
            count: 0,
            fps: 30.0,
            playing: false,
            play_start: 0,
            last_tick: Instant::now(),
            speed: 1.0,
            requested: None,
            displayed: None,
            dirty: false,
            scrubbing: false,
            last_seek: Instant::now(),
            store: None,
            labels_path: None,
            candidates: Vec::new(),
            status: "先開啟一支影片".into(),
            error: None,
            picker: None,
            started: Instant::now(),
            screenshot: None,
            screenshot_requested: false,
            start_frame: None,
            pending_events: None,
        }
    }

    fn open_video(&mut self, path: PathBuf, labels: Option<PathBuf>) {
        self.error = None;
        self.texture = None;
        self.store = None;
        self.candidates.clear();
        self.frame = 0;
        self.count = 0;
        self.playing = false;
        self.requested = None;
        self.displayed = None;
        self.video_name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.labels_path = Some(labels.unwrap_or_else(|| default_labels_path(&path)));
        match video::Preview::open(path.clone()) {
            Ok(preview) => {
                self.preview = Some(preview);
                self.dirty = true;
                self.status = "正在載入影片…".into();
            }
            Err(error) => {
                self.preview = None;
                self.error = Some(error);
            }
        }
        self.video_path = Some(path);
    }

    fn request_picker(&mut self, kind: PickKind, ctx: &egui::Context) {
        if self.picker.is_some() {
            return;
        }
        let (output, result) = mpsc::sync_channel(1);
        let ctx = ctx.clone();
        let spawned = std::thread::Builder::new().name("file-picker".into()).spawn(move || {
            let dialog = rfd::FileDialog::new();
            let chosen = match kind {
                PickKind::Video => dialog.set_title("開啟影片").add_filter("影片", &["mp4", "avi", "mkv"]).pick_file(),
                PickKind::Events => dialog.set_title("載入偵測結果 (events.jsonl)").add_filter("事件檔", &["jsonl"]).pick_file(),
            };
            let _ = output.send(chosen);
            ctx.request_repaint();
        });
        match spawned {
            Ok(_) => self.picker = Some((kind, result)),
            Err(e) => self.error = Some(format!("無法開啟檔案視窗：{e}")),
        }
    }

    fn load_events(&mut self, path: &Path) {
        match labels::load_candidates(path) {
            Ok(mut rows) => {
                if let Some(labels_path) = &self.labels_path {
                    labels::apply_review(&labels::review_path(labels_path), &mut rows);
                }
                self.status = format!("已載入 {} 個偵測事件,逐一確認或否決", rows.len());
                self.candidates = rows;
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn candidate_frame(&self, time_s: f64) -> u64 {
        ((time_s * self.fps).round() as i64 - 1).clamp(0, self.count.saturating_sub(1) as i64) as u64
    }

    fn seek(&mut self, frame: u64) {
        if self.count == 0 {
            return;
        }
        self.frame = frame.min(self.count - 1);
        self.playing = false;
        self.dirty = true;
    }

    fn step(&mut self, delta: i64) {
        let target = (self.frame as i64 + delta).clamp(0, self.count.saturating_sub(1) as i64) as u64;
        self.seek(target);
    }

    fn toggle_play(&mut self) {
        if self.count == 0 {
            return;
        }
        if self.playing {
            self.playing = false;
        } else {
            if self.frame + 1 >= self.count {
                self.frame = 0;
            }
            self.start_play();
        }
    }

    fn start_play(&mut self) {
        self.playing = true;
        self.play_start = self.frame;
        self.last_tick = Instant::now();
    }

    fn save_review(&mut self) {
        if let Some(path) = &self.labels_path {
            if let Err(e) = labels::save_review(&labels::review_path(path), &self.candidates) {
                self.error = Some(e);
            }
        }
    }

    fn mark(&mut self, kind: &str, source: &str, frame: u64) {
        let Some(store) = &mut self.store else {
            self.error = Some("影片還沒載入完成,無法標記".into());
            return;
        };
        match store.add(kind, frame + 1, source) {
            Ok(()) => {
                self.status = format!("已標記 {} @ {}", if kind == FAILURE { "沒讓行人" } else { "有讓行人" }, clock(time_of(frame + 1, self.fps)));
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn reset_candidate_for(&mut self, label: &Label) {
        if label.source == "confirmed_candidate" {
            if let Some(c) = self.candidates.iter_mut().find(|c| (c.time_s - label.time_s).abs() < 0.0015) {
                c.decision = Decision::Pending;
            }
            self.save_review();
        }
    }

    fn undo(&mut self) {
        let Some(store) = &mut self.store else { return };
        match store.undo() {
            Ok(Some(label)) => {
                self.status = format!("已復原 {}", clock(label.time_s));
                self.reset_candidate_for(&label);
            }
            Ok(None) => self.status = "沒有可以復原的標記".into(),
            Err(e) => self.error = Some(e),
        }
    }

    fn declare_none(&mut self) {
        let Some(store) = &mut self.store else { return };
        match store.declare_none() {
            Ok(true) => self.status = "已宣告:這支影片沒有任何事件".into(),
            Ok(false) => self.error = Some("已經有標記,請先逐一刪除或復原,才能宣告沒有事件".into()),
            Err(e) => self.error = Some(e),
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let pressed = |key: egui::Key| ctx.input(|i| i.key_pressed(key));
        let shift = ctx.input(|i| i.modifiers.shift);
        if pressed(egui::Key::Space) {
            self.toggle_play();
        }
        if pressed(egui::Key::A) || pressed(egui::Key::ArrowLeft) {
            self.step(if shift { -10 } else { -1 });
        }
        if pressed(egui::Key::D) || pressed(egui::Key::ArrowRight) {
            self.step(if shift { 10 } else { 1 });
        }
        if pressed(egui::Key::Z) {
            self.step(-10);
        }
        if pressed(egui::Key::C) {
            self.step(10);
        }
        if pressed(egui::Key::J) {
            self.step(-(self.fps.round() as i64));
        }
        if pressed(egui::Key::L) {
            self.step(self.fps.round() as i64);
        }
        if pressed(egui::Key::Home) {
            self.seek(0);
        }
        if pressed(egui::Key::End) {
            self.seek(self.count.saturating_sub(1));
        }
        if pressed(egui::Key::F) {
            self.mark(FAILURE, "manual", self.frame);
        }
        if pressed(egui::Key::Y) {
            self.mark(YIELDED, "manual", self.frame);
        }
        if pressed(egui::Key::U) {
            self.undo();
        }
        if pressed(egui::Key::N) {
            self.declare_none();
        }
        for (key, value) in [(egui::Key::Num1, 0.25), (egui::Key::Num2, 0.5), (egui::Key::Num3, 1.0), (egui::Key::Num4, 2.0)] {
            if pressed(key) {
                self.speed = value;
                if self.playing {
                    self.start_play();
                }
            }
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if let Some((kind, result)) = &self.picker {
            match result.try_recv() {
                Ok(chosen) => {
                    let kind = *kind;
                    self.picker = None;
                    if let Some(path) = chosen {
                        match kind {
                            PickKind::Video => self.open_video(path, None),
                            PickKind::Events => {
                                if self.store.is_some() {
                                    self.load_events(&path);
                                } else {
                                    self.pending_events = Some(path);
                                }
                            }
                        }
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => self.picker = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let frames: Vec<_> = self.preview.as_ref().map(|p| p.frames.try_iter().collect()).unwrap_or_default();
        for frame in frames {
            match frame {
                Ok(Some(frame)) => {
                    if !self.playing && frame.requested != self.frame {
                        continue; // a stale frame from before the latest seek
                    }
                    self.count = frame.count;
                    self.fps = frame.fps;
                    self.displayed = Some(frame.requested);
                    if self.store.is_none() {
                        if let (Some(path), false) = (self.labels_path.clone(), self.video_name.is_empty()) {
                            match Store::open(&path, &self.video_name, self.fps) {
                                Ok(store) => {
                                    self.status = format!("標註存到 {}", path.display());
                                    self.store = Some(store);
                                    if let Some(events) = self.pending_events.take() {
                                        self.load_events(&events);
                                    }
                                    if let Some(start) = self.start_frame.take() {
                                        self.frame = start.min(self.count.saturating_sub(1));
                                        self.dirty = true;
                                    }
                                }
                                Err(e) => self.error = Some(e),
                            }
                        }
                    }
                    match &mut self.texture {
                        Some(texture) => texture.set(frame.image, egui::TextureOptions::LINEAR),
                        None => self.texture = Some(ctx.load_texture("frame", frame.image, egui::TextureOptions::LINEAR)),
                    }
                }
                Ok(None) => self.playing = false,
                Err(e) => {
                    self.error = Some(e);
                    self.playing = false;
                    self.preview = None;
                }
            }
        }
    }

    fn drive_playback(&mut self, ctx: &egui::Context) {
        if self.playing {
            let target = self.play_start + (self.last_tick.elapsed().as_secs_f64() * self.fps * self.speed as f64) as u64;
            self.frame = target.min(self.count.saturating_sub(1));
            self.playing = target < self.count;
            self.dirty = self.displayed != Some(self.frame);
        }
        if self.dirty
            && self.requested != Some(self.frame)
            && (!self.scrubbing || self.last_seek.elapsed() >= Duration::from_millis(50))
            && let Some(preview) = &self.preview
        {
            let sent = if self.playing { preview.request_playback(self.frame) } else { preview.request(self.frame) };
            match sent {
                Ok(()) => {
                    self.requested = Some(self.frame);
                    self.last_seek = Instant::now();
                    self.dirty = false;
                }
                Err(e) => {
                    self.error = Some(e);
                    self.preview = None;
                    self.playing = false;
                }
            }
        }
        if self.requested == Some(self.frame) {
            self.dirty = false;
        }
        if self.playing || self.dirty || self.displayed != Some(self.frame) || self.picker.is_some() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            if ui.button("開啟影片…").clicked() {
                self.request_picker(PickKind::Video, ctx);
            }
            if ui.add_enabled(self.video_path.is_some(), egui::Button::new("載入偵測結果…")).clicked() {
                self.request_picker(PickKind::Events, ctx);
            }
            ui.separator();
            if !self.video_name.is_empty() {
                ui.label(egui::RichText::new(&self.video_name).strong());
            }
            if let Some(path) = &self.labels_path {
                ui.weak(format!("→ {}", path.display()));
            }
        });
        if let Some(error) = self.error.clone() {
            ui.horizontal(|ui| {
                ui.colored_label(RED, error);
                if ui.small_button("知道了").clicked() {
                    self.error = None;
                }
            });
        } else {
            ui.weak(&self.status);
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button(if self.playing { "暫停 (空白)" } else { "播放 (空白)" }).clicked() {
                self.toggle_play();
            }
            for (label, delta) in [("-1秒 (J)", -(self.fps.round() as i64)), ("-10幀 (Z)", -10), ("-1幀 (A)", -1), ("+1幀 (D)", 1), ("+10幀 (C)", 10), ("+1秒 (L)", self.fps.round() as i64)] {
                if ui.button(label).clicked() {
                    self.step(delta);
                }
            }
            ui.separator();
            let before = self.speed;
            for (value, text) in [(0.25_f32, "0.25x"), (0.5, "0.5x"), (1.0, "1x"), (2.0, "2x")] {
                ui.selectable_value(&mut self.speed, value, text);
            }
            if before != self.speed && self.playing {
                self.start_play();
            }
            ui.separator();
            let fail = egui::Button::new(egui::RichText::new("沒讓行人 (F)").strong().color(egui::Color32::WHITE)).fill(RED);
            if ui.add(fail).clicked() {
                self.mark(FAILURE, "manual", self.frame);
            }
            let ok = egui::Button::new(egui::RichText::new("有讓行人 (Y)").strong().color(egui::Color32::WHITE)).fill(GREEN);
            if ui.add(ok).clicked() {
                self.mark(YIELDED, "manual", self.frame);
            }
            if ui.button("復原 (U)").clicked() {
                self.undo();
            }
            let none_ok = self.store.as_ref().is_some_and(|s| s.labels.is_empty());
            if ui.add_enabled(none_ok, egui::Button::new("整支沒有事件 (N)")).clicked() {
                self.declare_none();
            }
        });
        ui.horizontal(|ui| {
            ui.monospace(format!(
                "第 {} / {} 幀    {} / {}    {}x",
                self.frame + 1,
                self.count,
                clock(time_of(self.frame + 1, self.fps)),
                clock(time_of(self.count, self.fps)),
                self.speed
            ));
            if self.store.as_ref().is_some_and(|s| s.declared_none) {
                ui.colored_label(GREEN, "已宣告:沒有事件");
            }
        });
    }

    /// Whole-video bar (click or drag to seek) and a zoomed bar around the playhead for precise placement.
    fn timeline(&mut self, ui: &mut egui::Ui) {
        let width = ui.available_width();
        let candidate_frames: Vec<(u64, Decision)> = self.candidates.iter().map(|c| (self.candidate_frame(c.time_s), c.decision)).collect();
        let marks: Vec<(u64, bool)> = self.store.as_ref().map(|s| s.labels.iter().map(|l| (l.frame.saturating_sub(1), l.kind == FAILURE)).collect()).unwrap_or_default();

        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 44.0), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4, egui::Color32::from_gray(38));
        let to_x = |frame: u64| labels::frame_to_x(frame, rect.left() + 4.0, rect.width() - 8.0, self.count);
        for (frame, decision) in &candidate_frames {
            let colour = match decision {
                Decision::Pending => AMBER,
                Decision::Confirmed => GREEN,
                Decision::Rejected => GREY,
            };
            painter.circle_filled(egui::pos2(to_x(*frame), rect.top() + 8.0), 5.0, colour);
        }
        for (frame, failure) in &marks {
            let x = to_x(*frame);
            painter.line_segment([egui::pos2(x, rect.top() + 16.0), egui::pos2(x, rect.bottom() - 2.0)], egui::Stroke::new(3.0_f32, if *failure { RED } else { GREEN }));
        }
        let head = to_x(self.frame);
        painter.line_segment([egui::pos2(head, rect.top()), egui::pos2(head, rect.bottom())], egui::Stroke::new(2.0_f32, egui::Color32::WHITE));
        if let Some(position) = response.interact_pointer_pos() && (response.clicked() || response.dragged()) {
            self.seek(labels::x_to_frame(position.x, rect.left() + 4.0, rect.width() - 8.0, self.count));
            self.scrubbing = response.dragged();
        }
        if response.drag_stopped() {
            self.scrubbing = false;
        }
        if let Some(position) = response.hover_pos() {
            let frame = labels::x_to_frame(position.x, rect.left() + 4.0, rect.width() - 8.0, self.count);
            response.on_hover_text(clock(time_of(frame + 1, self.fps)));
        }

        // Zoomed bar: ±3 s around the playhead, a tick every half second.
        let (start, end) = labels::zoom_window(self.frame, self.count, self.fps, 3.0);
        let (zoom, zoom_response) = ui.allocate_exact_size(egui::vec2(width, 54.0), egui::Sense::click_and_drag());
        let zoom_painter = ui.painter_at(zoom);
        zoom_painter.rect_filled(zoom, 4, egui::Color32::from_gray(30));
        let span = end.saturating_sub(start).max(1) as f32;
        let zx = |frame: u64| zoom.left() + 4.0 + (zoom.width() - 8.0) * (frame.saturating_sub(start) as f32 / span).min(1.0);
        if self.count > 0 {
            // Ticks sit on whole/half seconds of the displayed (one-based) time, hence the -1 to the zero-based index.
            let half_second = (self.fps / 2.0).round().max(1.0) as u64;
            let mut tick = (start + 1).div_ceil(half_second) * half_second;
            while tick - 1 <= end {
                let x = zx(tick - 1);
                let whole = tick % (half_second * 2) == 0;
                zoom_painter.line_segment([egui::pos2(x, zoom.bottom() - if whole { 14.0 } else { 7.0 }), egui::pos2(x, zoom.bottom())], egui::Stroke::new(1.0_f32, egui::Color32::from_gray(120)));
                if whole {
                    zoom_painter.text(egui::pos2(x + 2.0, zoom.bottom() - 15.0), egui::Align2::LEFT_BOTTOM, clock(time_of(tick, self.fps)), egui::FontId::monospace(10.0), egui::Color32::from_gray(150));
                }
                tick += half_second;
            }
        }
        for (frame, decision) in &candidate_frames {
            if (start..=end).contains(frame) {
                let colour = match decision {
                    Decision::Pending => AMBER,
                    Decision::Confirmed => GREEN,
                    Decision::Rejected => GREY,
                };
                zoom_painter.circle_filled(egui::pos2(zx(*frame), zoom.top() + 9.0), 6.0, colour);
            }
        }
        for (frame, failure) in &marks {
            if (start..=end).contains(frame) {
                let x = zx(*frame);
                zoom_painter.line_segment([egui::pos2(x, zoom.top() + 18.0), egui::pos2(x, zoom.bottom() - 16.0)], egui::Stroke::new(3.0_f32, if *failure { RED } else { GREEN }));
            }
        }
        let zoom_head = zx(self.frame);
        zoom_painter.line_segment([egui::pos2(zoom_head, zoom.top()), egui::pos2(zoom_head, zoom.bottom())], egui::Stroke::new(2.0_f32, egui::Color32::WHITE));
        if let Some(position) = zoom_response.interact_pointer_pos() && (zoom_response.clicked() || zoom_response.dragged()) {
            let fraction = ((position.x - zoom.left() - 4.0) / (zoom.width() - 8.0)).clamp(0.0, 1.0);
            self.seek(start + (fraction * span).round() as u64);
            self.scrubbing = zoom_response.dragged();
        }
        if zoom_response.drag_stopped() {
            self.scrubbing = false;
        }
        ui.weak("上:整支影片(點擊或拖曳跳轉)  下:目前位置前後 3 秒(精準對位)  紅線=沒讓行人 綠線=有讓行人  圓點=偵測候選(橘=待確認 綠=已確認 灰=已否決)");
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) -> Vec<Action> {
        let mut actions = Vec::new();
        ui.heading("標記");
        let sorted = self.store.as_ref().map(|s| s.sorted()).unwrap_or_default();
        if sorted.is_empty() {
            ui.weak("還沒有標記。暫停在車輛前輪壓上斑馬線的那一幀,按 F。");
        }
        for (index, label) in sorted.iter().enumerate() {
            ui.horizontal(|ui| {
                let (text, colour) = if label.kind == FAILURE { ("沒讓行人", RED) } else { ("有讓行人", GREEN) };
                let row = egui::RichText::new(format!("{}  {}", clock(label.time_s), text)).color(colour);
                if ui.selectable_label(false, row).clicked() {
                    actions.push(Action::Jump(label.frame.saturating_sub(1)));
                }
                if ui.small_button("刪除").clicked() {
                    actions.push(Action::DeleteLabel(index));
                }
            });
        }
        ui.add_space(12.0);
        ui.heading("偵測候選");
        if self.candidates.is_empty() {
            ui.weak("按「載入偵測結果…」選 events.jsonl,逐一確認或否決。");
        }
        for (index, candidate) in self.candidates.iter().enumerate() {
            ui.horizontal(|ui| {
                let (state, colour) = match candidate.decision {
                    Decision::Pending => ("待確認", AMBER),
                    Decision::Confirmed => ("已確認", GREEN),
                    Decision::Rejected => ("已否決", GREY),
                };
                if ui.selectable_label(false, egui::RichText::new(format!("{}  {}", clock(candidate.time_s), state)).color(colour)).clicked() {
                    actions.push(Action::Jump(self.candidate_frame(candidate.time_s)));
                }
                match candidate.decision {
                    Decision::Pending => {
                        if ui.small_button("確認").clicked() {
                            actions.push(Action::Confirm(index));
                        }
                        if ui.small_button("否決").clicked() {
                            actions.push(Action::Reject(index));
                        }
                    }
                    _ => {
                        if ui.small_button("改回待確認").clicked() {
                            actions.push(Action::ResetCandidate(index));
                        }
                    }
                }
            });
        }
        actions
    }

    fn apply(&mut self, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Jump(frame) => self.seek(frame),
                Action::DeleteLabel(index) => {
                    if let Some(store) = &mut self.store {
                        match store.remove_sorted(index) {
                            Ok(Some(label)) => self.reset_candidate_for(&label),
                            Ok(None) => {}
                            Err(e) => self.error = Some(e),
                        }
                    }
                }
                Action::Confirm(index) => {
                    let frame = self.candidate_frame(self.candidates[index].time_s);
                    self.candidates[index].decision = Decision::Confirmed;
                    self.mark(FAILURE, "confirmed_candidate", frame);
                    self.save_review();
                    self.seek(frame);
                }
                Action::Reject(index) => {
                    self.candidates[index].decision = Decision::Rejected;
                    self.save_review();
                }
                Action::ResetCandidate(index) => {
                    let time_s = self.candidates[index].time_s;
                    if self.candidates[index].decision == Decision::Confirmed {
                        let position = self.store.as_ref().and_then(|s| {
                            s.sorted().iter().position(|l| l.source == "confirmed_candidate" && (l.time_s - time_s).abs() < 0.0015)
                        });
                        if let (Some(position), Some(store)) = (position, &mut self.store) {
                            if let Err(e) = store.remove_sorted(position) {
                                self.error = Some(e);
                            }
                        }
                    }
                    self.candidates[index].decision = Decision::Pending;
                    self.save_review();
                }
            }
        }
    }

    fn video_area(&self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
        ui.painter().rect_filled(rect, 0, egui::Color32::BLACK);
        if let Some(texture) = &self.texture {
            let size = texture.size_vec2();
            let scale = (rect.width() / size.x).min(rect.height() / size.y);
            let shown = egui::Rect::from_center_size(rect.center(), size * scale);
            ui.painter().image(texture.id(), shown, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        } else {
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "按左上角「開啟影片…」", egui::FontId::proportional(20.0), egui::Color32::from_gray(180));
        }
    }

    fn capture_screenshot(&mut self, ctx: &egui::Context) {
        let Some(path) = self.screenshot.clone() else { return };
        if self.started.elapsed() > Duration::from_secs(25) {
            eprintln!("Screenshot timed out");
            std::process::exit(1);
        }
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = shot {
            let bytes: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
            image::save_buffer(&path, &bytes, image.size[0] as u32, image.size[1] as u32, image::ColorType::Rgba8).expect("save screenshot");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let ready = self.store.is_some() && self.texture.is_some() && self.displayed == Some(self.frame) && self.started.elapsed() > Duration::from_secs(2);
        if !self.screenshot_requested && ready {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            self.screenshot_requested = true;
        }
        ctx.request_repaint_after(Duration::from_millis(50));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll(ctx);
        self.handle_keys(ctx);
        let mut actions = Vec::new();
        egui::TopBottomPanel::top("top").show(ctx, |ui| self.top_bar(ui, ctx));
        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            self.controls(ui);
            self.timeline(ui);
        });
        egui::SidePanel::right("labels").min_width(300.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| actions = self.side_panel(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| self.video_area(ui));
        self.apply(actions);
        self.drive_playback(ctx);
        self.capture_screenshot(ctx);
    }
}

fn main() -> eframe::Result {
    let mut app = App::new();
    let mut video = None;
    let mut labels_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--video" => video = args.next().map(PathBuf::from),
            "--labels" => labels_path = args.next().map(PathBuf::from),
            "--events" => app.pending_events = args.next().map(PathBuf::from),
            "--frame" => app.start_frame = args.next().and_then(|v| v.parse::<u64>().ok()).map(|f| f.saturating_sub(1)),
            "--screenshot" => app.screenshot = args.next().map(PathBuf::from),
            other => {
                eprintln!("未知參數:{other}\n用法:event_labeler [--video V] [--labels L.jsonl] [--events E.jsonl] [--frame N]");
                std::process::exit(2);
            }
        }
    }
    if let Some(path) = video {
        app.open_video(path, labels_path);
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1360.0, 860.0]).with_min_inner_size([980.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "事件標註工具",
        options,
        Box::new(move |cc| {
            let mut fonts = egui::FontDefinitions::default();
            for path in [std::env::var_os("TRAFFIC_FONT").map(PathBuf::from), Some(PathBuf::from("C:/Windows/Fonts/msjh.ttc"))].into_iter().flatten() {
                if let Ok(bytes) = std::fs::read(path) {
                    fonts.font_data.insert("chinese".into(), egui::FontData::from_owned(bytes).into());
                    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                        fonts.families.entry(family).or_default().insert(0, "chinese".into());
                    }
                    break;
                }
            }
            cc.egui_ctx.set_fonts(fonts);
            cc.egui_ctx.all_styles_mut(|style| {
                style.spacing.item_spacing = egui::vec2(8.0, 6.0);
                style.spacing.button_padding = egui::vec2(10.0, 6.0);
                for (kind, size) in [(egui::TextStyle::Heading, 20.0), (egui::TextStyle::Body, 14.0), (egui::TextStyle::Button, 14.0), (egui::TextStyle::Small, 12.0)] {
                    style.text_styles.insert(kind, egui::FontId::proportional(size));
                }
            });
            Ok(Box::new(app))
        }),
    )
}
