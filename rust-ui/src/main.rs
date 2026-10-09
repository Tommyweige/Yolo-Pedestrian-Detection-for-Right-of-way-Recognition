#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod video;
use eframe::egui;
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    time::{Duration, Instant},
};

fn project_root() -> PathBuf {
    std::env::var_os("TRAFFIC_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .into()
        })
}

fn hidden_command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
}

struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    events: Receiver<Value>,
}

impl Worker {
    fn start(mode: &str) -> Result<Self, String> {
        let python = std::env::var_os("TRAFFIC_PYTHON").unwrap_or_else(|| "python".into());
        let root = project_root();
        let mut child = hidden_command(python)
            .arg("-u")
            .arg(root.join("desktop_bridge.py"))
            .arg(mode)
            .current_dir(root)
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("無法啟動 Python：{e}。請設定 TRAFFIC_PYTHON。"))?;
        let (sender, events) = mpsc::sync_channel(32);
        let output = child.stdout.take().unwrap();
        let errors = child.stderr.take().unwrap();
        let error_sender = sender.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(errors).lines().map_while(Result::ok) {
                if error_sender
                    .send(json!({"type": "log", "message": line}))
                    .is_err()
                {
                    break;
                }
            }
        });
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines().map_while(Result::ok) {
                let event = serde_json::from_str(&line)
                    .unwrap_or_else(|_| json!({"type": "log", "message": line}));
                if sender.send(event).is_err() {
                    return;
                }
            }
            let _ = sender.send(json!({"type": "exited"}));
        });
        let input = child.stdin.take();
        Ok(Self {
            child,
            input,
            events,
        })
    }

    fn send(&mut self, value: Value) -> Result<(), String> {
        let input = self.input.as_mut().ok_or("Python 連線已關閉")?;
        writeln!(input, "{value}")
            .and_then(|_| input.flush())
            .map_err(|e| e.to_string())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.input.take();
        // Give Python time to release captures and terminate its prediction child.
        for _ in 0..10 {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        #[cfg(windows)]
        let _ = hidden_command("taskkill")
            .args(["/PID", &self.child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone, Copy)]
enum PickKind {
    Videos,
    Reference,
    Output,
}

fn start_picker(
    ctx: egui::Context,
    choose: impl FnOnce() -> Option<Vec<PathBuf>> + Send + 'static,
) -> Result<Receiver<Option<Vec<PathBuf>>>, String> {
    let (output, result) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("file-picker".into())
        .spawn(move || {
            let _ = output.send(choose());
            ctx.request_repaint();
        })
        .map_err(|e| format!("無法開啟檔案選擇視窗：{e}"))?;
    Ok(result)
}

struct Desktop {
    videos: Vec<PathBuf>,
    output: Option<PathBuf>,
    picker: Option<(PickKind, Receiver<Option<Vec<PathBuf>>>)>,
    model: usize,
    task: usize,
    dark: bool,
    calibration: bool,
    angle: f32,
    applied_angle: f32,
    preview_path: Option<PathBuf>,
    preview: Option<video::Preview>,
    detection: Option<Worker>,
    texture: Option<egui::TextureHandle>,
    frame: u64,
    count: u64,
    fps: f64,
    playing: bool,
    pending: bool,
    dirty: bool,
    last_frame: Instant,
    play_start_frame: u64,
    displayed_frame: Option<u64>,
    requested_frame: Option<u64>,
    scrubbing: bool,
    last_seek_send: Instant,
    single_progress: f32,
    batch_progress: f32,
    status: String,
    error: Option<String>,
    logs: String,
    screenshot: Option<PathBuf>,
    screenshot_requested: bool,
    check_playback: bool,
    check_folder: bool,
    check_detection: bool,
    detection_check_started: bool,
    folder_updates: usize,
    autoplay_check: bool,
    started: Instant,
}

impl Default for Desktop {
    fn default() -> Self {
        Self {
            videos: vec![],
            output: None,
            picker: None,
            model: 0,
            task: 0,
            dark: false,
            calibration: false,
            angle: 0.0,
            applied_angle: 0.0,
            preview_path: None,
            preview: None,
            detection: None,
            texture: None,
            frame: 0,
            count: 0,
            fps: 30.0,
            playing: false,
            pending: false,
            dirty: false,
            last_frame: Instant::now(),
            play_start_frame: 0,
            displayed_frame: None,
            requested_frame: None,
            scrubbing: false,
            last_seek_send: Instant::now(),
            single_progress: 0.0,
            batch_progress: 0.0,
            status: "選擇影片與輸出資料夾即可開始。".into(),
            error: None,
            logs: String::new(),
            screenshot: std::env::var_os("TRAFFIC_SCREENSHOT").map(PathBuf::from),
            screenshot_requested: false,
            check_playback: false,
            check_folder: false,
            check_detection: false,
            detection_check_started: false,
            folder_updates: 0,
            autoplay_check: false,
            started: Instant::now(),
        }
    }
}

const MODELS: [&str; 3] = ["yolov8s", "yolov8l", "yolov8x6"];

#[derive(Clone, Copy)]
struct Palette {
    background: egui::Color32,
    surface: egui::Color32,
    text: egui::Color32,
    muted: egui::Color32,
    border: egui::Color32,
    accent: egui::Color32,
    tint: egui::Color32,
    video: egui::Color32,
}

impl Palette {
    fn new(dark: bool) -> Self {
        let color = egui::Color32::from_rgb;
        if dark {
            Self {
                background: color(17, 35, 48),
                surface: color(25, 46, 61),
                text: color(233, 241, 245),
                muted: color(168, 186, 199),
                border: color(55, 78, 94),
                accent: color(111, 211, 211),
                tint: color(32, 69, 79),
                video: color(10, 24, 34),
            }
        } else {
            Self {
                background: color(242, 246, 248),
                surface: egui::Color32::WHITE,
                text: color(33, 54, 69),
                muted: color(92, 112, 126),
                border: color(218, 229, 235),
                accent: color(17, 116, 124),
                tint: color(231, 244, 244),
                video: color(18, 38, 51),
            }
        }
    }
}

fn apply_theme(ctx: &egui::Context, dark: bool) {
    let palette = Palette::new(dark);
    ctx.set_theme(if dark {
        egui::ThemePreference::Dark
    } else {
        egui::ThemePreference::Light
    });
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    visuals.override_text_color = Some(palette.text);
    visuals.weak_text_color = Some(palette.muted);
    visuals.panel_fill = palette.background;
    visuals.window_fill = palette.surface;
    visuals.faint_bg_color = palette.tint;
    visuals.extreme_bg_color = palette.background;
    visuals.selection.bg_fill = palette.tint;
    visuals.selection.stroke = egui::Stroke::new(1.5_f32, palette.accent);
    visuals.slider_trailing_fill = true;
    visuals.hyperlink_color = palette.accent;
    visuals.error_fg_color = if dark {
        egui::Color32::from_rgb(255, 174, 159)
    } else {
        egui::Color32::from_rgb(179, 61, 65)
    };
    for widget in [
        &mut visuals.widgets.noninteractive,
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
    ] {
        widget.corner_radius = egui::CornerRadius::same(8);
        widget.bg_fill = palette.tint;
        widget.weak_bg_fill = palette.surface;
        widget.bg_stroke = egui::Stroke::new(1.0_f32, palette.border);
        widget.fg_stroke = egui::Stroke::new(1.5_f32, palette.text);
        widget.expansion = 0.0;
    }
    visuals.widgets.hovered.weak_bg_fill = palette.tint;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.5_f32, palette.accent);
    visuals.widgets.active = visuals.widgets.hovered;
    ctx.set_visuals(visuals);
}

fn card(palette: Palette) -> egui::Frame {
    egui::Frame::NONE
        .fill(palette.surface)
        .stroke(egui::Stroke::new(1.0_f32, palette.border))
        .corner_radius(14)
        .inner_margin(18)
}

fn crossing_mark(ui: &mut egui::Ui, size: f32, accent: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 10, accent);
    for index in 0..3 {
        let x = rect.left() + size * (0.22 + index as f32 * 0.21);
        painter.add(egui::Shape::convex_polygon(
            vec![
                egui::pos2(x, rect.top() + size * 0.27),
                egui::pos2(x + size * 0.13, rect.top() + size * 0.27),
                egui::pos2(x + size * 0.05, rect.top() + size * 0.73),
                egui::pos2(x - size * 0.08, rect.top() + size * 0.73),
            ],
            egui::Color32::WHITE,
            egui::Stroke::NONE,
        ));
    }
}

fn primary_button(
    ui: &mut egui::Ui,
    text: &str,
    enabled: bool,
    palette: Palette,
) -> egui::Response {
    ui.scope(|ui| {
        let ink = if ui.visuals().dark_mode {
            palette.video
        } else {
            egui::Color32::WHITE
        };
        let style = ui.style_mut();
        style.visuals.override_text_color = Some(ink);
        for widget in [
            &mut style.visuals.widgets.inactive,
            &mut style.visuals.widgets.hovered,
            &mut style.visuals.widgets.active,
        ] {
            widget.weak_bg_fill = palette.accent;
            widget.bg_stroke = egui::Stroke::new(1.0_f32, palette.accent);
            widget.fg_stroke = egui::Stroke::new(1.5_f32, ink);
        }
        style.visuals.widgets.hovered.weak_bg_fill = palette.accent.gamma_multiply(0.86);
        style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(2.0_f32, palette.text);
        ui.add_enabled(
            enabled,
            egui::Button::new(egui::RichText::new(text).strong().color(ink))
                .min_size(egui::vec2(ui.available_width(), 46.0)),
        )
    })
    .inner
}

fn path_name(path: &std::path::Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

fn timestamp(frame: u64, fps: f64) -> String {
    let seconds = (frame as f64 / fps) as u64;
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

fn section_label(ui: &mut egui::Ui, text: &str, palette: Palette) {
    ui.label(egui::RichText::new(text).size(12.0).color(palette.muted));
}

fn progress(ui: &mut egui::Ui, title: &str, value: f32, palette: Palette) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).size(13.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("{:.0}%", value * 100.0))
                    .monospace()
                    .size(12.0)
                    .color(palette.muted),
            );
        });
    });
    ui.add(
        egui::ProgressBar::new(value)
            .desired_height(5.0)
            .fill(palette.accent)
            .corner_radius(3),
    );
}

impl Desktop {
    fn request_picker(&mut self, kind: PickKind, ctx: &egui::Context) {
        if self.picker.is_some() {
            return;
        }
        let choose = move || {
            let dialog = rfd::FileDialog::new();
            match kind {
                PickKind::Output => dialog
                    .set_title("選擇輸出資料夾")
                    .pick_folder()
                    .map(|p| vec![p]),
                PickKind::Reference => dialog
                    .set_title("選擇校正參考影片")
                    .add_filter("影片", &["mp4", "avi", "mkv"])
                    .pick_file()
                    .map(|p| vec![p]),
                PickKind::Videos => dialog
                    .set_title("匯入影片")
                    .add_filter("影片", &["mp4", "avi", "mkv"])
                    .pick_files(),
            }
        };
        match start_picker(ctx.clone(), choose) {
            Ok(result) => self.picker = Some((kind, result)),
            Err(error) => self.error = Some(error),
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui, palette: Palette) {
        let busy = self.detection.is_some();
        card(palette).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(egui::RichText::new("分析設定").size(18.0).strong());
            section_label(ui, "設定本次影片的處理方式", palette);
            ui.add_space(8.0);
            ui.add_enabled_ui(!busy, |ui| {
                section_label(ui, "偵測模式", palette);
                for (index, title, description) in [
                    (0, "闖紅燈", "分析車輛通過號誌的行為"),
                    (1, "未禮讓行人", "分析車輛與行人的路權關係"),
                ] {
                    let mut text = egui::text::LayoutJob::default();
                    text.append(
                        title,
                        0.0,
                        egui::TextFormat {
                            font_id: egui::FontId::proportional(15.0),
                            color: palette.text,
                            ..Default::default()
                        },
                    );
                    text.append(
                        &format!("\n{description}"),
                        0.0,
                        egui::TextFormat {
                            font_id: egui::FontId::proportional(12.0),
                            color: palette.muted,
                            ..Default::default()
                        },
                    );
                    let selected = self.task == index;
                    let response = ui.add(
                        egui::Button::new(text)
                            .selected(selected)
                            .min_size(egui::vec2(ui.available_width(), 58.0)),
                    );
                    response.widget_info(|| {
                        egui::WidgetInfo::selected(
                            egui::WidgetType::RadioButton,
                            ui.is_enabled(),
                            selected,
                            title,
                        )
                    });
                    let center = response.rect.right_top() + egui::vec2(-15.0, 17.0);
                    ui.painter().circle_stroke(
                        center,
                        5.0,
                        egui::Stroke::new(
                            1.0_f32,
                            if selected {
                                palette.accent
                            } else {
                                palette.muted
                            },
                        ),
                    );
                    if selected {
                        ui.painter().circle_filled(center, 2.5, palette.accent);
                    }
                    if response.clicked() {
                        self.task = index;
                    }
                }
                ui.add_space(6.0);
                section_label(ui, "模型", palette);
                let labels = ["YOLOv8s  ·  輕量", "YOLOv8l  ·  標準", "YOLOv8x6  ·  大型"];
                egui::ComboBox::from_id_salt("model")
                    .width(ui.available_width() - 28.0)
                    .selected_text(labels[self.model])
                    .show_ui(ui, |ui| {
                        for (index, label) in labels.iter().enumerate() {
                            ui.selectable_value(&mut self.model, index, *label);
                        }
                    });
                ui.add_space(6.0);
                section_label(ui, "輸出位置", palette);
                if ui
                    .add_enabled(
                        self.picker.is_none(),
                        egui::Button::new(
                            self.output
                                .as_ref()
                                .map(|p| path_name(p))
                                .unwrap_or("選擇輸出資料夾".into()),
                        )
                        .min_size(egui::vec2(ui.available_width(), 40.0))
                        .truncate(),
                    )
                    .on_hover_text(
                        self.output
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or("選擇存放偵測結果的資料夾".into()),
                    )
                    .clicked()
                {
                    self.request_picker(PickKind::Output, ui.ctx());
                }
                if self.applied_angle != 0.0 {
                    section_label(
                        ui,
                        &format!("旋轉校正已套用：{:.0}°", self.applied_angle),
                        palette,
                    );
                }
            });
            ui.add_space(8.0);
            if busy {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("正在分析影片");
                });
                if ui
                    .add_sized([ui.available_width(), 40.0], egui::Button::new("取消分析"))
                    .clicked()
                {
                    if let Some(worker) = &mut self.detection
                        && let Err(error) = worker.send(json!({"cancel": true}))
                    {
                        self.error = Some(error);
                    }
                    self.status = "正在取消；旋轉轉檔完成後會停止。".into();
                }
            } else {
                let ready =
                    !self.videos.is_empty() && self.output.is_some() && self.picker.is_none();
                if primary_button(ui, "開始分析", ready, palette).clicked() {
                    self.start_detection(if self.task == 0 { "tf" } else { "zebra" });
                }
                if !ready {
                    section_label(ui, "先匯入影片並選擇輸出資料夾", palette);
                }
            }
        });
        ui.add_space(8.0);
        card(palette).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(egui::RichText::new("處理進度").size(16.0).strong());
            ui.add_space(4.0);
            progress(ui, "當前影片", self.single_progress, palette);
            ui.add_space(4.0);
            progress(ui, "整體批次", self.batch_progress, palette);
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(&self.status)
                    .size(12.0)
                    .color(palette.muted),
            );
        });
    }

    fn workspace(&mut self, ui: &mut egui::Ui, palette: Palette) {
        let busy = self.detection.is_some();
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(if self.calibration {
                        "旋轉校正"
                    } else {
                        "影片工作區"
                    })
                    .size(22.0)
                    .strong(),
                );
                section_label(
                    ui,
                    if self.calibration {
                        "調整拍攝角度，確認後套用到本次分析"
                    } else {
                        "預覽畫面，確認素材後開始分析"
                    },
                    palette,
                );
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_enabled_ui(!busy && self.picker.is_none(), |ui| {
                    if ui
                        .button(if self.calibration {
                            "更換參考影片"
                        } else {
                            "匯入影片"
                        })
                        .clicked()
                    {
                        self.request_picker(
                            if self.calibration {
                                PickKind::Reference
                            } else {
                                PickKind::Videos
                            },
                            ui.ctx(),
                        );
                    }
                });
            });
        });
        ui.add_space(10.0);
        card(palette).inner_margin(16).show(ui, |ui| {
            ui.set_width(ui.available_width());
            let width = ui.available_width();
            let height = (width * 9.0 / 16.0).clamp(200.0, 340.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
            ui.painter().rect_filled(rect, 10, palette.video);
            if let Some(texture) = &self.texture {
                let image = texture.size_vec2();
                let angle = if self.calibration {
                    -self.angle.to_radians()
                } else {
                    0.0
                };
                let rotation = egui::emath::Rot2::from_angle(angle);
                let bounds = egui::vec2(
                    image.x * angle.cos().abs() + image.y * angle.sin().abs(),
                    image.x * angle.sin().abs() + image.y * angle.cos().abs(),
                );
                let scale = ((width - 24.0) / bounds.x).min((height - 24.0) / bounds.y);
                let half = image * scale * 0.5;
                let mut mesh = egui::Mesh::with_texture(texture.id());
                for (position, uv) in [
                    (-half, egui::pos2(0.0, 0.0)),
                    (egui::vec2(half.x, -half.y), egui::pos2(1.0, 0.0)),
                    (half, egui::pos2(1.0, 1.0)),
                    (egui::vec2(-half.x, half.y), egui::pos2(0.0, 1.0)),
                ] {
                    mesh.vertices.push(egui::epaint::Vertex {
                        pos: rect.center() + rotation * position,
                        uv,
                        color: egui::Color32::WHITE,
                    });
                }
                mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
                ui.painter().add(egui::Shape::mesh(mesh));
            } else {
                let center = rect.center();
                let color = egui::Color32::from_rgb(70, 111, 127);
                for index in 0..4 {
                    let x = center.x - 48.0 + index as f32 * 26.0;
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x, center.y - 70.0),
                            egui::vec2(15.0, 40.0),
                        ),
                        3,
                        color,
                    );
                }
                ui.painter().text(
                    center + egui::vec2(0.0, 4.0),
                    egui::Align2::CENTER_CENTER,
                    "從一部影片開始",
                    egui::FontId::proportional(21.0),
                    egui::Color32::from_rgb(233, 241, 245),
                );
                ui.painter().text(
                    center + egui::vec2(0.0, 38.0),
                    egui::Align2::CENTER_CENTER,
                    "支援 MP4、AVI 與 MKV",
                    egui::FontId::proportional(13.0),
                    egui::Color32::from_rgb(171, 194, 208),
                );
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(
                            self.preview_path
                                .as_ref()
                                .map(|p| path_name(p))
                                .unwrap_or("尚未匯入影片".into()),
                        )
                        .strong(),
                    )
                    .truncate(),
                )
                .on_hover_text(
                    self.preview_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.count > 0 {
                        section_label(ui, &format!("{:.0} FPS", self.fps), palette);
                    }
                });
            });
            ui.add_enabled_ui(self.preview.is_some() && self.count > 0 && !busy, |ui| {
                ui.scope(|ui| {
                    ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(120.0);
                    ui.spacing_mut().interact_size.y = 20.0;
                    let response = ui
                        .add(
                            egui::Slider::new(&mut self.frame, 0..=self.count.saturating_sub(1))
                                .show_value(false)
                                .handle_shape(egui::style::HandleShape::Circle)
                                .trailing_fill(true),
                        )
                        .on_hover_text("拖曳定位影格；也可使用方向鍵調整");
                    self.scrubbing = response.dragged();
                    if response.changed() {
                        self.playing = false;
                        self.dirty = true;
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(if self.playing { "暫停" } else { "播放" })
                                .min_size(egui::vec2(80.0, 36.0)),
                        )
                        .clicked()
                    {
                        self.playing = !self.playing;
                        self.play_start_frame = self.frame;
                        self.last_frame = Instant::now();
                    }
                    if ui.button("停止").clicked() {
                        self.playing = false;
                        self.frame = 0;
                        self.dirty = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}  /  {}",
                                timestamp(self.frame, self.fps),
                                timestamp(self.count, self.fps)
                            ))
                            .monospace()
                            .size(12.0)
                            .color(palette.muted),
                        );
                    });
                });
            });
        });
        ui.add_space(10.0);
        if self.calibration {
            card(palette).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(egui::RichText::new("畫面角度").size(16.0).strong());
                ui.add_enabled_ui(!busy, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().slider_width = (ui.available_width() - 150.0).max(100.0);
                        if ui
                            .add(egui::Slider::new(&mut self.angle, -45.0..=45.0).suffix("°"))
                            .changed()
                        {
                            self.playing = false;
                            self.dirty = true;
                        }
                        if ui.button("重設").clicked() {
                            self.angle = 0.0;
                            self.dirty = true;
                        }
                    });
                    if ui.button("套用旋轉角度").clicked() {
                        self.applied_angle = self.angle;
                        self.status = format!("旋轉校正已套用：{:.0}°", self.angle);
                    }
                });
                ui.collapsing("參考影片資訊", |ui| {
                    if let Some(path) = &self.preview_path {
                        ui.label(path.display().to_string());
                    }
                    if !self.logs.is_empty() {
                        ui.label(&self.logs);
                    }
                });
            });
        } else {
            card(palette).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("影片清單").size(16.0).strong());
                    section_label(ui, &format!("{} 部影片", self.videos.len()), palette);
                });
                if self.videos.is_empty() {
                    section_label(
                        ui,
                        "匯入一部或多部影片，偵測結果將存入指定資料夾。",
                        palette,
                    );
                } else {
                    let mut selected = None;
                    egui::ScrollArea::vertical()
                        .max_height(125.0)
                        .id_salt("video_queue")
                        .show(ui, |ui| {
                            for path in &self.videos {
                                ui.horizontal(|ui| {
                                    let previewing = self.preview_path.as_ref() == Some(path);
                                    if ui
                                        .add_enabled(
                                            !busy,
                                            egui::Button::selectable(previewing, "預覽"),
                                        )
                                        .clicked()
                                    {
                                        selected = Some(path.clone());
                                    }
                                    ui.add(egui::Label::new(path_name(path)).truncate())
                                        .on_hover_text(path.display().to_string());
                                });
                            }
                        });
                    if let Some(path) = selected {
                        self.open_preview(path);
                    }
                }
                ui.add_space(4.0);
                ui.collapsing("完整路徑與執行記錄", |ui| {
                    for path in &self.videos {
                        ui.label(path.display().to_string());
                    }
                    if let Some(path) = &self.output {
                        ui.label(format!("輸出：{}", path.display()));
                    }
                    if !self.logs.is_empty() {
                        ui.label(&self.logs);
                    }
                });
            });
        }
        if let Some(error) = &self.error {
            ui.add_space(10.0);
            card(palette).show(ui, |ui| {
                ui.label(
                    egui::RichText::new("需要處理")
                        .strong()
                        .color(ui.visuals().error_fg_color),
                );
                ui.label(error);
            });
            if ui.button("關閉錯誤訊息").clicked() {
                self.error = None;
            }
        }
    }

    fn open_preview(&mut self, path: PathBuf) {
        self.preview = None;
        self.texture = None;
        self.preview_path = Some(path.clone());
        self.displayed_frame = None;
        self.requested_frame = None;
        self.scrubbing = false;
        self.error = None;
        self.frame = 0;
        self.count = 0;
        self.playing = false;
        self.pending = false;
        self.dirty = true;
        match video::Preview::open(path) {
            Ok(worker) => self.preview = Some(worker),
            Err(error) => self.error = Some(error),
        }
    }

    fn start_detection(&mut self, task: &str) {
        self.error = None;
        self.playing = false;
        let request = json!({"videos": self.videos, "output": self.output,
            "model": MODELS[self.model], "task": task, "angle": self.applied_angle});
        match Worker::start("detect").and_then(|mut worker| {
            worker.send(request)?;
            Ok(worker)
        }) {
            Ok(worker) => {
                self.detection = Some(worker);
                self.single_progress = 0.0;
                self.batch_progress = 0.0;
                self.logs.clear();
                self.status = "正在準備偵測…".into();
            }
            Err(error) => self.error = Some(error),
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if let Some((kind, result)) = &self.picker {
            match result.try_recv() {
                Ok(paths) => {
                    let kind = *kind;
                    self.picker = None;
                    if let Some(paths) = paths {
                        match kind {
                            PickKind::Output => self.output = paths.into_iter().next(),
                            PickKind::Reference => {
                                if let Some(path) = paths.into_iter().next() {
                                    self.open_preview(path);
                                }
                            }
                            PickKind::Videos => {
                                self.videos = paths;
                                if let Some(path) = self.videos.first().cloned() {
                                    self.open_preview(path);
                                }
                            }
                        }
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.picker = None;
                    self.error = Some("檔案選擇程序意外結束，請重試。".into());
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let frames: Vec<_> = self
            .preview
            .as_ref()
            .map(|preview| preview.frames.try_iter().collect())
            .unwrap_or_default();
        for frame in frames {
            match frame {
                Ok(Some(frame)) => {
                    if !self.playing && frame.requested != self.frame {
                        continue;
                    }
                    self.pending = self.requested_frame != Some(frame.requested);
                    self.count = frame.count;
                    self.fps = frame.fps;
                    self.displayed_frame = Some(frame.requested);
                    if self.autoplay_check {
                        self.autoplay_check = false;
                        self.playing = true;
                        self.play_start_frame = self.frame;
                        self.last_frame = Instant::now();
                    }
                    if let Some(texture) = &mut self.texture {
                        texture.set(frame.image, egui::TextureOptions::LINEAR);
                    } else {
                        self.texture = Some(ctx.load_texture(
                            "preview",
                            frame.image,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }
                Ok(None) => {
                    self.pending = false;
                    self.playing = false;
                    self.dirty = false;
                }
                Err(error) => {
                    self.pending = false;
                    self.error = Some(error);
                    self.playing = false;
                    self.preview = None;
                }
            }
        }
        let events: Vec<_> = self
            .detection
            .as_ref()
            .map(|w| w.events.try_iter().collect())
            .unwrap_or_default();
        for event in events {
            match event["type"].as_str().unwrap_or("") {
                "progress" => {
                    let index = event["index"].as_u64().unwrap_or(0) as f32;
                    let current = event["current"].as_u64().unwrap_or(0) as f32;
                    let total = event["total"].as_u64().unwrap_or(1).max(1) as f32;
                    self.single_progress = (current / total).clamp(0.0, 1.0);
                    self.batch_progress = (index / self.videos.len().max(1) as f32).clamp(0.0, 1.0);
                    self.status = event["message"].as_str().unwrap_or("").into();
                }
                "done" => {
                    self.status = "影片處理完成，請至輸出資料夾查看結果。".into();
                    self.single_progress = 1.0;
                    self.batch_progress = 1.0;
                    self.detection = None;
                }
                "cancelled" => {
                    self.status = "偵測已取消。".into();
                    self.detection = None;
                }
                "error" => {
                    self.error = Some(event["message"].as_str().unwrap_or("偵測失敗").into());
                    self.status = "偵測失敗，請檢查錯誤訊息後重試。".into();
                    self.detection = None;
                }
                "exited" if self.detection.is_some() => {
                    self.error = Some("Python 偵測程序意外結束，請檢查執行記錄。".into());
                    self.detection = None;
                }
                "log" => self.append_log(event["message"].as_str().unwrap_or("")),
                _ => {}
            }
        }
    }

    fn append_log(&mut self, line: &str) {
        self.logs.push_str(line);
        self.logs.push('\n');
        if self.logs.len() > 32_000 {
            let cut = self
                .logs
                .char_indices()
                .find(|(i, _)| *i >= 16_000)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.logs.drain(..cut);
        }
    }
}

impl eframe::App for Desktop {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll(ctx);
        if self.check_detection && !self.detection_check_started && self.texture.is_some() {
            self.detection_check_started = true;
            self.start_detection("zebra");
        }
        if self.check_folder && self.texture.is_some() {
            if self.folder_updates == 0 {
                self.request_picker(PickKind::Output, ctx);
            }
            if self.picker.is_some() {
                self.folder_updates += 1;
            }
        }
        if let Some(path) = &self.screenshot {
            if self.preview_path.is_some() && self.preview.is_none() {
                eprintln!("Preview smoke test failed: {:?}", self.error);
                std::process::exit(1);
            }
            if self.check_detection && self.error.is_some() {
                eprintln!("Detection check failed: {:?}", self.error);
                std::process::exit(1);
            }
            if self.started.elapsed()
                > Duration::from_secs(if self.check_detection { 600 } else { 15 })
            {
                eprintln!("Screenshot smoke test timed out");
                std::process::exit(1);
            }
            let screenshot = ctx.input(|input| {
                input.events.iter().find_map(|event| match event {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(image) = screenshot {
                let bytes: Vec<_> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                image::save_buffer(
                    path,
                    &bytes,
                    image.size[0] as u32,
                    image.size[1] as u32,
                    image::ColorType::Rgba8,
                )
                .expect("Save screenshot");
                if self.check_detection {
                    std::fs::write(path.with_extension("json"), json!({
                        "batch_progress": self.batch_progress, "single_progress": self.single_progress,
                        "status": self.status, "error": self.error,
                        "elapsed_seconds": self.started.elapsed().as_secs_f64()
                    }).to_string()).expect("Save detection check");
                }
                if self.check_folder {
                    std::fs::write(
                        path.with_extension("json"),
                        serde_json::json!({
                            "picker_open": self.picker.is_some(), "updates": self.folder_updates,
                            "displayed_frame": self.displayed_frame
                        })
                        .to_string(),
                    )
                    .expect("Save folder responsiveness check");
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            if !self.screenshot_requested
                && (!self.check_detection
                    || (self.detection_check_started
                        && self.batch_progress == 1.0
                        && self.detection.is_none()))
                && self.started.elapsed() > Duration::from_secs(2)
                && (self.preview.is_none() || self.texture.is_some())
                && (!self.check_playback
                    || (!self.playing
                        && !self.pending
                        && self.displayed_frame == Some(self.count.saturating_sub(1))))
                && (!self.check_folder
                    || (self.picker.is_some()
                        && self.folder_updates >= 10
                        && self.displayed_frame.is_some_and(|frame| frame >= 5)))
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                self.screenshot_requested = true;
            }
            ctx.request_repaint_after(Duration::from_millis(50));
        }
        let palette = Palette::new(self.dark);
        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::NONE
                    .fill(palette.surface)
                    .inner_margin(egui::Margin {
                        left: 28,
                        right: 28,
                        top: 18,
                        bottom: 18,
                    })
                    .stroke(egui::Stroke::new(1.0_f32, palette.border)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    crossing_mark(ui, 42.0, palette.accent);
                    ui.add_space(4.0);
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new("路權辨識").size(23.0).strong());
                        ui.label(
                            egui::RichText::new("交通違規影片分析")
                                .size(12.0)
                                .color(palette.muted),
                        );
                    });
                    ui.add_space(if ui.available_width() > 650.0 {
                        48.0
                    } else {
                        12.0
                    });
                    for (value, title) in [(false, "影片分析"), (true, "旋轉校正")] {
                        if ui
                            .add(
                                egui::Button::selectable(self.calibration == value, title)
                                    .min_size(egui::vec2(104.0, 38.0))
                                    .corner_radius(8),
                            )
                            .clicked()
                        {
                            self.calibration = value;
                            self.dirty = true;
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(if self.dark {
                                "淺色外觀"
                            } else {
                                "深色外觀"
                            })
                            .clicked()
                        {
                            self.dark = !self.dark;
                            apply_theme(ctx, self.dark);
                        }
                    });
                });
            });
        egui::SidePanel::right("settings")
            .exact_width(if ctx.content_rect().width() < 980.0 {
                294.0
            } else {
                330.0
            })
            .resizable(false)
            .frame(
                egui::Frame::NONE
                    .fill(palette.background)
                    .inner_margin(egui::Margin {
                        left: 0,
                        right: 24,
                        top: 24,
                        bottom: 24,
                    }),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.settings(ui, palette);
                });
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(palette.background).inner_margin(24))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.workspace(ui, palette);
                });
            });
        if self.playing {
            let target = self
                .play_start_frame
                .saturating_add((self.last_frame.elapsed().as_secs_f64() * self.fps) as u64);
            self.frame = target.min(self.count.saturating_sub(1));
            self.playing = target < self.count;
            self.dirty = self.displayed_frame != Some(self.frame);
        }
        if self.dirty
            && self.requested_frame != Some(self.frame)
            && (!self.scrubbing || self.last_seek_send.elapsed() >= Duration::from_millis(50))
            && let Some(preview) = &self.preview
        {
            let request = if self.playing {
                preview.request_playback(self.frame)
            } else {
                preview.request(self.frame)
            };
            match request {
                Ok(()) => {
                    self.requested_frame = Some(self.frame);
                    self.last_seek_send = Instant::now();
                    self.pending = true;
                    self.dirty = false;
                }
                Err(error) => {
                    self.error = Some(error);
                    self.preview = None;
                    self.playing = false;
                }
            }
        }
        if self.requested_frame == Some(self.frame) || self.preview.is_none() {
            self.dirty = false;
        }
        if self.pending
            || self.playing
            || self.dirty
            || self.detection.is_some()
            || self.picker.is_some()
        {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }
}

fn main() -> eframe::Result {
    let capturing = std::env::var_os("TRAFFIC_SCREENSHOT").is_some();
    let compact =
        capturing && std::env::var("TRAFFIC_SCREENSHOT_LAYOUT").as_deref() == Ok("compact");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(if compact {
                [860.0, 640.0]
            } else {
                [1280.0, 880.0]
            })
            .with_min_inner_size([860.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "交通違規偵測系統",
        options,
        Box::new(|cc| {
            if let Some(gl) = &cc.gl {
                use eframe::glow::HasContext;
                // SAFETY: eframe's OpenGL context is current during app creation.
                unsafe {
                    eprintln!(
                        "UI GPU: {} / {}",
                        gl.get_parameter_string(eframe::glow::VENDOR),
                        gl.get_parameter_string(eframe::glow::RENDERER)
                    );
                }
            }
            let mut fonts = egui::FontDefinitions::default();
            let paths = [
                std::env::var_os("TRAFFIC_FONT").map(PathBuf::from),
                Some(PathBuf::from("C:/Windows/Fonts/msjh.ttc")),
                Some(PathBuf::from(
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                )),
                Some(PathBuf::from("/System/Library/Fonts/PingFang.ttc")),
            ];
            for path in paths.into_iter().flatten() {
                if let Ok(bytes) = std::fs::read(path) {
                    fonts
                        .font_data
                        .insert("chinese".into(), egui::FontData::from_owned(bytes).into());
                    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                        fonts
                            .families
                            .entry(family)
                            .or_default()
                            .insert(0, "chinese".into());
                    }
                    break;
                }
            }
            if let Ok(bytes) = std::fs::read("C:/Windows/Fonts/segoeui.ttf") {
                fonts
                    .font_data
                    .insert("body".into(), egui::FontData::from_owned(bytes).into());
                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .insert(0, "body".into());
            }
            cc.egui_ctx.set_fonts(fonts);
            apply_theme(&cc.egui_ctx, false);
            cc.egui_ctx.all_styles_mut(|style| {
                style.spacing.item_spacing = egui::vec2(10.0, 8.0);
                style.spacing.button_padding = egui::vec2(14.0, 10.0);
                style.spacing.interact_size.y = 36.0;
                style.spacing.slider_rail_height = 4.0;
                for (kind, size) in [
                    (egui::TextStyle::Heading, 22.0),
                    (egui::TextStyle::Body, 14.0),
                    (egui::TextStyle::Button, 14.0),
                    (egui::TextStyle::Small, 12.0),
                    (egui::TextStyle::Monospace, 13.0),
                ] {
                    let font = if kind == egui::TextStyle::Monospace {
                        egui::FontId::monospace(size)
                    } else {
                        egui::FontId::proportional(size)
                    };
                    style.text_styles.insert(kind, font);
                }
            });
            let mut desktop = Desktop::default();
            if capturing {
                desktop.check_detection =
                    std::env::var("TRAFFIC_SCREENSHOT_VIEW").as_deref() == Ok("detect");
                if desktop.check_detection {
                    desktop.task = 1;
                    desktop.output = std::env::var_os("TRAFFIC_CHECK_OUTPUT").map(PathBuf::from);
                }
                desktop.check_playback =
                    std::env::var("TRAFFIC_SCREENSHOT_VIEW").as_deref() == Ok("playback");
                desktop.autoplay_check = desktop.check_playback;
                desktop.check_folder =
                    std::env::var("TRAFFIC_SCREENSHOT_VIEW").as_deref() == Ok("folder");
                desktop.autoplay_check |= desktop.check_folder;
                desktop.dark = std::env::var("TRAFFIC_SCREENSHOT_THEME").as_deref() == Ok("dark");
                desktop.calibration =
                    std::env::var("TRAFFIC_SCREENSHOT_VIEW").as_deref() == Ok("calibration");
                if desktop.calibration {
                    desktop.angle = 15.0;
                }
                if std::env::var("TRAFFIC_SCREENSHOT_READY").as_deref() == Ok("1") {
                    desktop.output = Some(project_root().join("runtime"));
                }
                apply_theme(&cc.egui_ctx, desktop.dark);
            }
            if let Some(path) = std::env::var_os("TRAFFIC_PREVIEW_VIDEO") {
                let path = PathBuf::from(path);
                desktop.videos = vec![path.clone()];
                desktop.open_preview(path);
            }
            Ok(Box::new(desktop))
        }),
    )
}

#[cfg(test)]
mod interaction_tests {
    use super::*;
    #[test]
    fn open_picker_does_not_block_the_ui() {
        let started = Instant::now();
        let result = start_picker(egui::Context::default(), || {
            std::thread::sleep(Duration::from_millis(150));
            Some(vec![PathBuf::from("output")])
        })
        .unwrap();
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "UI blocked for {:?} while the chooser was open",
            started.elapsed()
        );
        assert!(
            result.try_recv().is_err(),
            "Chooser should still be open while UI continues"
        );
        assert_eq!(
            result.recv_timeout(Duration::from_secs(1)).unwrap(),
            Some(vec![PathBuf::from("output")])
        );
    }
}
