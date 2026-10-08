//! Native Windows video decoding. COM and decoder objects stay on one worker thread.
use eframe::egui;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
};

pub struct Frame {
    pub image: egui::ColorImage,
    pub requested: u64,
    pub count: u64,
    pub fps: f64,
}

pub struct Preview {
    commands: SyncSender<()>,
    latest: Arc<Mutex<Request>>,
    pub frames: Receiver<Result<Option<Frame>, String>>,
}

#[derive(Clone, Copy)]
struct Request {
    frame: u64,
    seeking: bool,
}

fn timing_rate(timestamps: &[i64], declared: u64) -> u64 {
    let mut timestamps = timestamps.to_vec();
    timestamps.sort_unstable();
    timestamps.dedup();
    let mut intervals: Vec<_> = timestamps
        .windows(2)
        .filter_map(|pair| pair[1].checked_sub(pair[0]))
        .filter(|delta| *delta > 0 && *delta <= u32::MAX as i64)
        .collect();
    intervals.sort_unstable();
    let Some(delta) = intervals.get(intervals.len() / 2) else {
        return declared;
    };
    let fps = (declared >> 32) as f64 / declared as u32 as f64;
    let observed = 10_000_000.0 / *delta as f64;
    if (observed / fps - 1.0).abs() > 0.01 {
        (10_000_000_u64 << 32) | *delta as u64
    } else {
        declared
    }
}

impl Preview {
    pub fn open(path: PathBuf) -> Result<Self, String> {
        if !path.is_file() {
            return Err(format!("找不到影片：{}", path.display()));
        }
        let (commands, requests) = mpsc::sync_channel(1);
        let (output, frames) = mpsc::sync_channel(1);
        let latest = Arc::new(Mutex::new(Request {
            frame: 0,
            seeking: true,
        }));
        let wanted = latest.clone();
        std::thread::Builder::new()
            .name("native-video".into())
            .spawn(move || {
                let mut reader = match native::Reader::open(&path) {
                    Ok(reader) => reader,
                    Err(error) => {
                        let _ = output.send(Err(error));
                        return;
                    }
                };
                for () in &requests {
                    loop {
                        for () in requests.try_iter() {}
                        let request = *wanted.lock().unwrap();
                        let frame = reader.read(request.frame, || {
                            let current = *wanted.lock().unwrap();
                            current.seeking && current.frame != request.frame
                        });
                        let current = *wanted.lock().unwrap();
                        if current.seeking && current.frame != request.frame {
                            continue;
                        }
                        let failed = frame.is_err();
                        if output.send(frame).is_err() || failed {
                            return;
                        }
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            commands,
            latest,
            frames,
        })
    }

    pub fn request(&self, frame: u64) -> Result<(), String> {
        self.queue(frame, true)
    }

    pub fn request_playback(&self, frame: u64) -> Result<(), String> {
        self.queue(frame, false)
    }

    fn queue(&self, frame: u64, seeking: bool) -> Result<(), String> {
        *self.latest.lock().unwrap() = Request { frame, seeking };
        match self.commands.try_send(()) {
            Ok(()) | Err(mpsc::TrySendError::Full(())) => Ok(()),
            Err(mpsc::TrySendError::Disconnected(())) => Err("影片預覽連線中斷。".into()),
        }
    }
}

// Validate signed strides and padding before indexing decoder memory.
fn rgb_image(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: i32,
    first_row: usize,
) -> Result<egui::ColorImage, String> {
    let count = width
        .checked_mul(height)
        .filter(|count| *count > 0 && *count <= 16_777_216)
        .ok_or("影片尺寸無效或過大。")?;
    let row_bytes = width.checked_mul(4).ok_or("影格大小溢位。")?;
    if (stride as i64).unsigned_abs() < row_bytes as u64 {
        return Err("影片行距無效。".into());
    }
    let mut pixels = Vec::with_capacity(count);
    for y in 0..height {
        let offset = first_row as i128 + y as i128 * stride as i128;
        let offset = usize::try_from(offset).map_err(|_| "影片影格超出緩衝區。")?;
        let end = offset.checked_add(row_bytes).ok_or("影片影格溢位。")?;
        let row = bytes.get(offset..end).ok_or("影片影格緩衝區不足。")?;
        pixels.extend(
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|p| egui::Color32::from_rgb(p[2], p[1], p[0])),
        );
    }
    Ok(egui::ColorImage::new([width, height], pixels))
}

#[cfg(windows)]
mod native {
    use super::*;
    use std::{os::windows::ffi::OsStrExt, path::Path};
    use windows::{
        Win32::{
            Graphics::{
                Direct3D::D3D_DRIVER_TYPE_UNKNOWN, Direct3D10::ID3D10Multithread, Direct3D11::*,
                Dxgi::*,
            },
            Media::MediaFoundation::*,
            System::Com::{
                COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize,
                StructuredStorage::{PROPVARIANT, PropVariantToUInt64},
            },
        },
        core::{GUID, HRESULT, Interface, PCWSTR},
    };

    const VIDEO: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;

    struct Hardware {
        manager: IMFDXGIDeviceManager,
        name: String,
    }

    impl Hardware {
        fn create() -> windows::core::Result<Self> {
            // SAFETY: COM is initialized; all D3D objects stay on the decoder thread.
            unsafe {
                let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
                let mut adapters = Vec::new();
                let mut index = 0;
                loop {
                    let adapter = match factory.EnumAdapters1(index) {
                        Ok(adapter) => adapter,
                        Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                        Err(error) => return Err(error),
                    };
                    let description = adapter.GetDesc1()?;
                    if description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                        adapters.push((description.VendorId != 0x10de, adapter, description));
                    }
                    index += 1;
                }
                adapters.sort_by_key(|item| item.0);
                let mut last_error = invalid("找不到支援影片解碼的顯卡。");
                for (_, adapter, description) in adapters {
                    let mut device = None;
                    if let Err(error) = D3D11CreateDevice(
                        &adapter,
                        D3D_DRIVER_TYPE_UNKNOWN,
                        Default::default(),
                        D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                        None,
                        D3D11_SDK_VERSION,
                        Some(&mut device),
                        None,
                        None,
                    ) {
                        last_error = error;
                        continue;
                    }
                    let device = device.ok_or_else(|| invalid("無法建立 D3D11 裝置。"))?;
                    let _ = device
                        .cast::<ID3D10Multithread>()?
                        .SetMultithreadProtected(true);
                    let (mut token, mut manager) = (0, None);
                    MFCreateDXGIDeviceManager(&mut token, &mut manager)?;
                    let manager = manager.ok_or_else(|| invalid("無法建立 DXGI 裝置管理員。"))?;
                    manager.ResetDevice(&device, token)?;
                    let end = description
                        .Description
                        .iter()
                        .position(|value| *value == 0)
                        .unwrap_or(description.Description.len());
                    return Ok(Self {
                        manager,
                        name: String::from_utf16_lossy(&description.Description[..end]),
                    });
                }
                Err(last_error)
            }
        }
    }

    fn read_sample(source: &IMFSourceReader) -> windows::core::Result<Option<(IMFSample, i64)>> {
        // SAFETY: caller owns a source reader initialized on this decoder thread.
        unsafe {
            loop {
                let (mut flags, mut timestamp, mut sample) = (0, 0, None);
                source.ReadSample(
                    VIDEO,
                    0,
                    None,
                    Some(&mut flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )?;
                if let Some(sample) = sample {
                    return Ok(Some((sample, timestamp)));
                }
                if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                    return Ok(None);
                }
            }
        }
    }

    fn invalid(message: &str) -> windows::core::Error {
        windows::core::Error::new(HRESULT(0x80070057_u32 as i32), message)
    }

    struct Runtime;
    impl Runtime {
        fn start() -> windows::core::Result<Self> {
            // SAFETY: initialized and uninitialized on this same decoder thread.
            unsafe {
                CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
                if let Err(error) = MFStartup(MF_VERSION, 0) {
                    CoUninitialize();
                    return Err(error);
                }
            }
            Ok(Self)
        }
    }
    impl Drop for Runtime {
        fn drop(&mut self) {
            unsafe {
                let _ = MFShutdown();
                CoUninitialize();
            }
        }
    }

    pub struct Reader {
        source: IMFSourceReader,
        fps: f64,
        count: u64,
        last: Option<i64>,
        origin: i64,
        #[cfg(test)]
        pub(super) gpu_output: bool,
        _hardware: Option<Hardware>,
        // Fields drop in declaration order: source must release before MFShutdown.
        _runtime: Runtime,
    }

    impl Reader {
        pub fn open(path: &Path) -> Result<Self, String> {
            let mode =
                std::env::var("TRAFFIC_VIDEO_ACCELERATION").unwrap_or_else(|_| "auto".into());
            match mode.as_str() {
                "software" => Self::open_mode(path, false),
                "hardware" => Self::open_mode(path, true),
                "auto" => Self::open_mode(path, true).or_else(|error| {
                    eprintln!("Preview hardware unavailable: {error}; falling back to software");
                    Self::open_mode(path, false)
                }),
                _ => Err("TRAFFIC_VIDEO_ACCELERATION 必須為 auto、hardware 或 software。".into()),
            }
        }

        pub(super) fn open_mode(path: &Path, accelerated: bool) -> Result<Self, String> {
            let open = || -> windows::core::Result<Self> {
                let runtime = Runtime::start()?;
                let hardware = if accelerated {
                    Some(Hardware::create()?)
                } else {
                    None
                };
                let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
                // SAFETY: all pointers refer to owned, live values; reader stays on this thread.
                unsafe {
                    let mut attributes = None;
                    MFCreateAttributes(&mut attributes, 4)?;
                    let attributes = attributes.ok_or_else(|| invalid("無法建立影片設定。"))?;
                    attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 1)?;
                    if let Some(hardware) = &hardware {
                        attributes.SetUnknown(&MF_SOURCE_READER_D3D_MANAGER, &hardware.manager)?;
                        attributes.SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1)?;
                    } else {
                        attributes.SetUINT32(&MF_SOURCE_READER_DISABLE_DXVA, 1)?;
                    }
                    let source = MFCreateSourceReaderFromURL(PCWSTR(path.as_ptr()), &attributes)?;
                    source.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)?;
                    source.SetStreamSelection(VIDEO, true)?;
                    let original = source.GetCurrentMediaType(VIDEO)?;
                    if hardware.is_some() {
                        let native_output = MFCreateMediaType()?;
                        native_output.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
                        native_output.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
                        source
                            .SetCurrentMediaType(VIDEO, None, &native_output)
                            .map_err(|e| {
                                windows::core::Error::new(e.code(), format!("Set NV12 output: {e}"))
                            })?;
                    }
                    let rate = original.GetUINT64(&MF_MT_FRAME_RATE)?;
                    let numerator = (rate >> 32) as u32;
                    let denominator = rate as u32;
                    if numerator == 0 || denominator == 0 {
                        return Err(invalid("影片缺少有效的幀率。"));
                    }
                    // Use source timestamps before conversion: Windows can report the wrong MKV FPS.
                    // ponytail: VFR preview uses a nominal output rate; use a PTS-driven timeline if exact frame counts are needed.
                    let mut timestamps = Vec::new();
                    // Compressed B-frames arrive in decode order; sample a group and sort by presentation time.
                    for _ in 0..8 {
                        if let Some((_, timestamp)) = read_sample(&source).map_err(|e| {
                            windows::core::Error::new(e.code(), format!("Probe timestamps: {e}"))
                        })? {
                            timestamps.push(timestamp);
                        } else {
                            break;
                        }
                    }
                    if timestamps.is_empty() {
                        return Err(invalid("影片沒有可解碼的影格。"));
                    }
                    let output_rate = timing_rate(&timestamps, rate);
                    let fps = (output_rate >> 32) as f64 / output_rate as u32 as f64;
                    // MP4 edit lists / B-frame delay can make the first PTS nonzero.
                    let origin = *timestamps.iter().min().unwrap();
                    source.SetCurrentPosition(&GUID::zeroed(), &PROPVARIANT::from(0_i64))?;
                    let duration = source.GetPresentationAttribute(
                        MF_SOURCE_READER_MEDIASOURCE.0 as u32,
                        &MF_PD_DURATION,
                    )?;
                    let duration = PropVariantToUInt64(&duration)?;
                    if duration == 0 {
                        return Err(invalid("影片缺少有效的長度。"));
                    }
                    let size = original.GetUINT64(&MF_MT_FRAME_SIZE)?;
                    let (width, height) = ((size >> 32) as u32, size as u32);
                    if width == 0 || height == 0 {
                        return Err(invalid("影片尺寸無效。"));
                    }
                    let scale = (960.0 / width.max(height) as f64).min(1.0);
                    let width = ((width as f64 * scale).round() as u32).max(1);
                    let height = ((height as f64 * scale).round() as u32).max(1);
                    let output = MFCreateMediaType()?;
                    output.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
                    output.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
                    if output_rate != rate {
                        output.SetUINT64(&MF_MT_FRAME_RATE, output_rate)?;
                    }
                    output.SetUINT64(&MF_MT_FRAME_SIZE, ((width as u64) << 32) | height as u64)?;
                    source
                        .SetCurrentMediaType(VIDEO, None, &output)
                        .map_err(|e| {
                            windows::core::Error::new(e.code(), format!("Set RGB output: {e}"))
                        })?;
                    // Decode a real sample before accepting this pipeline: unsupported formats
                    // must fall back during open, rather than leaving the preview unusable.
                    let (sample, _) = read_sample(&source)
                        .map_err(|e| {
                            windows::core::Error::new(e.code(), format!("Read RGB output: {e}"))
                        })?
                        .ok_or_else(|| invalid("影片沒有解碼輸出。"))?;
                    let buffer = sample.GetBufferByIndex(0)?;
                    let gpu_output = buffer.cast::<IMFDXGIBuffer>().is_ok();
                    eprintln!(
                        "Preview device: {}; GPU output: {}",
                        hardware
                            .as_ref()
                            .map_or("software", |hardware| hardware.name.as_str()),
                        gpu_output
                    );
                    source.SetCurrentPosition(&GUID::zeroed(), &PROPVARIANT::from(0_i64))?;
                    let count = (duration as f64 * fps / 10_000_000.0).round().max(1.0) as u64;
                    Ok(Self {
                        source,
                        fps,
                        count,
                        last: None,
                        origin,
                        #[cfg(test)]
                        gpu_output,
                        _hardware: hardware,
                        _runtime: runtime,
                    })
                }
            };
            open().map_err(|e| format!("Windows 無法解碼影片：{e}。請確認檔案與影片編碼。"))
        }

        pub fn read(
            &mut self,
            requested: u64,
            superseded: impl Fn() -> bool,
        ) -> Result<Option<Frame>, String> {
            self.read_inner(requested, superseded)
                .map_err(|e| format!("無法讀取影片影格：{e}"))
        }

        fn read_inner(
            &mut self,
            requested: u64,
            superseded: impl Fn() -> bool,
        ) -> windows::core::Result<Option<Frame>> {
            if requested >= self.count {
                return Ok(None);
            }
            let position = (requested as f64 / self.fps * 10_000_000.0).round() as i64;
            let target = self
                .origin
                .checked_add(position)
                .ok_or_else(|| invalid("影片時間溢位。"))?;
            // SAFETY: source, samples and locked buffers are all local to this thread.
            unsafe {
                // ponytail: decode nearby forward frames sequentially; seek for backward or >1s jumps.
                if superseded() {
                    return Ok(None);
                }
                if self
                    .last
                    .is_some_and(|last| target <= last || target.saturating_sub(last) > 10_000_000)
                {
                    self.source
                        .SetCurrentPosition(&GUID::zeroed(), &PROPVARIANT::from(position))?;
                    self.last = None;
                }
                loop {
                    if superseded() {
                        return Ok(None);
                    }
                    let Some((sample, timestamp)) = read_sample(&self.source)? else {
                        return Ok(None);
                    };
                    self.last = Some(timestamp);
                    if superseded() {
                        return Ok(None);
                    }
                    if timestamp.saturating_add((5_000_000.0 / self.fps) as i64) < target {
                        continue;
                    }
                    let media_type = self.source.GetCurrentMediaType(VIDEO)?;
                    let size = media_type.GetUINT64(&MF_MT_FRAME_SIZE)?;
                    let (width, height) = ((size >> 32) as usize, size as u32 as usize);
                    // ponytail: GPU frames are read back for egui's pixel upload; use shared
                    // D3D/render textures if this copy becomes the preview bottleneck.
                    let buffer = sample.ConvertToContiguousBuffer()?;
                    let image = if let Ok(buffer2d) = buffer.cast::<IMF2DBuffer2>() {
                        let (mut first, mut start) = (std::ptr::null_mut(), std::ptr::null_mut());
                        let (mut stride, mut length) = (0, 0);
                        buffer2d.Lock2DSize(
                            MF2DBuffer_LockFlags_Read,
                            &mut first,
                            &mut stride,
                            &mut start,
                            &mut length,
                        )?;
                        let result = if start.is_null() || length == 0 {
                            Err("影片緩衝區為空。".into())
                        } else if let Some(offset) = (first as usize).checked_sub(start as usize) {
                            // SAFETY: Lock2DSize owns this memory until Unlock2D; indexing checks every row.
                            rgb_image(
                                std::slice::from_raw_parts(start, length as usize),
                                width,
                                height,
                                stride,
                                offset,
                            )
                        } else {
                            Err("影片影格位置無效。".into())
                        };
                        buffer2d.Unlock2D()?;
                        result.map_err(|e| invalid(&e))?
                    } else {
                        let mut start = std::ptr::null_mut();
                        let mut length = 0;
                        buffer.Lock(&mut start, None, Some(&mut length))?;
                        let result = (|| -> Result<egui::ColorImage, String> {
                            if start.is_null() || length == 0 {
                                return Err("影片緩衝區為空。".into());
                            }
                            let stride = media_type
                                .GetUINT32(&MF_MT_DEFAULT_STRIDE)
                                .map(|v| v as i32)
                                .or_else(|_| {
                                    MFGetStrideForBitmapInfoHeader(
                                        MFVideoFormat_RGB32.data1,
                                        width as u32,
                                    )
                                })
                                .map_err(|e| e.to_string())?;
                            let first = if stride < 0 {
                                (height.saturating_sub(1))
                                    .checked_mul(stride.unsigned_abs() as usize)
                                    .ok_or("影片行距溢位。")?
                            } else {
                                0
                            };
                            // SAFETY: Lock returns length valid bytes until Unlock, and rgb_image uses checked slices.
                            rgb_image(
                                std::slice::from_raw_parts(start, length as usize),
                                width,
                                height,
                                stride,
                                first,
                            )
                        })();
                        buffer.Unlock()?;
                        result.map_err(|e| invalid(&e))?
                    };
                    return Ok(Some(Frame {
                        image,
                        requested,
                        count: self.count,
                        fps: self.fps,
                    }));
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod native {
    use super::*;
    pub struct Reader;
    impl Reader {
        pub fn open(_: &std::path::Path) -> Result<Self, String> {
            Err("原生影片預覽目前支援 Windows。".into())
        }
        pub fn read(&mut self, _: u64, _: impl Fn() -> bool) -> Result<Option<Frame>, String> {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(windows)]
    #[test]
    #[ignore = "Set TRAFFIC_TEST_VIDEO to the check-ui fixture"]
    fn rapid_scrubbing_keeps_the_latest_request() {
        let path = std::env::var("TRAFFIC_TEST_VIDEO").unwrap();
        let preview = Preview::open(path.into()).unwrap();
        preview.request(0).unwrap();
        preview
            .frames
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let started = std::time::Instant::now();
        for requested in [19, 0, 18, 1, 17] {
            preview
                .request(requested)
                .expect("UI must be able to replace an in-flight seek");
        }
        assert!(started.elapsed() < std::time::Duration::from_millis(50));
        loop {
            let frame = preview
                .frames
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap()
                .unwrap()
                .unwrap();
            if frame.requested == 17 {
                break;
            }
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Set TRAFFIC_SEEK_VIDEO to a long GOP clip"]
    fn measure_seek_latency() {
        let preview = Preview::open(std::env::var("TRAFFIC_SEEK_VIDEO").unwrap().into()).unwrap();
        preview.request(0).unwrap();
        let first = preview
            .frames
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap()
            .unwrap();
        let mut timings = Vec::new();
        for fraction in [0.9, 0.1, 0.8, 0.2, 0.95, 0.05] {
            let frame = ((first.count - 1) as f64 * fraction) as u64;
            let started = std::time::Instant::now();
            preview.request(frame).unwrap();
            let result = preview
                .frames
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(result.requested, frame);
            timings.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        eprintln!(
            "SEEK_PROFILE {}",
            serde_json::json!({"size": first.image.size, "fps": first.fps,
            "frames": first.count, "seek_ms": timings})
        );
        for serialized in [true, false] {
            let mut latest_ms = Vec::new();
            for _ in 0..3 {
                preview.request(0).unwrap();
                preview
                    .frames
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .unwrap()
                    .unwrap();
                let far = (first.count as f64 * 0.9) as u64;
                let latest = (first.count as f64 * 0.05) as u64;
                preview.request(far).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(20));
                let started = std::time::Instant::now();
                if serialized {
                    preview
                        .frames
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap()
                        .unwrap();
                }
                preview.request(latest).unwrap();
                loop {
                    let frame = preview
                        .frames
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap()
                        .unwrap()
                        .unwrap();
                    if frame.requested == latest {
                        break;
                    }
                }
                latest_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            eprintln!(
                "LATEST_PROFILE {}",
                serde_json::json!({"serialized":serialized,"latest_ms":latest_ms})
            );
        }
    }
    #[test]
    fn reordered_timestamps_and_incorrect_metadata() {
        let ten_fps = (10_u64 << 32) | 1;
        assert_eq!(
            timing_rate(&[0, 4_000_000, 2_000_000, 1_000_000, 3_000_000], ten_fps),
            ten_fps
        );
        let corrected = timing_rate(&[0, 1_000_000, 2_000_000], (5_u64 << 32) | 1);
        assert_eq!((corrected >> 32) / corrected as u32 as u64, 10);
        assert_eq!(timing_rate(&[0, 0], ten_fps), ten_fps);
        assert_eq!(timing_rate(&[i64::MIN, i64::MAX], ten_fps), ten_fps);
    }
    #[test]
    fn strides_and_truncated_buffers() {
        let bytes = [
            0, 0, 255, 0, 0, 255, 0, 0, 9, 9, 9, 9, 255, 0, 0, 0, 255, 255, 255, 0, 9, 9, 9, 9,
        ];
        let top_down = rgb_image(&bytes, 2, 2, 12, 0).unwrap();
        assert_eq!(
            top_down.pixels,
            [
                egui::Color32::RED,
                egui::Color32::GREEN,
                egui::Color32::BLUE,
                egui::Color32::WHITE
            ]
        );
        let bottom_up = rgb_image(&bytes, 2, 2, -12, 12).unwrap();
        assert_eq!(bottom_up.pixels[0], egui::Color32::BLUE);
        assert!(rgb_image(&bytes[..15], 2, 2, 12, 0).is_err());
        assert!(rgb_image(&bytes, 2, 2, 4, 0).is_err());
        assert!(rgb_image(&bytes, 2, 2, -12, 0).is_err());
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Set TRAFFIC_TEST_VIDEO to a clip for the software/hardware decode benchmark"]
    fn decode_throughput() {
        let path = PathBuf::from(std::env::var("TRAFFIC_TEST_VIDEO").unwrap());
        let hardware = std::env::var("TRAFFIC_VIDEO_ACCELERATION").as_deref() == Ok("hardware");
        for _ in 0..3 {
            let mut reader = native::Reader::open_mode(&path, hardware).unwrap();
            if hardware {
                assert!(reader.gpu_output);
            }
            let started = std::time::Instant::now();
            let mut decoded = 0;
            while let Some(frame) = reader.read(decoded, || false).unwrap() {
                assert!(!frame.image.pixels.is_empty());
                decoded += 1;
            }
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "DECODE_PROFILE {}",
                serde_json::json!({"hardware":hardware,
                "frames":decoded,"seconds":elapsed,"fps":decoded as f64 / elapsed})
            );
            assert!(decoded > 0);
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Set TRAFFIC_TEST_VIDEO to an H264 clip; requires a D3D11 video device"]
    fn hardware_matches_software() {
        let path = PathBuf::from(std::env::var("TRAFFIC_TEST_VIDEO").unwrap());
        let mut software = native::Reader::open_mode(&path, false).unwrap();
        let mut hardware = native::Reader::open_mode(&path, true).unwrap();
        assert!(
            hardware.gpu_output,
            "Hardware pipeline must produce D3D11 buffers"
        );
        for frame in [0, 9, 1, 17] {
            let expected = software.read(frame, || false).unwrap().unwrap();
            let actual = hardware.read(frame, || false).unwrap().unwrap();
            assert_eq!(actual.image.size, expected.image.size);
            assert_eq!(actual.count, expected.count);
            assert_eq!(actual.fps, expected.fps);
            let difference: u64 = actual
                .image
                .pixels
                .iter()
                .zip(&expected.image.pixels)
                .map(|(a, b)| {
                    a.to_array()[..3]
                        .iter()
                        .zip(&b.to_array()[..3])
                        .map(|(a, b)| a.abs_diff(*b) as u64)
                        .sum::<u64>()
                })
                .sum();
            let mean = difference as f64 / (actual.image.pixels.len() * 3) as f64;
            eprintln!("HW_PIXEL_PROFILE frame={frame} mean_error={mean:.4}");
            assert!(mean < 3.0, "Hardware frame differs from software: {mean}");
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Set TRAFFIC_TEST_VIDEO to the check-ui.ps1 preview fixture"]
    fn native_decode_seek_and_end() {
        let path = std::env::var("TRAFFIC_TEST_VIDEO").expect("TRAFFIC_TEST_VIDEO is required");
        let width: usize = std::env::var("TRAFFIC_TEST_WIDTH")
            .unwrap_or("640".into())
            .parse()
            .unwrap();
        let preview = Preview::open(path.into()).unwrap();
        for requested in [0, 1, 9, 0, 19] {
            preview.request(requested).unwrap();
            let frame = preview
                .frames
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(frame.requested, requested.min(frame.count - 1));
            assert_eq!(frame.image.size, [width, width * 9 / 16]);
            assert!((frame.fps - 10.0).abs() < 0.01);
            assert_eq!(frame.count, 20);
            assert!(
                (frame.image.pixels[2 * width + 2].r() as i32 - requested as i32 * 10).abs() <= 15,
                "requested={requested}, actual={}",
                frame.image.pixels[2 * width + 2].r()
            );
        }
        preview.request(20).unwrap();
        assert!(
            preview
                .frames
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap()
                .unwrap()
                .is_none()
        );
        assert!(Preview::open(PathBuf::from("missing-video.avi")).is_err());
        let corrupt =
            std::env::temp_dir().join(format!("traffic-corrupt-{}.avi", std::process::id()));
        std::fs::write(&corrupt, b"not a video").unwrap();
        let preview = Preview::open(corrupt.clone()).unwrap();
        assert!(
            preview
                .frames
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap()
                .is_err()
        );
        drop(preview);
        std::fs::remove_file(corrupt).unwrap();
    }
}
