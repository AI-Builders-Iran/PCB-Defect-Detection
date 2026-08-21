"""
PCB Defect Detection — Streamlit Dashboard
============================================

A simple, clean dashboard for the PCB defect detection pipeline. Two modes:

    * Single Image — upload one PCB photo, see detected defects and download
      the annotated result.
    * Video — upload a production-line clip, process it end to end, and
      download the annotated video + a JSON report.

Design note: this UI intentionally uses no custom CSS. It relies entirely on
Streamlit's native theme (see .streamlit/config.toml) and built-in widgets
(st.metric, st.tabs, st.expander, ...), so it always renders correctly and
never fights with Streamlit's own styling.

Run (from the project root):
    streamlit run app/streamlit_app/app.py
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

import cv2
import streamlit as st

# ---------------------------------------------------------------------------
# Make sure the project root is importable regardless of the working
# directory the app is launched from. When Streamlit runs a script, Python
# only adds the script's own folder (app/streamlit_app) to sys.path, not the
# project root two levels up — so "from src.pipeline import ..." fails with
# ModuleNotFoundError unless we add the root ourselves.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import PCBPipeline, PipelineConfig

try:
    import imageio_ffmpeg
    _FFMPEG_AVAILABLE = True
except ImportError:
    _FFMPEG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# i18n — minimal, text-only (no direction/font hacks; Streamlit's native
# layout handles both languages fine without extra CSS)
# ---------------------------------------------------------------------------
I18N = {
    "fa": {
        "title": "تشخیص عیوب PCB",
        "subtitle": "بازرسی خودکار عیوب مونتاژ و اتصالات روی برد با هوش مصنوعی",
        "tab_image": "🖼️ تصویر",
        "tab_video": "🎬 ویدیو",
        "sidebar_lang": "زبان",
        "sidebar_models": "تنظیمات مدل",
        "model1_label": "مسیر مدل قطعه‌بندی برد",
        "model2_label": "مسیر مدل تشخیص عیب",
        "conf_label": "آستانه اطمینان",
        "iou_label": "آستانه IoU",
        "imgsz_label": "وضوح ورودی",
        "tracking_label": "ردیابی عیوب بین فریم‌ها",
        "models_ready": "مدل‌ها آماده‌اند",
        "models_missing": "فایل مدل‌ها پیدا نشد — مسیرها را در نوار کناری بررسی کنید",
        "ffmpeg_ready": "انکودر ویدیو (ffmpeg) فعال است",
        "ffmpeg_missing": "ffmpeg موجود نیست؛ ویدیوی خروجی ممکن است در مرورگر پخش نشود",
        "upload_img": "یک تصویر از برد PCB بارگذاری کنید",
        "upload_vid": "یک ویدیوی خط تولید بارگذاری کنید",
        "run_img": "شروع تحلیل",
        "run_vid": "شروع پردازش ویدیو",
        "original": "تصویر ورودی",
        "annotated": "نتیجه تشخیص",
        "kpi_pcb": "برد شناسایی شد",
        "kpi_defects": "تعداد عیوب",
        "kpi_latency": "زمان پردازش",
        "yes": "بله",
        "no": "خیر",
        "defect_table": "لیست عیوب شناسایی‌شده",
        "col_class": "نوع عیب",
        "col_conf": "اطمینان",
        "col_bbox": "محدوده (BBox)",
        "download_img": "دانلود تصویر نتیجه",
        "download_json": "دانلود گزارش JSON",
        "download_video": "دانلود ویدیوی نتیجه",
        "max_frames": "حداکثر تعداد فریم (-۱ یعنی کل ویدیو)",
        "total_frames": "کل فریم‌ها",
        "pcb_frames": "فریم‌های دارای برد",
        "fps": "نرخ فریم پردازش",
        "error": "خطا در پردازش",
        "spinner_img": "در حال تحلیل تصویر...",
        "spinner_vid": "در حال آماده‌سازی ویدیو برای پخش...",
        "no_models_warning": "قبل از شروع، مسیر فایل‌های مدل را در نوار کناری تنظیم کنید.",
        "codec_note": "اگر پخش‌کننده بالا کار نکرد (به دلیل کدک مرورگر)، فایل را دانلود کنید.",
    },
    "en": {
        "title": "PCB Defect Detection",
        "subtitle": "Automated assembly & solder defect inspection powered by AI",
        "tab_image": "🖼️ Image",
        "tab_video": "🎬 Video",
        "sidebar_lang": "Language",
        "sidebar_models": "Model Settings",
        "model1_label": "Board segmentation model path",
        "model2_label": "Defect detection model path",
        "conf_label": "Confidence threshold",
        "iou_label": "IoU threshold",
        "imgsz_label": "Input resolution",
        "tracking_label": "Track defects across frames",
        "models_ready": "Models ready",
        "models_missing": "Model files not found — check the paths in the sidebar",
        "ffmpeg_ready": "Video encoder (ffmpeg) available",
        "ffmpeg_missing": "ffmpeg not available; output video may not play in-browser",
        "upload_img": "Upload a PCB image",
        "upload_vid": "Upload a production-line video",
        "run_img": "Run Analysis",
        "run_vid": "Run Video Processing",
        "original": "Original",
        "annotated": "Detection Result",
        "kpi_pcb": "Board Detected",
        "kpi_defects": "Defects Found",
        "kpi_latency": "Processing Time",
        "yes": "Yes",
        "no": "No",
        "defect_table": "Detected Defects",
        "col_class": "Defect Type",
        "col_conf": "Confidence",
        "col_bbox": "Bounding Box",
        "download_img": "Download Result Image",
        "download_json": "Download JSON Report",
        "download_video": "Download Result Video",
        "max_frames": "Max frames to process (-1 = entire video)",
        "total_frames": "Total Frames",
        "pcb_frames": "Frames with Board",
        "fps": "Avg. Throughput (FPS)",
        "error": "Processing error",
        "spinner_img": "Analyzing image...",
        "spinner_vid": "Preparing video for playback...",
        "no_models_warning": "Set the model file paths in the sidebar before running.",
        "codec_note": "If the player above doesn't work (browser codec issue), download the file instead.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

txt = I18N[st.session_state.lang]


# ---------------------------------------------------------------------------
# Sidebar — language, model paths, and status
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL1 = str(BASE_DIR / "models" / "best-pcb.onnx")
DEFAULT_MODEL2 = str(BASE_DIR / "models" / "best_detect2.onnx")

with st.sidebar:
    lang_choice = st.selectbox(
        txt["sidebar_lang"],
        options=["en", "fa"],
        format_func=lambda x: "English" if x == "en" else "فارسی",
        index=0 if st.session_state.lang == "en" else 1,
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    st.divider()
    st.subheader(txt["sidebar_models"])

    model1_path = st.text_input(txt["model1_label"], value=DEFAULT_MODEL1)
    model2_path = st.text_input(txt["model2_label"], value=DEFAULT_MODEL2)
    conf_threshold = st.slider(txt["conf_label"], 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider(txt["iou_label"], 0.05, 0.95, 0.45, 0.05)
    imgsz = st.select_slider(txt["imgsz_label"], options=[320, 416, 512, 640], value=416)
    enable_tracking = st.checkbox(txt["tracking_label"], value=True)

    st.divider()
    models_ok = os.path.exists(model1_path) and os.path.exists(model2_path)
    if models_ok:
        st.success(txt["models_ready"], icon="✅")
    else:
        st.warning(txt["models_missing"], icon="⚠️")

    if _FFMPEG_AVAILABLE:
        st.caption(f"✅ {txt['ffmpeg_ready']}")
    else:
        st.caption(f"⚠️ {txt['ffmpeg_missing']}")


# ---------------------------------------------------------------------------
# Cached pipeline loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(m1, m2, conf, iou, sz, tracking):
    config = PipelineConfig(
        model1_path=m1,
        model2_path=m2,
        conf_threshold=conf,
        iou_threshold=iou,
        imgsz=sz,
        enable_tracking=tracking,
        save_video=False,
        save_json=False,
        show_preview=False,
        verbose=False,
    )
    return PCBPipeline(config)


def make_browser_playable(input_path: str) -> str:
    """Re-encodes a video to H.264/yuv420p so it reliably plays in browsers."""
    if not _FFMPEG_AVAILABLE:
        return input_path

    output_path = str(Path(input_path).with_name(Path(input_path).stem + "_h264.mp4"))
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception:
        pass
    return input_path


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"🔍 {txt['title']}")
st.caption(txt["subtitle"])

if not models_ok:
    st.info(txt["no_models_warning"], icon="ℹ️")

st.divider()

tab_image, tab_video = st.tabs([txt["tab_image"], txt["tab_video"]])

# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------
with tab_image:
    uploaded_image = st.file_uploader(
        txt["upload_img"],
        type=["jpg", "jpeg", "png", "bmp"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image.name).suffix) as tmp_in:
            tmp_in.write(uploaded_image.getvalue())
            tmp_in_path = tmp_in.name

        run_clicked = st.button(txt["run_img"], type="primary", key="run_image", disabled=not models_ok)

        if run_clicked:
            try:
                pipeline = load_pipeline(model1_path, model2_path, conf_threshold, iou_threshold, imgsz, enable_tracking)

                with st.spinner(txt["spinner_img"]):
                    result = pipeline.process_image(tmp_in_path)
                    original_frame = cv2.imread(tmp_in_path)
                    annotated = pipeline._draw_results(original_frame, result)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{txt['original']}**")
                    st.image(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.markdown(f"**{txt['annotated']}**")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.write("")
                m1, m2, m3 = st.columns(3)
                m1.metric(txt["kpi_pcb"], txt["yes"] if result.has_pcb else txt["no"])
                m2.metric(txt["kpi_defects"], len(result.defects))
                m3.metric(txt["kpi_latency"], f"{result.processing_time * 1000:.1f} ms")

                if result.defects:
                    st.markdown(f"**{txt['defect_table']}**")
                    st.dataframe(
                        [
                            {
                                txt["col_class"]: d.class_name,
                                txt["col_conf"]: round(d.confidence, 3),
                                txt["col_bbox"]: f"({d.bbox['x1']}, {d.bbox['y1']}) - ({d.bbox['x2']}, {d.bbox['y2']})",
                            }
                            for d in result.defects
                        ],
                        use_container_width=True,
                    )

                out_dir = tempfile.mkdtemp()
                out_img_path = os.path.join(out_dir, "annotated.png")
                cv2.imwrite(out_img_path, annotated)

                result_json = {
                    "metadata": {
                        "source_file": uploaded_image.name,
                        "conf_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                    },
                    "result": result.to_dict(),
                }
                out_json_path = os.path.join(out_dir, "result.json")
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)

                d1, d2 = st.columns(2)
                with d1:
                    with open(out_img_path, "rb") as f:
                        st.download_button(txt["download_img"], f, file_name="annotated_result.png", mime="image/png")
                with d2:
                    with open(out_json_path, "rb") as f:
                        st.download_button(txt["download_json"], f, file_name="result.json", mime="application/json")

            except Exception as e:
                st.error(f"{txt['error']}: {e}")
            finally:
                os.unlink(tmp_in_path)
        else:
            st.image(uploaded_image, use_container_width=True)

# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader(
        txt["upload_vid"],
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    max_frames = st.number_input(txt["max_frames"], min_value=-1, value=-1, step=10)

    if uploaded_video is not None:
        st.video(uploaded_video)

        run_clicked = st.button(txt["run_vid"], type="primary", key="run_video", disabled=not models_ok)

        if run_clicked:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_in:
                tmp_in.write(uploaded_video.getvalue())
                tmp_in_path = tmp_in.name

            out_dir = tempfile.mkdtemp()
            out_video_path = os.path.join(out_dir, "annotated_result.mp4")

            try:
                config = PipelineConfig(
                    model1_path=model1_path,
                    model2_path=model2_path,
                    input_source=tmp_in_path,
                    output_path=out_video_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    imgsz=imgsz,
                    max_frames=max_frames,
                    enable_tracking=enable_tracking,
                    save_video=True,
                    save_json=True,
                    show_preview=False,
                    verbose=False,
                )
                pipeline = PCBPipeline(config)

                progress_bar = st.progress(0)
                status_text = st.empty()

                def on_progress(current, total):
                    if total > 0:
                        progress_bar.progress(min(current / total, 1.0))
                        status_text.caption(f"{current}/{total}")
                    else:
                        status_text.caption(str(current))

                results = pipeline.process(progress_callback=on_progress)
                progress_bar.progress(1.0)
                status_text.empty()

                stats = results["stats"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(txt["total_frames"], stats["total_frames"])
                m2.metric(txt["pcb_frames"], stats["frames_with_pcb"])
                m3.metric(txt["kpi_defects"], stats["total_defects"])
                m4.metric(txt["fps"], stats["avg_fps"])

                st.markdown(f"**{txt['annotated']}**")
                playable_video_path = out_video_path
                if os.path.exists(out_video_path):
                    with st.spinner(txt["spinner_vid"]):
                        playable_video_path = make_browser_playable(out_video_path)
                    st.video(playable_video_path)
                    if playable_video_path == out_video_path and _FFMPEG_AVAILABLE:
                        st.caption(txt["codec_note"])

                d1, d2 = st.columns(2)
                with d1:
                    if os.path.exists(playable_video_path):
                        with open(playable_video_path, "rb") as f:
                            st.download_button(txt["download_video"], f, file_name="annotated_result.mp4", mime="video/mp4")
                with d2:
                    json_path = results.get("output_json")
                    if json_path and os.path.exists(json_path):
                        with open(json_path, "rb") as f:
                            st.download_button(txt["download_json"], f, file_name="result.json", mime="application/json")

            except Exception as e:
                st.error(f"{txt['error']}: {e}")
            finally:
                if os.path.exists(tmp_in_path):
                    os.unlink(tmp_in_path)
