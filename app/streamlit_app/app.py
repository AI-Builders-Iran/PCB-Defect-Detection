import cv2
import json
import numpy as np
import os
import streamlit as st
import subprocess
import sys
import tempfile
from pathlib import Path

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
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aletheia PCB Vision Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# State Management (Theme & Language)
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ---------------------------------------------------------------------------
# i18n Localization Dictionary
# ---------------------------------------------------------------------------
I18N = {
    "fa": {
        "dir": "rtl",
        "font": "'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "brand_title": "سیستم بینایی ماشین و پایش کیفیت PCB",
        "brand_subtitle": "تشخیص خودکار عیوب مونتاژ و اتصالات با هوش مصنوعی بلادرنگ",
        "tab_image": "تحلیل تک‌فریم (تصویر)",
        "tab_video": "تحلیل پیوسته (ویدئو/استریم)",
        "sidebar_brand": "مرکز کنترل و مانیتورینگ",
        "appearance_sec": "رابط کاربری و محلی‌سازی",
        "theme_toggle": "حالت نمایش",
        "dark": "دارک مود",
        "light": "لایت مود",
        "lang_toggle": "زبان سیستم",
        "model_sec": "تنظیمات پایپ‌لاین و هوش مصنوعی",
        "model1_label": "مدل قطعه‌بندی و مکان‌یابی برد (PCB-SEG)",
        "model2_label": "مدل تشخیص و دسته‌بندی عیوب (Defect Detection)",
        "conf_label": "آستانه اطمینان تشخیص (Confidence)",
        "iou_label": "آستانه اشتراک مکانی (IoU / NMS)",
        "imgsz_label": "وضوح ورودی استنباط (Input Resolution)",
        "tracking_label": "ردیابی عیوب در طول زمان (Defect Tracking)",
        "sys_status": "وضعیت سیستم و منابع",
        "models_ready": "مدل‌ها: آماده استنباط",
        "models_fail": "مدل‌ها: فایل یافت نشد",
        "ffmpeg_ready": "موتور انکودر: H.264 فعال",
        "ffmpeg_fail": "موتور انکودر: غیرفعال",
        "upload_img_prompt": "بارگذاری تصویر برد PCB (فرمت‌های JPG, PNG, BMP)",
        "upload_vid_prompt": "بارگذاری ویدئوی خط تولید PCB (فرمت‌های MP4, AVI, MOV)",
        "btn_run_img": "شروع اسکن و آنالیز عیوب",
        "btn_run_vid": "آغاز پردازش استریم ویدئویی",
        "orig_view": "ورودی خام خط تولید",
        "annotated_view": "خروجی هوش مصنوعی (شناسایی‌شده)",
        "kpi_pcb": "وضعیت حضور برد",
        "kpi_defects": "تعداد کل عیوب",
        "kpi_latency": "تاخیر پردازش فریم",
        "yes": "شناسایی شد",
        "no": "یافت نشد",
        "ms": "میلی‌ثانیه",
        "tbl_title": "لیست عیوب و ویژگی‌های هندسی استخراج‌شده",
        "tbl_class": "کلاس عیب",
        "tbl_conf": "اطمینان",
        "tbl_coords": "مختصات پیوندی (BBox)",
        "down_img": "دانلود تصویر تحلیلی",
        "down_vid": "دانلود ویدئوی پردازش‌شده",
        "down_json": "دانلود تلمتری کامل (JSON)",
        "frames_limit": "محدودیت فریم پردازشی (-۱ به معنای پردازش کامل)",
        "total_frames": "کل فریم‌ها",
        "pcb_frames": "فریم‌های دارای برد",
        "fps_rate": "نرخ فریم (FPS)",
        "err_title": "خطا در فرآیند استنباط",
        "err_model_paths": "مسیر فایل‌های وزن مدل یافت نشد. لطفاً در پنل تنظیمات بررسی نمایید:",
        "spinner_img": "در حال پردازش تصویر...",
        "spinner_vid": "در حال آماده‌سازی ویدیو برای پخش...",
        "progress_start": "شروع پردازش...",
        "progress_frame": "در حال پردازش فریم",
        "progress_done": "پردازش کامل شد ✅",
        "frame_processed": "فریم پردازش شد",
        "preview_caption": "پیش‌نمایش تصویر آپلودشده",
        "codec_note": "اگر پیش‌نمایش بالا نمایش داده نشد (به‌خاطر کدک مرورگر)، فایل رو دانلود کن.",
    },
    "en": {
        "dir": "ltr",
        "font": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
        "brand_title": "PCB Optical Inspection & Quality AI",
        "brand_subtitle": "Real-time automated assembly defect & solder bridge detection suite",
        "tab_image": "Single Frame Analysis",
        "tab_video": "Stream & Video Pipeline",
        "sidebar_brand": "Control & Diagnostics",
        "appearance_sec": "UI & Localization",
        "theme_toggle": "Color Theme",
        "dark": "Dark Mode",
        "light": "Light Mode",
        "lang_toggle": "System Language",
        "model_sec": "Inference Configuration",
        "model1_label": "Board Segmentation Model (PCB-SEG)",
        "model2_label": "Defect Classification Model",
        "conf_label": "Confidence Threshold (Conf)",
        "iou_label": "Spatial Overlap Threshold (IoU)",
        "imgsz_label": "Inference Image Resolution",
        "tracking_label": "Multi-Frame Defect Tracking",
        "sys_status": "System Hardware & Engines",
        "models_ready": "Models: Ready for Inference",
        "models_fail": "Models: Missing Weights",
        "ffmpeg_ready": "Encoder: H.264 Hardware-Ready",
        "ffmpeg_fail": "Encoder: Disabled",
        "upload_img_prompt": "Upload PCB Inspection Frame (JPG, PNG, BMP)",
        "upload_vid_prompt": "Upload Production Line Stream (MP4, AVI, MOV)",
        "btn_run_img": "Execute Defect Scan",
        "btn_run_vid": "Execute Video Inspection",
        "orig_view": "Raw Production Frame",
        "annotated_view": "AI Telemetry Result",
        "kpi_pcb": "PCB Detected",
        "kpi_defects": "Total Defects",
        "kpi_latency": "Inference Latency",
        "yes": "VERIFIED",
        "no": "NONE",
        "ms": "ms",
        "tbl_title": "Extracted Defect Vector Registry",
        "tbl_class": "Defect Class",
        "tbl_conf": "Confidence",
        "tbl_coords": "Bounding Geometry (BBox)",
        "down_img": "Export Annotated Image",
        "down_vid": "Export Processed Video",
        "down_json": "Export Telemetry JSON",
        "frames_limit": "Max Frame Limit (-1 for entire sequence)",
        "total_frames": "Processed Frames",
        "pcb_frames": "Frames w/ Board",
        "fps_rate": "Throughput FPS",
        "err_title": "Pipeline Execution Error",
        "err_model_paths": "Weight files not found at specified paths. Please recheck sidebar config:",
        "spinner_img": "Analyzing frame features...",
        "spinner_vid": "Preparing video for playback...",
        "progress_start": "Starting processing...",
        "progress_frame": "Processing frame",
        "progress_done": "Processing complete ✅",
        "frame_processed": "frame processed",
        "preview_caption": "Uploaded image preview",
        "codec_note": "If the preview above does not play (browser codec), download the file instead.",
    }
}

txt = I18N[st.session_state.lang]


# ---------------------------------------------------------------------------
# High-End Design System Injection
# ---------------------------------------------------------------------------
def apply_custom_theme():
    is_dark = st.session_state.theme == "dark"

    bg_color = "#0B1320" if is_dark else "#FFFFFF"
    card_bg = "#121F33" if is_dark else "#F8FAFC"
    card_border = "#1D3557" if is_dark else "#E5E5E5"
    text_primary = "#FFFFFF" if is_dark else "#1D3557"
    text_secondary = "#94A3B8" if is_dark else "#457B9D"
    accent_blue = "#457B9D"
    deep_navy = "#1D3557"
    sidebar_bg = "#080E18" if is_dark else "#F1F5F9"
    badge_bg = "rgba(69, 123, 157, 0.15)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700;800;900&display=swap');

    * {{
        font-family: {txt["font"]} !important;
        box-sizing: border-box;
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {bg_color} !important;
        color: {text_primary} !important;
        direction: {txt["dir"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-{'left' if txt['dir'] == 'rtl' else 'right'}: 1px solid {card_border} !important;
    }}

    /* Header Brand Card */
    .brand-hero {{
        background: linear-gradient(135deg, {deep_navy} 0%, rgba(69, 123, 157, 0.8) 100%);
        border: 1px solid {accent_blue};
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(29, 53, 87, 0.4);
        position: relative;
        overflow: hidden;
    }}
    .brand-hero::after {{
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transform: rotate(45deg);
    }}
    .brand-hero h1 {{
        color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 900 !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.5px;
    }}
    .brand-hero p {{
        color: #E5E5E5 !important;
        font-size: 14px !important;
        margin: 0 !important;
    }}

    /* KPI Card Block */
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}
    .kpi-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 16px 20px;
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{
        border-color: {accent_blue};
        transform: translateY(-2px);
    }}
    .kpi-title {{
        color: {text_secondary};
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .kpi-val {{
        color: {text_primary};
        font-size: 24px;
        font-weight: 900;
    }}
    .kpi-unit {{
        font-size: 14px;
        font-weight: 500;
        color: {accent_blue};
        margin-{'right' if txt['dir'] == 'rtl' else 'left'}: 4px;
    }}

    /* Buttons */
    div.stButton > button:first-child {{
        background: linear-gradient(135deg, {deep_navy} 0%, {accent_blue} 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid {accent_blue} !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 14px rgba(29, 53, 87, 0.3) !important;
        transition: all 0.25s ease !important;
        width: 100%;
    }}
    div.stButton > button:first-child:hover {{
        box-shadow: 0 6px 20px rgba(69, 123, 157, 0.5) !important;
        transform: translateY(-1px) !important;
        filter: brightness(1.1);
    }}

    div.stDownloadButton > button {{
        background: {card_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s ease;
    }}
    div.stDownloadButton > button:hover {{
        border-color: {accent_blue} !important;
        background: {badge_bg} !important;
    }}

    /* Tabs System */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {card_bg};
        border-radius: 12px;
        padding: 6px;
        border: 1px solid {card_border};
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: {text_secondary} !important;
        font-weight: 700 !important;
        padding: 8px 20px !important;
        border: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {deep_navy} !important;
        color: #FFFFFF !important;
    }}

    /* Pulse Status Indicator */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 700;
        background: {badge_bg};
        border: 1px solid {card_border};
        color: {text_primary};
        margin-bottom: 8px;
        width: 100%;
    }}
    .pulse-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite;
    }}
    .pulse-dot.bad {{
        background-color: #EF4444;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}

    /* Image Preview Container */
    .img-box {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)


apply_custom_theme()

# ---------------------------------------------------------------------------
# Path & Resource Defaults (UNCHANGED LOGIC)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL1 = str(BASE_DIR / "models" / "best-pcb.onnx")
DEFAULT_MODEL2 = str(BASE_DIR / "models" / "best_detect2.onnx")

# ---------------------------------------------------------------------------
# Sidebar Diagnostics & Configuration Suite
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### ⚙️ {txt['sidebar_brand']}")

    # 1. Appearance & Locale Switcher
    with st.expander(f"🌐 {txt['appearance_sec']}", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            theme_choice = st.selectbox(
                txt["theme_toggle"],
                options=["dark", "light"],
                format_func=lambda x: txt[x],
                index=0 if st.session_state.theme == "dark" else 1
            )
        with c2:
            lang_choice = st.selectbox(
                txt["lang_toggle"],
                options=["fa", "en"],
                format_func=lambda x: "فارسی" if x == "fa" else "English",
                index=0 if st.session_state.lang == "fa" else 1
            )

        if theme_choice != st.session_state.theme or lang_choice != st.session_state.lang:
            st.session_state.theme = theme_choice
            st.session_state.lang = lang_choice
            st.rerun()

    # 2. AI Inference Engine Parameters (UNCHANGED LOGIC)
    with st.expander(f"🧠 {txt['model_sec']}", expanded=True):
        model1_path = st.text_input(txt["model1_label"], value=DEFAULT_MODEL1)
        model2_path = st.text_input(txt["model2_label"], value=DEFAULT_MODEL2)

        conf_threshold = st.slider(txt["conf_label"], 0.05, 0.95, 0.25, 0.05)
        iou_threshold = st.slider(txt["iou_label"], 0.05, 0.95, 0.45, 0.05)
        imgsz = st.select_slider(txt["imgsz_label"], options=[320, 416, 512, 640], value=416)
        enable_tracking = st.checkbox(txt["tracking_label"], value=True)

    # 3. Telemetry & Hardware Engine Health
    with st.container():
        models_ok = os.path.exists(model1_path) and os.path.exists(model2_path)

        st.markdown(f"""
        <div class="status-pill">
            <span class="pulse-dot {'bad' if not models_ok else ''}"></span>
            <span>{txt['models_ready'] if models_ok else txt['models_fail']}</span>
        </div>
        <div class="status-pill">
            <span class="pulse-dot {'bad' if not _FFMPEG_AVAILABLE else ''}"></span>
            <span>{txt['ffmpeg_ready'] if _FFMPEG_AVAILABLE else txt['ffmpeg_fail']}</span>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline Loader Cache (UNCHANGED LOGIC)
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


# ---------------------------------------------------------------------------
# Browser-Playable Video Conversion (UNCHANGED LOGIC)
# ---------------------------------------------------------------------------
def make_browser_playable(input_path: str) -> str:
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
# Hero Section
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="brand-hero">
    <h1> {txt['brand_title']}</h1>
    <p>{txt['brand_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(f"⚠️ {txt['err_model_paths']}\n- `{model1_path}`\n- `{model2_path}`")

# ---------------------------------------------------------------------------
# Workspace Tabs
# ---------------------------------------------------------------------------
tab_image, tab_video = st.tabs([f"📸 {txt['tab_image']}", f"🎥 {txt['tab_video']}"])

# ---------------------------------------------------------------------------
# IMAGE PIPELINE WORKSPACE (UNCHANGED LOGIC)
# ---------------------------------------------------------------------------
with tab_image:
    uploaded_image = st.file_uploader(
        txt["upload_img_prompt"],
        type=["jpg", "jpeg", "png", "bmp"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image.name).suffix) as tmp_in:
            tmp_in.write(uploaded_image.getvalue())
            tmp_in_path = tmp_in.name

        if st.button(f"⚡ {txt['btn_run_img']}", type="primary", key="run_image"):
            try:
                pipeline = load_pipeline(model1_path, model2_path, conf_threshold, iou_threshold, imgsz,
                                         enable_tracking)

                with st.spinner(txt["spinner_img"]):
                    result = pipeline.process_image(tmp_in_path)
                    original_frame = cv2.imread(tmp_in_path)
                    annotated = pipeline._draw_results(original_frame, result)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{txt['orig_view']}**")
                    st.image(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.markdown(f"**{txt['annotated_view']}**")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                # KPI Blocks
                st.markdown(f"""
                <div class="kpi-container" style="margin-top: 20px;">
                    <div class="kpi-card">
                        <div class="kpi-title">{txt['kpi_pcb']}</div>
                        <div class="kpi-val" style="color: {'#10B981' if result.has_pcb else '#EF4444'}">
                            {txt['yes'] if result.has_pcb else txt['no']}
                        </div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">{txt['kpi_defects']}</div>
                        <div class="kpi-val">{len(result.defects)}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">{txt['kpi_latency']}</div>
                        <div class="kpi-val">{result.processing_time * 1000:.1f}<span class="kpi-unit">{txt['ms']}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Defect Table
                if result.defects:
                    st.markdown(f"#### 📋 {txt['tbl_title']}")
                    st.dataframe(
                        [
                            {
                                txt["tbl_class"]: d.class_name,
                                txt["tbl_conf"]: round(d.confidence, 3),
                                txt[
                                    "tbl_coords"]: f"({d.bbox['x1']}, {d.bbox['y1']}) - ({d.bbox['x2']}, {d.bbox['y2']})",
                            }
                            for d in result.defects
                        ],
                        use_container_width=True,
                    )

                # Download (UNCHANGED LOGIC)
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
                        st.download_button(f"📥 {txt['down_img']}", f, file_name="annotated_result.png",
                                           mime="image/png")
                with d2:
                    with open(out_json_path, "rb") as f:
                        st.download_button(f"📦 {txt['down_json']}", f, file_name="result.json",
                                           mime="application/json")

            except Exception as e:
                st.error(f"❌ {txt['err_title']}: {e}")
            finally:
                os.unlink(tmp_in_path)
        else:
            st.image(uploaded_image, caption=txt["preview_caption"], use_container_width=True)

# ---------------------------------------------------------------------------
# VIDEO PIPELINE WORKSPACE (UNCHANGED LOGIC)
# ---------------------------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader(
        txt["upload_vid_prompt"],
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )
    max_frames = st.number_input(
        txt["frames_limit"],
        min_value=-1, value=-1, step=10,
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button(f"⚡ {txt['btn_run_vid']}", type="primary", key="run_video"):
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

                progress_bar = st.progress(0, text=txt["progress_start"])
                status_text = st.empty()


                def on_progress(current, total):
                    if total > 0:
                        pct = min(current / total, 1.0)
                        progress_bar.progress(pct, text=f"{txt['progress_frame']} {current}/{total}")
                    else:
                        status_text.text(f"{current} {txt['frame_processed']}")


                results = pipeline.process(progress_callback=on_progress)
                progress_bar.progress(1.0, text=txt["progress_done"])

                stats = results["stats"]

                # Stream KPI Row
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-card">
                        <div class="kpi-title">{txt['total_frames']}</div>
                        <div class="kpi-val">{stats['total_frames']}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">{txt['pcb_frames']}</div>
                        <div class="kpi-val">{stats['frames_with_pcb']}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">{txt['kpi_defects']}</div>
                        <div class="kpi-val">{stats['total_defects']}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">{txt['fps_rate']}</div>
                        <div class="kpi-val">{stats['avg_fps']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**{txt['annotated_view']}**")
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
                            st.download_button(f"📥 {txt['down_vid']}", f, file_name="annotated_result.mp4",
                                               mime="video/mp4")
                with d2:
                    json_path = results.get("output_json")
                    if json_path and os.path.exists(json_path):
                        with open(json_path, "rb") as f:
                            st.download_button(f"📦 {txt['down_json']}", f, file_name="result.json",
                                               mime="application/json")

            except Exception as e:
                st.error(f"❌ {txt['err_title']}: {e}")
            finally:
                if os.path.exists(tmp_in_path):
                    os.unlink(tmp_in_path)
