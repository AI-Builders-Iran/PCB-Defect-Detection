import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")


class InputType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"


class OutputFormat(Enum):
    VIDEO = "video"
    JSON = "json"
    BOTH = "both"


@dataclass
class DefectTrack:
    """اطلاعات ردیابی یک عیب"""
    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: Dict[str, int]
    center: Dict[str, int]
    first_seen: int
    last_seen: int
    trajectory: List[Dict[str, int]] = field(default_factory=list)


@dataclass
class DetectionResult:
    frame_number: int
    has_pcb: bool
    pcb_mask: Optional[np.ndarray]
    pcb_id: int = 0
    defects: List[DefectTrack] = field(default_factory=list)
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "frame": self.frame_number,
            "has_pcb": self.has_pcb,
            "pcb_id": self.pcb_id,
            "defect_count": len(self.defects),
            "defects": [
                {
                    "track_id": d.track_id,
                    "class": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "center": d.center
                } for d in self.defects
            ],
            "processing_time_ms": round(self.processing_time * 1000, 2)
        }


@dataclass
class PipelineConfig:
    model1_path: str = "PCB-SEG.pt"
    model2_path: str = "defect_model.pt"

    input_source: Union[str, int] = 0
    output_path: Optional[str] = None

    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 416
    max_frames: int = -1

    enable_tracking: bool = True
    tracker_config: str = "bytetrack.yaml"
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 30
    match_thresh: float = 0.8

    output_format: OutputFormat = OutputFormat.BOTH
    save_video: bool = True
    save_json: bool = True
    show_preview: bool = False

    colors: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    use_onnx: bool = False
    device: str = "cpu"
    verbose: bool = True


class PCBPipeline:
    def __init__(self, config: Union[PipelineConfig, Dict]):
        if isinstance(config, dict):
            self.config = PipelineConfig(**config)
        else:
            self.config = config

        self._models_loaded = False
        self._model1 = None
        self._model2 = None
        self._frame_count = 0
        self._results: List[DetectionResult] = []
        self._fps_history: List[float] = []

        self._class_colors = {}

        self._load_models()

    def _load_models(self):
        if self.config.verbose:
            print("🔄 در حال بارگذاری مدل‌ها...")

        try:
            self._model1 = YOLO(self.config.model1_path)
            if self.config.verbose:
                print(f"✅ مدل PCB-SEG بارگذاری شد: {self.config.model1_path}")
        except Exception as e:
            raise RuntimeError(f"❌ خطا در بارگذاری مدل PCB-SEG: {e}")

        try:
            self._model2 = YOLO(self.config.model2_path)
            if self.config.verbose:
                print(f"✅ مدل Defect Detection بارگذاری شد: {self.config.model2_path}")
        except Exception as e:
            raise RuntimeError(f"❌ خطا در بارگذاری مدل Defect Detection: {e}")

        self._models_loaded = True

    def _get_color_for_class(self, class_id: int, class_name: str = "") -> Tuple[int, int, int]:
        if class_id not in self._class_colors:
            palette = [
                (255, 50, 50),
                (50, 255, 50),
                (50, 50, 255),
                (255, 255, 50),
                (255, 50, 255),
                (50, 255, 255),
                (255, 128, 0),
                (128, 0, 255),
                (0, 255, 128),
                (255, 0, 128),
            ]
            self._class_colors[class_id] = palette[class_id % len(palette)]
        return self._class_colors[class_id]

    def _get_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def _process_frame(self, frame: np.ndarray, frame_num: int) -> DetectionResult:
        start_time = time.time()

        results1 = self._model1(frame, imgsz=self.config.imgsz, verbose=False)

        mask = None
        has_pcb = False
        pcb_id = 0

        if results1 and len(results1) > 0 and results1[0].masks is not None:
            masks_data = results1[0].masks.data
            if masks_data is not None and len(masks_data) > 0:
                has_pcb = True
                h, w = frame.shape[:2]
                combined_mask = np.zeros((h, w), dtype=np.uint8)

                for m in masks_data:
                    m_np = m.cpu().numpy()
                    if m_np.size > 0:
                        m_resized = cv2.resize(m_np, (w, h))
                        combined_mask = cv2.bitwise_or(
                            combined_mask,
                            (m_resized > 0.5).astype(np.uint8) * 255
                        )

                mask = combined_mask

        defects = []

        if mask is not None and np.any(mask > 0):
            x, y, w, h = cv2.boundingRect(mask)
            if w > 10 and h > 10:
                margin = 5
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                cropped_pcb = frame[y1:y2, x1:x2]

                if self.config.enable_tracking:
                    track_args = {
                        'persist': True,
                        'tracker': self.config.tracker_config,
                        'conf': self.config.conf_threshold,
                        'iou': self.config.iou_threshold,
                        'imgsz': self.config.imgsz,
                        'verbose': False
                    }
                    results2 = self._model2.track(cropped_pcb, **track_args)
                else:
                    results2 = self._model2(
                        cropped_pcb,
                        imgsz=self.config.imgsz,
                        conf=self.config.conf_threshold,
                        iou=self.config.iou_threshold,
                        verbose=False
                    )

                defects = self._extract_defects(results2, offset_x=x1, offset_y=y1, frame_num=frame_num)
            else:
                defects = []
        else:
            if self.config.enable_tracking:
                results2 = self._model2.track(
                    frame,
                    persist=True,
                    tracker=self.config.tracker_config,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.imgsz,
                    verbose=False
                )
            else:
                results2 = self._model2(
                    frame,
                    imgsz=self.config.imgsz,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    verbose=False
                )
            defects = self._extract_defects(results2, offset_x=0, offset_y=0, frame_num=frame_num)

        processing_time = time.time() - start_time

        return DetectionResult(
            frame_number=frame_num,
            has_pcb=has_pcb,
            pcb_mask=mask,
            pcb_id=pcb_id,
            defects=defects,
            processing_time=processing_time
        )

    def _extract_defects(self, results, offset_x: int = 0, offset_y: int = 0, frame_num: int = 0) -> List[DefectTrack]:
        defects = []

        if not results or len(results) == 0:
            return defects

        result = results[0]
        if result.boxes is None:
            return defects

        boxes = result.boxes
        for i, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls] if hasattr(result, 'names') else str(cls)

                track_id = -1
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = frame_num * 100 + i

                defects.append(DefectTrack(
                    track_id=track_id,
                    class_name=label,
                    class_id=cls,
                    confidence=conf,
                    bbox={
                        "x1": x1 + offset_x,
                        "y1": y1 + offset_y,
                        "x2": x2 + offset_x,
                        "y2": y2 + offset_y,
                        "width": x2 - x1,
                        "height": y2 - y1
                    },
                    center={
                        "x": (x1 + x2) // 2 + offset_x,
                        "y": (y1 + y2) // 2 + offset_y
                    },
                    first_seen=frame_num,
                    last_seen=frame_num,
                    trajectory=[]
                ))
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️ خطا در استخراج عیب: {e}")
                continue

        return defects

    def _draw_fancy_box(self, frame: np.ndarray, defect: DefectTrack) -> np.ndarray:
        bbox = defect.bbox
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        track_id = defect.track_id
        label = defect.class_name
        conf = defect.confidence

        color = self._get_color_for_class(defect.class_id, label)

        thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

        text = f"ID:{track_id} {label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)

        text_x = x1
        text_y = y1 - 5

        if text_y - th < 0:
            text_y = y2 + th + 5

        bg_x1 = text_x - 2
        bg_y1 = text_y - th - 2
        bg_x2 = text_x + tw + 2
        bg_y2 = text_y + 2

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def _draw_results(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if result.pcb_mask is not None and np.any(result.pcb_mask > 0):
            contours, _ = cv2.findContours(result.pcb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            overlay = annotated.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)

            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 3, cv2.LINE_AA)

            text = f"PCB ID:{result.pcb_id}"
            cv2.putText(annotated, text, (10, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        for defect in result.defects:
            annotated = self._draw_fancy_box(annotated, defect)

        defect_count = len(result.defects)

        cv2.rectangle(annotated, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.line(annotated, (0, 40), (w, 40), (100, 100, 100), 1)

        info_text = f"Frame: {result.frame_number}  |  Defects: {defect_count}  |  PCB: {'Yes' if result.has_pcb else 'No'}"
        cv2.putText(annotated, info_text, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if self._fps_history:
            avg_fps = sum(self._fps_history[-30:]) / len(self._fps_history[-30:])
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(annotated, fps_text, (w - 120, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 100), 1, cv2.LINE_AA)

        return annotated

    def process(self, progress_callback=None) -> Dict:
        """
        پردازش کامل یک ویدیو.
        progress_callback(frame_idx, total_frames) در صورت وجود بعد از هر فریم صدا زده می‌شود.
        """
        if not self._models_loaded:
            raise RuntimeError("❌ مدل‌ها بارگذاری نشده‌اند")

        source = self.config.input_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"❌ نمی‌توان منبع {source} را باز کرد")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = None
        out_writer = None

        if self.config.save_video:
            if self.config.output_path is None:
                input_name = Path(str(source)).stem if isinstance(source, str) else "webcam"
                output_path = f"output_{input_name}_{int(time.time())}.mp4"
            else:
                output_path = self.config.output_path

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self._frame_count = 0
        self._results = []
        self._fps_history = []

        start_total = time.time()

        while True:
            frame = self._get_frame(cap)
            if frame is None:
                break

            if self.config.max_frames > 0 and self._frame_count >= self.config.max_frames:
                break

            result = self._process_frame(frame, self._frame_count)
            self._results.append(result)

            if result.processing_time > 0:
                self._fps_history.append(1.0 / result.processing_time)

            annotated = self._draw_results(frame, result)

            if out_writer is not None:
                out_writer.write(annotated)

            self._frame_count += 1

            if progress_callback is not None:
                progress_callback(self._frame_count, total_frames)

        cap.release()
        if out_writer is not None:
            out_writer.release()

        total_time = time.time() - start_total
        avg_fps = self._frame_count / total_time if total_time > 0 else 0

        stats = {
            "total_frames": self._frame_count,
            "total_time_seconds": round(total_time, 2),
            "avg_fps": round(avg_fps, 2),
            "frames_with_pcb": sum(1 for r in self._results if r.has_pcb),
            "total_defects": sum(len(r.defects) for r in self._results),
            "avg_processing_time_ms": round(
                sum(r.processing_time for r in self._results) / len(self._results) * 1000, 2
            ) if self._results else 0
        }

        json_path = None
        if self.config.save_json:
            json_path = output_path.replace('.mp4', '.json') if output_path else "results.json"
            self._save_json(json_path, stats)

        return {
            "stats": stats,
            "results": [r.to_dict() for r in self._results],
            "output_video": output_path,
            "output_json": json_path
        }

    def _save_json(self, json_path: str, stats: Dict) -> None:
        output_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                    for k, v in self.config.__dict__.items()
                    if not k.startswith("_")
                }
            },
            "statistics": stats,
            "frames": [r.to_dict() for r in self._results]
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def process_image(self, image_path: str) -> DetectionResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"❌ نمی‌توان تصویر {image_path} را باز کرد")

        result = self._process_frame(frame, 0)
        return result