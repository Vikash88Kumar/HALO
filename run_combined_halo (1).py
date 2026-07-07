"""
Combined HALO runner = batch runner + run_halo logic in ONE file.

Folder:
  TRIAL/
    run_combined_halo.py
    yolo11n.pt / yolov8n.pt / detection.pt
    INPUT_VIDEOS/
      video1.mp4
    OUTPUT/

Install:
  pip install ultralytics opencv-python numpy torch

Run all videos:
  python run_combined_halo.py

Run all videos with another model:
  python run_combined_halo.py --weights yolov8n.pt

Run one video:
  python run_combined_halo.py --video "video1.mp4" --weights yolo11n.pt
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "INPUT_VIDEOS"
DEFAULT_OUTPUT_DIR = BASE_DIR / "OUTPUT"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PLAYER_LABELS = {"person", "player", "goalkeeper", "referee"}
BALL_LABELS = {"sports ball", "ball"}
JERSEY_IMG_SIZE = 64


# -------------------- Optional Jersey CNN --------------------

class SpatialTransformer(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 7, padding=3), nn.MaxPool2d(2), nn.ReLU(True),
            nn.Conv2d(16, 32, 5, padding=2), nn.MaxPool2d(2), nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (JERSEY_IMG_SIZE // 4) ** 2, 64), nn.ReLU(True), nn.Linear(64, 6)
        )
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.loc(x).view(x.size(0), -1)
        theta = self.fc(theta).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class JerseyCNN(nn.Module):
    def __init__(self, num_digit_classes: int = 11):
        super().__init__()
        self.stn = SpatialTransformer()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(True), nn.Dropout(0.3))
        self.head_visible = nn.Linear(128, 2)
        self.head_tens = nn.Linear(128, num_digit_classes)
        self.head_units = nn.Linear(128, num_digit_classes)

    def forward(self, x: torch.Tensor):
        x = self.stn(x)
        feat = self.trunk(self.backbone(x))
        return self.head_visible(feat), self.head_tens(feat), self.head_units(feat)


def load_jersey_model(path: Optional[Path], device: torch.device) -> Optional[JerseyCNN]:
    if not path or not path.exists():
        print("[INFO] No jersey CNN found. Jersey numbers will show as '?'.")
        return None
    try:
        model = JerseyCNN().to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"[OK] Loaded jersey CNN: {path}")
        return model
    except Exception as exc:
        print(f"[WARN] Could not load jersey CNN: {exc}")
        return None


@torch.no_grad()
def predict_jersey_number(frame_bgr: np.ndarray, box_xyxy: np.ndarray, model: JerseyCNN, device: torch.device) -> Optional[str]:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    # upper torso crop
    yy1 = max(0, y1)
    yy2 = min(frame_bgr.shape[0], y1 + int(h * 0.45))
    xx1 = max(0, x1)
    xx2 = min(frame_bgr.shape[1], x2)
    crop = frame_bgr[yy1:yy2, xx1:xx2]
    if crop.size == 0 or w < 10 or h < 20:
        return None

    crop = cv2.resize(crop, (JERSEY_IMG_SIZE, JERSEY_IMG_SIZE))
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    crop_t = torch.from_numpy(crop_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)

    p_vis, p_tens, p_units = model(crop_t)
    if int(p_vis.argmax(dim=1).item()) != 1:
        return None

    tens = int(p_tens.argmax(dim=1).item())
    units = int(p_units.argmax(dim=1).item())
    if tens == 10 or units == 10:  # 10 means blank
        return None
    return f"{tens}{units}"


# -------------------- Simple Tracker --------------------

@dataclass
class Track:
    tid: int
    center: Tuple[float, float]
    box: np.ndarray
    missed: int = 0
    jersey_votes: Dict[str, int] = field(default_factory=dict)

    @property
    def jersey(self) -> str:
        if not self.jersey_votes:
            return "?"
        return max(self.jersey_votes.items(), key=lambda kv: kv[1])[0]


class SimpleCentroidTracker:
    def __init__(self, max_dist_px: float = 90.0, max_missed: int = 20):
        self.max_dist_px = max_dist_px
        self.max_missed = max_missed
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    @staticmethod
    def center(box: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, boxes: List[np.ndarray]) -> Dict[int, Track]:
        for t in self.tracks.values():
            t.missed += 1

        unmatched = set(range(len(boxes)))
        for tid, tr in list(self.tracks.items()):
            best_i, best_d = None, float("inf")
            for i in unmatched:
                c = self.center(boxes[i])
                d = math.hypot(c[0] - tr.center[0], c[1] - tr.center[1])
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None and best_d <= self.max_dist_px:
                tr.center = self.center(boxes[best_i])
                tr.box = boxes[best_i]
                tr.missed = 0
                unmatched.remove(best_i)

        for i in list(unmatched):
            box = boxes[i]
            self.tracks[self.next_id] = Track(self.next_id, self.center(box), box)
            self.next_id += 1

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

        return self.tracks


# -------------------- Helpers --------------------

def maybe_extract_package(base_dir: Path) -> None:
    zip_path = base_dir / "halo_run_package.zip"
    extracted = base_dir / "halo_run_extracted"
    if zip_path.exists() and not extracted.exists():
        print(f"[INFO] Extracting {zip_path} -> {extracted}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted)


def find_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def auto_find_yolo_weights(base_dir: Path) -> Optional[Path]:
    patterns = [
        "halo_run/checkpoints/yolo*/best.pt",
        "halo_run_extracted/checkpoints/yolo*/best.pt",
        "halo_run_extracted/**/weights/best.pt",
        "halo_run_extracted/**/best.pt",
        "runs/detect/**/weights/best.pt",
        "Model_weights_1/detection.pt",
        "Model_weights_1/best.pt",
        "detection.pt",
        "best.pt",
        "yolo11n.pt",
        "yolov11n.pt",
        "yolov8n.pt",
    ]
    for pat in patterns:
        matches = sorted(base_dir.glob(pat))
        if matches:
            return matches[0]
    return None


def resolve_weights(base_dir: Path, weights_arg: Optional[str]) -> Path:
    if weights_arg:
        p = Path(weights_arg)
        options = [p, base_dir / weights_arg, base_dir / "Model_weights_1" / weights_arg]
        found = find_first_existing(options)
        if found:
            return found.resolve()
        raise FileNotFoundError(f"Weights not found: {weights_arg}")

    found = auto_find_yolo_weights(base_dir)
    if found:
        return found.resolve()

    raise FileNotFoundError("No .pt detector found. Put yolo11n.pt / yolov8n.pt / detection.pt in this folder.")


def auto_find_jersey_weights(base_dir: Path) -> Optional[Path]:
    patterns = [
        "halo_run/checkpoints/jersey_cnn/best.pt",
        "halo_run_extracted/checkpoints/jersey_cnn/best.pt",
        "halo_run_extracted/**/jersey_cnn/best.pt",
        "Model_weights_1/jersey_ocr.pt",
        "jersey_ocr.pt",
        "jersey_cnn_best.pt",
    ]
    for pat in patterns:
        matches = sorted(base_dir.glob(pat))
        if matches:
            return matches[0].resolve()
    return None


def resolve_jersey_weights(base_dir: Path, jersey_arg: Optional[str]) -> Optional[Path]:
    if jersey_arg:
        p = Path(jersey_arg)
        options = [p, base_dir / jersey_arg, base_dir / "Model_weights_1" / jersey_arg]
        return find_first_existing(options)
    return auto_find_jersey_weights(base_dir)


def collect_videos(input_dir: Path, video_arg: Optional[str]) -> List[Path]:
    if video_arg:
        p = Path(video_arg)
        if p.exists():
            return [p.resolve()]
        p2 = input_dir / video_arg
        if p2.exists():
            return [p2.resolve()]
        raise FileNotFoundError(f"Video not found: {video_arg}")

    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_VIDEOS folder not found: {input_dir}")

    videos = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No videos found inside: {input_dir}")
    return videos


def label_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id)).lower()
    return str(names[class_id]).lower() if class_id < len(names) else str(class_id)


def is_player_label(name: str) -> bool:
    return name.strip().lower() in PLAYER_LABELS


def is_ball_label(name: str) -> bool:
    return name.strip().lower() in BALL_LABELS


def draw_text(frame: np.ndarray, text: str, xy: Tuple[int, int], scale: float = 0.7) -> None:
    x, y = xy
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def detect_touch_events(ball_history, player_history, threshold_px: float) -> List[dict]:
    events = []
    last_touch = None
    players_by_frame: Dict[int, List[Tuple[int, float, float]]] = {}

    for tid, pts in player_history.items():
        for f, x, y in pts:
            players_by_frame.setdefault(f, []).append((tid, x, y))

    for f, bx, by in ball_history:
        best_tid, best_d = None, float("inf")
        for tid, px, py in players_by_frame.get(f, []):
            d = math.hypot(px - bx, py - by)
            if d < best_d:
                best_tid, best_d = tid, d

        if best_tid is not None and best_d <= threshold_px:
            events.append({"frame": f, "type": "touch", "track_id": best_tid, "distance_px": round(best_d, 2)})
            if last_touch is not None and last_touch != best_tid:
                events.append({"frame": f, "type": "pass", "from": last_touch, "to": best_tid})
            last_touch = best_tid

    return events


# -------------------- Core video processing --------------------

def process_video(video_path: Path, output_path: Path, yolo: YOLO, weights: Path, jersey_model, device, args) -> dict:
    print("\n" + "=" * 70)
    print(f"[RUNNING] {video_path.name}")
    print(f"[OUTPUT]  {output_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = SimpleCentroidTracker(max_dist_px=args.track_distance, max_missed=args.max_missed)
    player_history: Dict[int, List[Tuple[int, float, float]]] = {}
    ball_history: List[Tuple[int, float, float]] = []

    frame_idx = 0
    names = yolo.names
    yolo_device = 0 if device.type == "cuda" else "cpu"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        result = yolo.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False, device=yolo_device)[0]
        boxes = result.boxes

        player_boxes: List[np.ndarray] = []
        ball_centers: List[Tuple[float, float]] = []

        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy()

            for box, cid, score in zip(xyxy, cls_ids, confs):
                name = label_name(names, cid)
                x1, y1, x2, y2 = [int(v) for v in box]

                if is_player_label(name):
                    player_boxes.append(box)
                elif is_ball_label(name):
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    ball_centers.append((cx, cy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    draw_text(frame, f"ball {score:.2f}", (x1, max(20, y1 - 8)), scale=0.6)
                elif args.draw_all:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    draw_text(frame, f"{name} {score:.2f}", (x1, max(20, y1 - 8)), scale=0.5)

        tracks = tracker.update(player_boxes)

        for tid, tr in tracks.items():
            if tr.missed != 0:
                continue

            x1, y1, x2, y2 = [int(v) for v in tr.box]
            cx, cy = tr.center
            player_history.setdefault(tid, []).append((frame_idx, cx, cy))

            if jersey_model is not None and frame_idx % args.jersey_every == 0:
                number = predict_jersey_number(frame, tr.box, jersey_model, device)
                if number is not None:
                    tr.jersey_votes[number] = tr.jersey_votes.get(number, 0) + 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_text(frame, f"id{tid} #{tr.jersey}", (x1, max(24, y1 - 10)), scale=0.7)

        if ball_centers:
            bx, by = ball_centers[0]
            ball_history.append((frame_idx, bx, by))
            cv2.circle(frame, (int(bx), int(by)), int(args.touch_radius), (0, 165, 255), 2)

        draw_text(frame, f"frame {frame_idx}", (12, 28), scale=0.7)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx} frames...")

    cap.release()
    writer.release()

    events = detect_touch_events(ball_history, player_history, threshold_px=args.touch_radius)
    summary = {
        "input_video": str(video_path),
        "output_video": str(output_path),
        "weights": str(weights),
        "frames_processed": frame_idx,
        "players_seen": sorted(list(player_history.keys())),
        "ball_detections": len(ball_history),
        "events": events[:500],
        "note": "Combined single-file runner: batch INPUT_VIDEOS loop + run_halo inference logic.",
    }

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Annotated video: {output_path}")
    print(f"[DONE] Event JSON:      {json_path}")
    print(f"[DONE] Frames: {frame_idx} | Players: {len(player_history)} | Ball detections: {len(ball_history)} | Events: {len(events)}")
    return summary


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined HALO batch + inference runner")
    parser.add_argument("--video", default=None, help="Single video path/name. If empty, runs all videos in INPUT_VIDEOS.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input videos folder")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    parser.add_argument("--weights", default=None, help="YOLO weights: yolo11n.pt, yolov8n.pt, detection.pt, etc.")
    parser.add_argument("--jersey-weights", default=None, help="Optional jersey CNN weights")
    parser.add_argument("--workdir", default=str(BASE_DIR), help="Folder containing weights/package")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--track-distance", type=float, default=90.0, help="Max centroid distance for same ID")
    parser.add_argument("--max-missed", type=int, default=20, help="Frames to keep missing track alive")
    parser.add_argument("--touch-radius", type=float, default=70.0, help="Pixel radius for simple touch/pass events")
    parser.add_argument("--jersey-every", type=int, default=10, help="Run jersey CNN every N frames per track")
    parser.add_argument("--draw-all", action="store_true", help="Draw other classes too")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.workdir).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    maybe_extract_package(base_dir)

    videos = collect_videos(input_dir, args.video)
    weights = resolve_weights(base_dir, args.weights)
    jersey_weights = resolve_jersey_weights(base_dir, args.jersey_weights)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Weights: {weights}")
    print(f"[INFO] Videos found: {len(videos)}")
    print(f"[INFO] Output folder: {output_dir}")

    yolo = YOLO(str(weights))
    jersey_model = load_jersey_model(jersey_weights, device)

    all_summaries = []
    for video in videos:
        output_path = output_dir / f"{video.stem}_annotated.mp4"
        summary = process_video(video, output_path, yolo, weights, jersey_model, device, args)
        all_summaries.append(summary)

    summary_path = output_dir / "combined_run_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("[ALL DONE]")
    print(f"Processed videos: {len(all_summaries)}")
    print(f"Outputs saved in: {output_dir}")
    print(f"Summary saved at: {summary_path}")


if __name__ == "__main__":
    main()
