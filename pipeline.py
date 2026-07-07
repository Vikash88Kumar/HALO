"""
JerseyIQ video processing pipeline.

Ties together three real, loaded models:
  1. detection_best.pt  -- Ultralytics YOLO, classes {0: 'football', 1: 'player'}
  2. jersey_ocr_best.pt -- JerseyCNN (STN + two-digit head) from models.py
  3. ccnn_best.pt       -- CCNNFilter, a temporal residual Conv1d touch/possession filter

IMPORTANT / HONEST CAVEATS (please read):
  - The exact input features the CCNN touch-filter was trained on were not
    included in the uploaded package (no training script for it was
    provided), so this pipeline builds a principled 3-channel feature
    (ball-distance, player speed, ball speed - all normalized) and feeds it
    through the *real* ccnn_best.pt weights. If your training used a
    different feature definition, edit `build_feature_vector()` below to
    match it exactly -- the model architecture itself is reverse-engineered
    precisely from the checkpoint shapes and will load correctly regardless.
  - "Shot" detection and "pass" detection are rule-based heuristics on top
    of the real detection/tracking/touch outputs (no dedicated shot/pass
    model was included in the package). They are clearly labelled as
    heuristics in the code/comments below.
  - Jersey numbers are inferred with the real STN-CNN on a torso crop taken
    from the player bounding box, sampled periodically per track and
    aggregated by confidence-weighted majority vote.
"""
import time
import os
import cv2
import numpy as np
import torch

from models import (
    JERSEY_INPUT_SIZE,
    CCNN_WINDOW,
    load_jersey_cnn,
    load_ccnn_filter,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
JERSEY_SAMPLE_EVERY = 8          # frames between jersey-OCR samples per track
TOUCH_MAX_DIST_FRAC = 0.06       # max ball<->player distance (frac of frame diag) to count as a touch candidate
TOUCH_PROB_THRESHOLD = 0.55      # ccnn touch probability threshold
PASS_MAX_GAP_SEC = 2.5           # max seconds between touches to call it a pass rather than a fresh touch
SHOT_SPEED_PERCENTILE = 97       # ball speed percentile treated as a "shot / long strike" heuristic
MIN_TRACK_FRAMES = 5             # ignore ultra-short spurious tracks when counting players
FLAG_CONF_THRESHOLD = 0.45       # jersey OCR confidence below this -> flagged for review

TEAM_A_COLOR_BGR = (193, 227, 79)   # from --team-a #4FE3C1 (cyan-ish), BGR order
TEAM_B_COLOR_BGR = (79, 185, 227)   # from --team-b #E3B94F (gold-ish), BGR order
BALL_COLOR_BGR = (60, 155, 200)     # gold
FLAG_COLOR_BGR = (60, 155, 200)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def crop_torso(frame, box, pad_top=0.05, bottom_frac=0.65):
    """Crop the upper/torso portion of a player bbox, where jersey numbers
    are usually printed (on the back) or chest (on the front)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    box_h = y2 - y1
    top = int(y1 + pad_top * box_h)
    bottom = int(y1 + bottom_frac * box_h)
    bottom = max(bottom, top + 1)
    crop = frame[top:bottom, x1:x2]
    if crop.size == 0:
        return None
    return crop


def preprocess_crop_for_cnn(crop):
    img = cv2.resize(crop, (JERSEY_INPUT_SIZE, JERSEY_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def kmeans_teams(color_samples):
    """color_samples: dict track_id -> mean BGR color (np.array shape (3,))
    Returns dict track_id -> 0/1 team label."""
    ids = list(color_samples.keys())
    if len(ids) < 2:
        return {tid: 0 for tid in ids}
    data = np.array([color_samples[t] for t in ids], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    k = min(2, len(ids))
    _, labels, _ = cv2.kmeans(data, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    return {tid: int(labels[i][0]) for i, tid in enumerate(ids)}


class VideoProcessor:
    def __init__(self, detector, jersey_model=None, ccnn_model=None, device=DEVICE):
        self.detector = detector          # ultralytics.YOLO instance
        self.jersey_model = jersey_model  # models.JerseyCNN or None
        self.ccnn_model = ccnn_model      # models.CCNNFilter or None
        self.device = device

    # ---------------------------------------------------------------- pass 1
    def _analyze(self, video_path, progress_cb):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        diag = float(np.hypot(w, h)) or 1.0

        track_positions = {}   # track_id -> list of (frame_idx, cx, cy)
        track_colors = {}      # track_id -> running mean BGR color (np array)
        track_color_n = {}
        jersey_votes = {}      # track_id -> list of (number, confidence)
        ball_positions = {}    # frame_idx -> (cx, cy)

        frame_idx = 0
        t0 = time.time()

        gen = self.detector.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            classes=None,
        )

        # Dynamically map class IDs by name
        player_class_ids = []
        ball_class_ids = []
        names_dict = getattr(self.detector, "names", {}) or {}
        for cid, name in names_dict.items():
            name_lower = name.lower().strip()
            if name_lower in {"person", "player", "goalkeeper", "referee"}:
                player_class_ids.append(cid)
            elif name_lower in {"sports ball", "ball", "football"}:
                ball_class_ids.append(cid)

        # Fallbacks if names dictionary was empty or unexpected
        if not player_class_ids:
            player_class_ids = [1]  # default player class ID in custom model
        if not ball_class_ids:
            ball_class_ids = [0]    # default ball class ID in custom model

        for result in gen:
            frame = result.orig_img
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.array([-1] * len(xyxy), dtype=int)
            else:
                xyxy, cls, ids, confs = np.empty((0, 4)), np.empty((0,)), np.empty((0,)), np.empty((0,))

            # --- ball ---
            ball_idx = np.array([i for i, c in enumerate(cls) if c in ball_class_ids], dtype=np.int64)
            if len(ball_idx) > 0:
                best = ball_idx[np.argmax(confs[ball_idx])]
                ball_positions[frame_idx] = _bbox_center(xyxy[best])

            # --- players ---
            player_idx = np.array([i for i, c in enumerate(cls) if c in player_class_ids], dtype=np.int64)
            for pi in player_idx:
                tid = int(ids[pi])
                if tid == -1:
                    continue
                box = xyxy[pi]
                center = _bbox_center(box)
                bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
                track_positions.setdefault(tid, []).append((frame_idx, center[0], center[1], bw, bh))

                if frame_idx % JERSEY_SAMPLE_EVERY == 0 or tid not in track_colors:
                    crop = crop_torso(frame, box)
                    if crop is not None:
                        # team color (mean BGR of the torso crop)
                        mean_color = crop.reshape(-1, 3).mean(axis=0)
                        prev = track_colors.get(tid, np.zeros(3))
                        n = track_color_n.get(tid, 0)
                        track_colors[tid] = (prev * n + mean_color) / (n + 1)
                        track_color_n[tid] = n + 1

                        if self.jersey_model is not None:
                            tensor = preprocess_crop_for_cnn(crop).to(self.device)
                            pred = self.jersey_model.predict_number(tensor)[0]
                            jersey_votes.setdefault(tid, []).append((pred["number"], pred["confidence"], pred["visible"]))

            frame_idx += 1
            if total_frames and frame_idx % 10 == 0:
                pct = 5 + int(50 * frame_idx / total_frames)  # 5-55% of overall progress
                progress_cb(min(pct, 55), f"Ingest+Process: frame {frame_idx}/{total_frames}")

        cap.release()
        elapsed = max(time.time() - t0, 1e-6)
        processing_fps = round(frame_idx / elapsed, 1)

        return {
            "fps_source": round(fps_source, 2),
            "total_frames": frame_idx,
            "frame_w": w,
            "frame_h": h,
            "diag": diag,
            "track_positions": track_positions,
            "track_colors": track_colors,
            "jersey_votes": jersey_votes,
            "ball_positions": ball_positions,
            "processing_fps": processing_fps,
        }

    # ------------------------------------------------------- feature/events
    def _build_ball_speed(self, ball_positions, total_frames):
        speeds = np.zeros(total_frames, dtype=np.float32)
        last = None
        for f in range(total_frames):
            pos = ball_positions.get(f)
            if pos is not None and last is not None:
                speeds[f] = _dist(pos, last)
            if pos is not None:
                last = pos
        return speeds

    def _nearest_player_per_frame(self, track_positions, ball_positions, diag, total_frames):
        """Returns dict frame_idx -> (track_id, dist_frac) for the nearest
        player to the ball, restricted to a plausible touch distance."""
        by_frame = {}
        for tid, pts in track_positions.items():
            for (f, cx, cy, bw, bh) in pts:
                by_frame.setdefault(f, []).append((tid, cx, cy))

        nearest = {}
        for f in range(total_frames):
            ball = ball_positions.get(f)
            if ball is None or f not in by_frame:
                continue
            best_tid, best_dist = None, None
            for tid, cx, cy in by_frame[f]:
                d = _dist((cx, cy), ball) / diag
                if best_dist is None or d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid is not None:
                nearest[f] = (best_tid, best_dist)
        return nearest

    # ------------------------------------------------------- events / video
    def process(self, video_path, output_path, progress_cb=None):
        if progress_cb is None:
            progress_cb = lambda pct, msg: None

        progress_cb(2, "Ingest: loading detector + tracker...")
        analysis = self._analyze(video_path, progress_cb)

        progress_cb(58, "Process: clustering teams + aggregating jersey numbers...")
        team_of_track = kmeans_teams(analysis["track_colors"])

        # aggregate jersey numbers per track (confidence-weighted majority vote)
        jersey_number_of_track = {}
        avg_conf_all = []
        for tid, votes in analysis["jersey_votes"].items():
            tallies = {}
            for number, conf, visible in votes:
                if not visible or number is None:
                    continue
                tallies.setdefault(number, 0.0)
                tallies[number] += conf
                avg_conf_all.append(conf)
            if tallies:
                best_number = max(tallies, key=tallies.get)
                total_w = sum(tallies.values())
                jersey_number_of_track[tid] = {
                    "number": best_number,
                    "confidence": round(tallies[best_number] / max(len(votes), 1), 3),
                }

        total_frames = analysis["total_frames"]
        diag = analysis["diag"]
        ball_positions = analysis["ball_positions"]
        track_positions = analysis["track_positions"]

        ball_speeds = self._build_ball_speed(ball_positions, total_frames)
        nearest = self._nearest_player_per_frame(track_positions, ball_positions, diag, total_frames)
        touch_probs = self._compute_touch_probs_batch(nearest, track_positions, ball_positions, diag, total_frames)

        progress_cb(65, "Synthesize: building event log + commentary...")
        players_tracked = sum(
            1 for tid, pts in track_positions.items() if len(pts) >= MIN_TRACK_FRAMES
        )
        events, commentary, kpis = self._synthesize_events(
            total_frames, analysis["fps_source"], nearest, touch_probs, ball_speeds,
            jersey_number_of_track, team_of_track, players_tracked,
        )

        progress_cb(72, "Synthesize: rendering annotated video...")
        self._render_annotated_video(
            video_path, output_path, track_positions, jersey_number_of_track,
            team_of_track, ball_positions, total_frames, progress_cb,
        )

        duration_sec = round(total_frames / analysis["fps_source"], 1) if analysis["fps_source"] else 0
        avg_ocr_confidence = round(100 * (sum(avg_conf_all) / len(avg_conf_all)), 1) if avg_conf_all else 0.0
        kpis["avg_ocr_confidence"] = avg_ocr_confidence
        kpis["processing_fps"] = analysis["processing_fps"]

        progress_cb(100, "Done.")
        return {
            "fps_source": analysis["fps_source"],
            "duration_sec": duration_sec,
            "commentary": commentary,
            "events": events,
            "kpis": kpis,
        }

    # ---- batch CCNN touch probability (proper implementation) ----
    def _compute_touch_probs_batch(self, nearest, track_positions, ball_positions, diag, total_frames):
        if self.ccnn_model is None or len(nearest) == 0:
            return {f: (1.0 if d <= TOUCH_MAX_DIST_FRAC else 0.0) for f, (tid, d) in nearest.items()}

        pos_by_tid = {tid: {f: (cx, cy) for (f, cx, cy, bw, bh) in pts} for tid, pts in track_positions.items()}
        half = CCNN_WINDOW // 2

        frames = sorted(nearest.keys())
        batch_feats = []
        for f in frames:
            tid, _ = nearest[f]
            window_feats = []
            last_p, last_ball = None, None
            for wf in range(f - half, f - half + CCNN_WINDOW):
                p = pos_by_tid.get(tid, {}).get(wf)
                ball = ball_positions.get(wf)
                if p is None or ball is None:
                    window_feats.append([1.0, 0.0, 0.0])  # far / no motion signal padding
                    continue
                d = _dist(p, ball) / diag
                pv = _dist(p, last_p) / diag if last_p is not None else 0.0
                bv = _dist(ball, last_ball) / diag if last_ball is not None else 0.0
                window_feats.append([d, pv, bv])
                last_p, last_ball = p, ball
            batch_feats.append(window_feats)

        arr = np.array(batch_feats, dtype=np.float32)              # (N, T, 3)
        arr = np.transpose(arr, (0, 2, 1))                          # (N, 3, T)
        tensor = torch.from_numpy(arr).to(self.device)
        probs = self.ccnn_model.predict_touch_prob(tensor)           # (N, T)
        center_probs = probs[:, half].cpu().numpy()

        return {f: float(center_probs[i]) for i, f in enumerate(frames)}

    # ---- rule-based event synthesis on top of real touch signal ----
    def _synthesize_events(self, total_frames, fps, nearest, touch_probs, ball_speeds,
                            jersey_number_of_track, team_of_track, players_tracked):
        events, commentary = [], []
        pass_gap_frames = PASS_MAX_GAP_SEC * fps
        current_possessor, last_touch_frame = None, None
        touches, flagged, passes = 0, 0, 0
        possession_frames = {0: 0, 1: 0}

        nonzero_speeds = ball_speeds[ball_speeds > 0]
        shot_speed_cut = (
            float(np.percentile(nonzero_speeds, SHOT_SPEED_PERCENTILE))
            if len(nonzero_speeds) > 5 else float("inf")
        )
        shots = 0

        def jersey_label(tid):
            info = jersey_number_of_track.get(tid)
            if info:
                return f"#{info['number']}", info["confidence"]
            return f"Track {tid}", 0.0

        for f in range(total_frames):
            t_sec = f / fps if fps else 0.0
            if f in nearest:
                tid, dist_frac = nearest[f]
                prob = touch_probs.get(f, 0.0)
                team = team_of_track.get(tid, 0)
                possession_frames[team] = possession_frames.get(team, 0) + 1

                if prob >= TOUCH_PROB_THRESHOLD:
                    label, conf = jersey_label(tid)
                    touches += 1
                    is_flagged = conf and conf < FLAG_CONF_THRESHOLD
                    if is_flagged:
                        flagged += 1

                    if current_possessor is not None and current_possessor != tid and last_touch_frame is not None \
                            and (f - last_touch_frame) <= pass_gap_frames:
                        prev_label, _ = jersey_label(current_possessor)
                        prev_team = team_of_track.get(current_possessor, 0)
                        if prev_team == team:
                            passes += 1
                            events.append({"t": t_sec, "ev": "PASS", "detail": f"{prev_label} -> {label}"})
                            commentary.append({"t": t_sec, "tag": "pass", "text": f"{prev_label} finds {label} with a pass."})
                        else:
                            events.append({"t": t_sec, "ev": "INTERCEPTION", "detail": f"{label} wins it from {prev_label}"})
                            commentary.append({"t": t_sec, "tag": "flag", "text": f"{label} intercepts possession from {prev_label}."})
                    else:
                        events.append({"t": t_sec, "ev": "TOUCH", "detail": f"{label} touches the ball"})

                    if is_flagged:
                        events.append({"t": t_sec, "ev": "FLAG", "detail": f"Low-confidence jersey read near {label} ({conf:.2f})"})

                    current_possessor = tid
                    last_touch_frame = f

            if ball_speeds[f] >= shot_speed_cut and ball_speeds[f] > 0:
                shots += 1
                label, _ = jersey_label(current_possessor) if current_possessor is not None else ("Unknown", 0)
                events.append({"t": t_sec, "ev": "SHOT", "detail": f"Fast ball strike near {label}"})
                commentary.append({"t": t_sec, "tag": "shot", "text": f"{label} strikes it hard downfield!"})

        total_poss = sum(possession_frames.values()) or 1
        possession_a = round(100 * possession_frames.get(0, 0) / total_poss)
        possession_b = 100 - possession_a

        kpis = {
            "players_tracked": max(players_tracked, len(jersey_number_of_track)),
            "touches": touches,
            "flagged": flagged,
            "passes": passes,
            "shots": shots,
            "possession_a": f"{possession_a}%",
            "possession_b": f"{possession_b}%",
        }
        return events, commentary, kpis

    # ---- render annotated output video ----
    def _render_annotated_video(self, video_path, output_path, track_positions,
                                 jersey_number_of_track, team_of_track, ball_positions,
                                 total_frames, progress_cb):
        # invert track_positions into frame -> [(tid, cx, cy, bw, bh)]
        by_frame = {}
        for tid, pts in track_positions.items():
            for (f, cx, cy, bw, bh) in pts:
                by_frame.setdefault(f, []).append((tid, cx, cy, bw, bh))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Try avc1 first for HTML5 browser compatibility, fallback to mp4v if it fails
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        f = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for tid, cx, cy, bw, bh in by_frame.get(f, []):
                team = team_of_track.get(tid, 0)
                color = TEAM_A_COLOR_BGR if team == 0 else TEAM_B_COLOR_BGR
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                info = jersey_number_of_track.get(tid)
                label = f"#{info['number']}" if info else f"id{tid}"
                # Dark outline for text readability
                cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            ball = ball_positions.get(f)
            if ball is not None:
                bx, by = int(ball[0]), int(ball[1])
                # Clean, glowing ball target indicator instead of a tiny circle
                cv2.circle(frame, (bx, by), 12, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 12, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 3, (0, 165, 255), -1, cv2.LINE_AA)

            writer.write(frame)
            f += 1
            if total_frames and f % 20 == 0:
                pct = 72 + int(27 * f / total_frames)
                progress_cb(min(pct, 99), f"Synthesize: rendering frame {f}/{total_frames}")

        cap.release()
        writer.release()

        # Convert the generated video to H.264 (avc1) using imageio-ffmpeg 
        # so it plays correctly in all modern web browsers.
        try:
            with open("ffmpeg_trace.log", "a") as f_trace:
                f_trace.write(f"Starting FFMPEG block for {output_path}\n")
            import imageio_ffmpeg
            import subprocess
            import time
            time.sleep(1)  # Give Windows time to release the OpenCV file lock
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_output = output_path + ".temp.mp4"
            with open("ffmpeg_trace.log", "a") as f_trace:
                f_trace.write(f"Replacing {output_path} with {temp_output}\n")
            os.replace(output_path, temp_output)
            with open("ffmpeg_trace.log", "a") as f_trace:
                f_trace.write(f"Running FFMPEG on {temp_output}\n")
            res = subprocess.run([
                ffmpeg_exe, "-y", "-i", temp_output, 
                "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path
            ], capture_output=True, text=True)
            with open("ffmpeg_trace.log", "a") as f_trace:
                f_trace.write(f"FFMPEG finished with code {res.returncode}\n")
                if res.returncode != 0:
                    f_trace.write(f"FFMPEG stderr: {res.stderr}\n")
            if os.path.exists(temp_output) and res.returncode == 0:
                os.remove(temp_output)
        except Exception as e:
            import traceback
            with open("ffmpeg_debug.log", "a") as f_debug:
                f_debug.write(f"H.264 conversion skipped or failed: {e}\n")
                f_debug.write(traceback.format_exc() + "\n")
            print(f"H.264 conversion skipped or failed: {e}", flush=True)



