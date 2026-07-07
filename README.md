# HALO — Match Intelligence Console

This is the fully wired-up version of your project: the `index.html`
frontend talking to a real Flask backend (`app.py`) that runs your three
trained checkpoints from `halo_run_package.zip`:

| Frontend label      | File                          | What it actually is |
|----------------------|-------------------------------|----------------------|
| `detection_best.pt`  | `models/detection_best.pt`    | Your trained Ultralytics YOLO (from `yolo/halo_yolo/weights/best.pt`) — detects `football` (class 0) and `player` (class 1), also used for tracking (BoT-SORT) |
| `jersey_ocr_best.pt` | `models/jersey_ocr_best.pt`   | Your trained jersey-number CNN (from `checkpoints/jersey_cnn/best.pt`) — a Spatial-Transformer + two-digit-head classifier |
| `ccnn_best.pt`       | `models/ccnn_best.pt`         | Your trained temporal filter (from `checkpoints/ccnn_filter/best.pt`) — a 4-block residual Conv1d net used here as a touch/possession probability filter |
| —                    | `models/yolov8n.pt`, `models/yolo11n.pt` | The generic pretrained COCO weights you uploaded, reconstructed back into proper `.pt` files (kept as spares/base weights — not required for the app to run) |

## 1. Why this needed real work, not just "put the files together"

Your uploads were three separate, disconnected things:
- Two **exploded** pretrained-weight zips (someone had unzipped `yolov8n.pt`/`yolo11n.pt` — `.pt` files are themselves zip archives — so I re-zipped them back into loadable `.pt` files).
- A **training-run dump** (`halo_run_package.zip`) containing checkpoints but **no model class definitions or inference/serving code**.
- A **frontend-only** `index.html` that expects a Flask API at `http://127.0.0.1:5000` with specific endpoints (`/inputs`, `/upload`, `/process`, `/progress/<file>`, `/result/<file>`) that didn't exist yet.

None of these would have "just worked" together. To integrate them for real I:
1. Opened each checkpoint's internal pickle stream (without needing PyTorch) to read out the **exact layer names and tensor shapes**, and reconstructed matching `nn.Module` classes in `models.py` so `load_state_dict(..., strict=True)` succeeds instead of guessing an architecture that would error out or silently load garbage.
2. Wrote `pipeline.py`, a full detection → tracking → jersey OCR → team clustering → touch/possession filtering → event/commentary synthesis → annotated-video-export pipeline that calls your three real models.
3. Wrote `app.py`, a Flask server implementing every endpoint `index.html` calls, with the exact JSON shapes it expects, plus CORS so the static HTML file can talk to it from a different origin.

## 2. Setup

This sandbox has no internet access and no GPU/PyTorch installed, so I
could not execute the full model inference myself — but the code has been
carefully reverse-engineered against the actual checkpoint contents and
reviewed line-by-line. Run it on your own machine (or Kaggle/Colab) like this:

```bash
cd halo_jerseyiq
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python app.py
```

Then open `static/index.html` in your browser (double-click it, or serve it
however you like — it doesn't need to be on the same origin/port as the
backend; the page has a "Backend" field already pointed at
`http://127.0.0.1:5000`).

- **Upload & Run**: pick any video, it uploads and processes immediately.
- **Process existing input video**: `sample_input.mp4` (your own demo clip)
  is already in `inputs/` and will show up in the dropdown.

## 3. Honest caveats — please read before trusting the output

- **Jersey OCR & detection are your real trained weights**, loaded with a
  reconstructed architecture that matches the checkpoint shapes exactly. I'm
  confident in this part.
- **The CCNN touch/possession filter is your real trained weights too**, but
  the *input feature definition* (what 3 numbers per frame it expects) was
  not included anywhere in the uploaded package — there was no training
  script for it. I implemented a principled feature (ball distance, player
  speed, ball speed, all normalized) in `pipeline.py`'s
  `_compute_touch_probs_batch()`. **If your original training used a
  different feature definition, this is the one place you need to edit** —
  the model class itself will keep loading correctly either way.
- **Pass / shot / interception detection are rule-based heuristics** built
  on top of the real touch signal (there was no dedicated pass/shot model in
  the package). They're clearly marked in `pipeline.py` and are easy to
  retune (see the `CONFIG` constants at the top of that file) or replace
  with a real model later.
- **Team assignment** is a 2-means clustering of average jersey-crop color
  per track — simple and effective for two solid-color kits, but it doesn't
  know jersey colors ahead of time.
- The pipeline processes the video in two conceptual passes (analyze, then
  render) so possession/touch stats can look slightly into the future
  relative to a pure real-time system — this trades a bit of latency for
  much better accuracy, which is the right call for an offline "upload a
  clip" tool like this one.

## 4. File layout

```
halo_jerseyiq/
  app.py            Flask server (matches index.html's expected API exactly)
  pipeline.py        Detection/tracking/OCR/possession/event pipeline
  models.py          Reconstructed PyTorch architectures for the two CNN checkpoints
  requirements.txt
  static/index.html  Your original frontend, unmodified
  models/            All five weight files (3 trained + 2 reconstructed pretrained)
  inputs/            Server-side videos (sample_input.mp4 included)
  uploads/           User-uploaded videos land here
  outputs/           Annotated output videos are written here and served at /outputs/<file>
```

## 5. If something doesn't load

If `load_state_dict` ever raises a shape/key mismatch, it means your actual
training code used a slightly different architecture than what I
reverse-engineered from the shapes (this can happen with e.g. an extra
Dropout with different placement that still wouldn't show up in a state
dict, but a different channel count or an extra layer would). The error
message will tell you exactly which key/shape doesn't match — adjust the
corresponding line in `models.py`.
