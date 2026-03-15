#!/usr/bin/env python3
"""
Dash Cam Viewer YOLO v1 - Roadside Post Detection Sample Builder
Runs YOLO on frames to draw bounding boxes around roadside posts.
User navigates frames, detects posts per-frame, and acquires boxes as training samples.
"""

import sys
import os
import csv
import json
import queue as _queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import shutil
import cv2
import math
import re
import subprocess
import statistics
from datetime import timedelta, datetime
from pathlib import Path
from collections import defaultdict
from io import BytesIO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QMessageBox,
    QDesktopWidget, QSplitter, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QFileDialog, QTextEdit, QStackedWidget,
    QInputDialog, QFormLayout, QGroupBox, QCheckBox, QColorDialog, QRubberBand,
    QScrollArea, QGridLayout, QFrame
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import pyqtSlot, QObject, Qt, QTimer, QThread, pyqtSignal, QUrl, QRect, QPoint, QSize
from PyQt5.QtGui import QPixmap, QFont, QColor, QPainter, QPen

import tempfile

import folium

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECTS_DIR = str(Path(__file__).parent / "dash_cam_projects")
GLOBAL_SETTINGS_PATH    = os.path.join(PROJECTS_DIR, "settings.json")
POST_SETTINGS_PATH      = os.path.join(PROJECTS_DIR, "post_settings.json")
CULVERT_SETTINGS_PATH   = os.path.join(PROJECTS_DIR, "culvert_settings.json")
POST_DATASET_DIR    = os.path.join(PROJECTS_DIR, "post_dataset")
POST_MODEL_DIR      = os.path.join(PROJECTS_DIR, "post_model")
POST_MODEL_PATH     = os.path.join(POST_MODEL_DIR, "weights", "best.pt")
FPS_FALLBACK = 30.0


# ─── Project Helpers ──────────────────────────────────────────────────────────

def load_global_settings():
    if os.path.exists(GLOBAL_SETTINGS_PATH):
        with open(GLOBAL_SETTINGS_PATH) as f:
            return json.load(f)
    return {"exiftool_path": ""}


def save_global_settings(settings):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(GLOBAL_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def load_post_settings():
    if os.path.exists(POST_SETTINGS_PATH):
        with open(POST_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def save_post_settings(settings):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(POST_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def list_projects():
    if not os.path.exists(PROJECTS_DIR):
        return []
    projects = []
    for name in sorted(os.listdir(PROJECTS_DIR)):
        cfg = os.path.join(PROJECTS_DIR, name, "project.json")
        if os.path.isfile(cfg):
            projects.append(name)
    return projects


def load_project(name):
    cfg = os.path.join(PROJECTS_DIR, name, "project.json")
    with open(cfg) as f:
        return json.load(f)


def save_project(name, data):
    folder = os.path.join(PROJECTS_DIR, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "project.json"), "w") as f:
        json.dump(data, f, indent=2)


def project_frames_dir(name):
    return os.path.join(PROJECTS_DIR, name, "frames")


def project_txt_path(name):
    return os.path.join(PROJECTS_DIR, name, "frames_latlon.txt")


def project_posts_csv(name):
    return os.path.join(PROJECTS_DIR, name, "posts.csv")


def project_culverts_csv(name):
    return os.path.join(PROJECTS_DIR, name, "culverts.csv")


def project_false_detections_csv(name):
    return os.path.join(PROJECTS_DIR, name, "false_detections.csv")


def load_culvert_settings():
    if os.path.exists(CULVERT_SETTINGS_PATH):
        with open(CULVERT_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def save_culvert_settings(settings):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(CULVERT_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


# ─── Survey Coordinate Conversion ─────────────────────────────────────────────

def utm_to_latlon(easting, northing, zone, southern=True):
    """Convert MGA/UTM easting+northing to WGS84 lat/lon (decimal degrees)."""
    k0 = 0.9996
    a = 6378137.0
    b = 6356752.3142
    e2 = 1.0 - (b / a) ** 2
    e_prime2 = e2 / (1.0 - e2)

    x = easting - 500000.0
    y = northing - (10000000.0 if southern else 0.0)

    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = mu
    phi1 += e1 * (3 / 2 - 27 * e1 ** 2 / 32) * math.sin(2 * mu)
    phi1 += e1 ** 2 * (21 / 16 - 55 * e1 ** 2 / 32) * math.sin(4 * mu)
    phi1 += e1 ** 3 * (151 / 96) * math.sin(6 * mu)
    phi1 += e1 ** 4 * (1097 / 512) * math.sin(8 * mu)

    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    T1 = math.tan(phi1) ** 2
    C1 = e_prime2 * math.cos(phi1) ** 2
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = x / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D ** 2 / 2
        - D ** 4 / 24 * (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * e_prime2)
        + D ** 6 / 720 * (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * e_prime2 - 3 * C1 ** 2)
    )
    lon = lon0 + (
        D
        - D ** 3 / 6 * (1 + 2 * T1 + C1)
        + D ** 5 / 120 * (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * e_prime2 + 24 * T1 ** 2)
    ) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon)


def load_preplot_csv(csv_path, zone):
    """Read pre-plot CSV with easting/northing columns and convert to lat/lon.
    Returns list of (lat, lon, station)."""
    points = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                easting = float(row['easting'])
                northing = float(row['northing'])
                station = row.get('station', '').strip('"').strip()
                lat, lon = utm_to_latlon(easting, northing, zone, southern=True)
                points.append((lat, lon, station))
            except (ValueError, KeyError):
                pass
    return points


# ─── Processing Worker (runs in background thread) ────────────────────────────

class ProcessWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, video_dir, output_dir, txt_path, interval, jpeg_quality, exiftool_path):
        super().__init__()
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.txt_path = txt_path
        self.interval = interval
        self.jpeg_quality = jpeg_quality
        self.exiftool_path = exiftool_path
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            # Always start fresh
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

            mp4_files = []
            for root, dirs, files in os.walk(self.video_dir):
                for file in files:
                    if file.lower().endswith('.mp4'):
                        mp4_files.append(os.path.join(root, file))

            self.log.emit(f"Found {len(mp4_files)} MP4 file(s) in {self.video_dir}")

            if not mp4_files:
                self.log.emit("No MP4 files found.")
                self.finished.emit(False)
                return

            jobs = []
            for mp4 in mp4_files:
                rel = os.path.relpath(os.path.dirname(mp4), self.video_dir).replace('\\', '_').replace('/', '_')
                base = os.path.splitext(os.path.basename(mp4))[0]
                prefix = f"{rel}_{base}" if rel != '.' else base
                jobs.append((mp4, prefix))

            num_workers = min(len(jobs), os.cpu_count() or 4)
            self.log.emit(f"Extracting with {num_workers} parallel worker(s)...")

            # Prevent OpenCV's internal thread pool from conflicting across concurrent captures
            cv2.setNumThreads(1)

            log_queue = _queue.Queue()
            all_entries = []
            total_saved = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_map = {executor.submit(self._extract_video, mp4, prefix, log_queue): mp4
                              for mp4, prefix in jobs}
                pending = set(future_map)

                while pending:
                    # Drain log queue — real-time output, no signal from worker threads
                    while True:
                        try:
                            self.log.emit(log_queue.get_nowait())
                        except _queue.Empty:
                            break

                    if self._abort:
                        self.log.emit("Aborted by user.")
                        self.finished.emit(False)
                        return

                    done, pending = concurrent.futures.wait(pending, timeout=0.1)
                    for future in done:
                        try:
                            count, entries = future.result()
                            total_saved += count
                            all_entries.extend(entries)
                        except Exception as e:
                            self.log.emit(f"  Worker error: {e}")
                # Final log drain
                while True:
                    try:
                        self.log.emit(log_queue.get_nowait())
                    except _queue.Empty:
                        break

            # Write txt file from QThread — no thread conflicts
            with open(self.txt_path, "w") as f:
                f.write("Filename\tLatitude\tLongitude\n")
                for filename, lat, lon in all_entries:
                    f.write(f"{filename}\t{lat}\t{lon}\n")

            self.log.emit(f"\nDone. Total frames saved: {total_saved}")
            self.finished.emit(True)

        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.finished.emit(False)

    @staticmethod
    def _is_valid_coord(lat, lon):
        return -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == -180 or lon == -180)

    def _extract_gps(self, video_path):
        cmd = [self.exiftool_path, "-m", "-ee", "-n", "-p",
               "${SampleTime} ${GPSLatitude} ${GPSLongitude}", video_path]
        try:
            result = subprocess.check_output(cmd, text=True, input='\n', stderr=subprocess.DEVNULL)
            gps_data = []
            for line in result.splitlines():
                line = line.strip()
                if line:
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 3:
                        try:
                            lat = float(parts[-2])
                            lon = float(parts[-1])
                            if self._is_valid_coord(lat, lon):
                                gps_data.append((float(parts[0]), parts[-2], parts[-1]))
                        except ValueError:
                            pass
            return gps_data
        except Exception:
            return []

    def _get_lat_lon(self, gps_data, target_sec):
        if not gps_data:
            return "", ""
        gps_data.sort(key=lambda x: x[0])
        if target_sec <= gps_data[0][0]:
            return gps_data[0][1], gps_data[0][2]
        if target_sec >= gps_data[-1][0]:
            if len(gps_data) < 2:
                return gps_data[-1][1], gps_data[-1][2]
            t2, lat2, lon2 = gps_data[-1]
            t1, lat1, lon1 = gps_data[-2]
            dt = t2 - t1
            if dt == 0:
                return lat2, lon2
            extrat = target_sec - t2
            vlat = (float(lat2) - float(lat1)) / dt
            vlon = (float(lon2) - float(lon1)) / dt
            return str(float(lat2) + vlat * extrat), str(float(lon2) + vlon * extrat)
        for i in range(len(gps_data) - 1):
            t1, lat1, lon1 = gps_data[i]
            t2, lat2, lon2 = gps_data[i + 1]
            if t1 <= target_sec <= t2:
                dt = t2 - t1
                if dt == 0:
                    return lat1, lon1
                frac = (target_sec - t1) / dt
                return (str(float(lat1) + frac * (float(lat2) - float(lat1))),
                        str(float(lon1) + frac * (float(lon2) - float(lon1))))
        return "", ""

    def _extract_video(self, video_path, prefix, log_queue):
        """Extract frames from one video file. Thread-safe — no Qt signal emissions.
        Returns (saved_count, entries) where entries = [(filename, lat, lon), ...]"""
        def log(msg):
            log_queue.put(msg)

        log(f"\nProcessing: {os.path.basename(video_path)}")
        gps_data = self._extract_gps(video_path)
        log(f"  GPS points: {len(gps_data)}")

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log("  Cannot open video.")
            return 0, []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not math.isfinite(fps):
            fps = FPS_FALLBACK
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        log(f"  FPS: {fps:.1f} | Frames: {total_frames} | Duration: {timedelta(seconds=int(duration_sec))}")

        saved_count = 0
        frame_idx = 0
        last_saved_sec = -1
        entries = []

        while True:
            if self._abort:
                break
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            bucket = math.floor(current_sec / self.interval) * self.interval

            if bucket > last_saved_sec:
                time_str = str(timedelta(seconds=int(bucket))).replace(':', '_')
                filename = f"{prefix}_frame_{frame_idx:06d}_sec_{time_str}.jpg"
                full_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                saved_count += 1
                last_saved_sec = bucket

                lat, lon = self._get_lat_lon(gps_data, current_sec)
                entries.append((filename, lat, lon))
                log(f"  Saved: {filename}  Lat: {lat}  Lon: {lon}")

            frame_idx += 1

        cap.release()
        log(f"  Finished. {saved_count} frames saved.")
        return saved_count, entries




# ─── YOLO subprocess scripts (run in a separate Python process to avoid
#     PyTorch/Qt thread conflict on Windows) ────────────────────────────────────

_TRAIN_SCRIPT = r"""
import sys, os, csv, json, shutil, struct

params_file = sys.argv[1]
with open(params_file) as f:
    p = json.load(f)
sources    = p['sources']       # list of {"csv": "...", "frames_dir": "..."}
dataset_dir = p['dataset_dir']
model_dir   = p['model_dir']
base_model  = p['base_model']
epochs      = p['epochs']

# ── prepare dataset ──────────────────────────────────────────────────────────
def jpeg_size(path):
    try:
        with open(path, 'rb') as f:
            f.read(2)
            while True:
                b = f.read(1)
                while b != b'\xff': b = f.read(1)
                while b == b'\xff': b = f.read(1)
                mk = ord(b)
                if mk in (0xC0, 0xC1, 0xC2):
                    f.read(3)
                    h, w = struct.unpack('>HH', f.read(4))
                    return w, h
                elif mk in (0xD8, 0xD9): break
                else:
                    sz = struct.unpack('>H', f.read(2))[0]
                    f.seek(sz - 2, 1)
    except Exception: pass
    return None, None

# Collect valid boxes from all project sources
# valid entries: {'src_path': full path to frame, 'frame': filename, 'bx/by/bw/bh': ints}
valid = []

for src in sources:
    csv_path   = src['csv']
    frames_dir = src['frames_dir']
    if not os.path.exists(csv_path):
        print(f"Skipping (no posts.csv): {frames_dir}", flush=True)
        continue
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        try:
            fname = r['source_frame']
            full  = os.path.join(frames_dir, fname)
            if not os.path.exists(full):
                continue
            valid.append({'src_path': full, 'frame': fname,
                          'bx': int(r['box_x']), 'by': int(r['box_y']),
                          'bw': int(r['box_w']), 'bh': int(r['box_h'])})
        except (KeyError, ValueError):
            pass
    pass  # neg_candidates built below via sources

if not valid:
    print("ERROR: No valid bounding boxes found across any project", flush=True)
    sys.exit(1)

print(f"Positive samples: {len(valid)} boxes across {len(sources)} project(s)", flush=True)

# Build negative set: frames adjacent to positive frames (per project)
positive_paths = {v['src_path'] for v in valid}
neg_paths = set()
hard_neg_paths = set()
for src in sources:
    frames_dir = src['frames_dir']
    false_frames = src.get('false_detections', [])
    all_frames_list = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg'))
    frame_index = {f: i for i, f in enumerate(all_frames_list)}
    pos_in_proj = {v['frame'] for v in valid if v['src_path'].startswith(frames_dir)}
    for fname in pos_in_proj:
        idx = frame_index.get(fname)
        if idx is None: continue
        for offset in (-2, 2):
            ni = idx + offset
            if 0 <= ni < len(all_frames_list):
                cand = os.path.join(frames_dir, all_frames_list[ni])
                if cand not in positive_paths:
                    neg_paths.add(cand)
    for fname in false_frames:
        full = os.path.join(frames_dir, fname)
        if os.path.exists(full) and full not in positive_paths:
            hard_neg_paths.add(full)

print(f"Negative samples: {len(neg_paths)} adjacent, {len(hard_neg_paths)} hard negatives (false detections)", flush=True)

img_train = os.path.join(dataset_dir, 'images', 'train')
lbl_train = os.path.join(dataset_dir, 'labels', 'train')
os.makedirs(img_train, exist_ok=True)
os.makedirs(lbl_train, exist_ok=True)

# Group positive boxes by source frame path
grouped = {}
for v in valid:
    grouped.setdefault(v['src_path'], []).append(v)

for src_path, boxes in grouped.items():
    fname = os.path.basename(src_path)
    iw, ih = jpeg_size(src_path)
    if iw is None: continue
    shutil.copy2(src_path, os.path.join(img_train, fname))
    with open(os.path.join(lbl_train, os.path.splitext(fname)[0] + '.txt'), 'w') as lf:
        for b in boxes:
            cx = (b['bx'] + b['bw'] / 2) / iw
            cy = (b['by'] + b['bh'] / 2) / ih
            nw = b['bw'] / iw
            nh = b['bh'] / ih
            lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

for src_path in neg_paths | hard_neg_paths:
    fname = os.path.basename(src_path)
    shutil.copy2(src_path, os.path.join(img_train, fname))
    open(os.path.join(lbl_train, os.path.splitext(fname)[0] + '.txt'), 'w').close()

yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
with open(yaml_path, 'w') as yf:
    yf.write(f"path: {dataset_dir}\n")
    yf.write("train: images/train\n")
    yf.write("val: images/train\n")
    yf.write("nc: 1\n")
    yf.write("names: ['post']\n")

print(f"Dataset ready: {len(grouped)} positive frames, {len(neg_paths)} negative frames", flush=True)

# ── train ─────────────────────────────────────────────────────────────────────
from ultralytics import YOLO
model = YOLO(base_model)
model.train(data=yaml_path, epochs=epochs, imgsz=640,
            project=model_dir, name='.', exist_ok=True, verbose=True,
            workers=0, cache=False, device=0)
print("TRAINING_DONE", flush=True)
"""

_DETECT_SCRIPT = r"""
import sys, os, json, csv, cv2, datetime
frames_dir, model_path, conf, posts_dir, csv_path, params_file = \
    sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
with open(params_file) as _f:
    _p = json.load(_f)
coord_map = _p['coord_map']
already   = set(_p['already'])
os.makedirs(posts_dir, exist_ok=True)
from ultralytics import YOLO
model  = YOLO(model_path)
frames = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg'))
todo = [f for f in frames if f not in already]
total = len(todo)
new_rows = []
for idx, fname in enumerate(todo):
    pct = int((idx + 1) / total * 100) if total else 100
    print(f"PROGRESS:{pct}:{idx+1}/{total} frames — {len(new_rows)} posts found", flush=True)
    fpath = os.path.join(frames_dir, fname)
    results = model(fpath, conf=conf, verbose=False, stream=True, half=True, device=0)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        img = cv2.imread(fpath)
        ih, iw = img.shape[:2]
        lat, lon = coord_map.get(fname, [0.0, 0.0])
        for box in r.boxes:
            x1,y1,x2,y2 = (int(v) for v in box.xyxy[0].tolist())
            bx,by = max(0,x1), max(0,y1)
            bw,bh = min(x2,iw)-bx, min(y2,ih)-by
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_fn = f"post_yolo_{ts}.jpg"
            crop = img[by:by+bh, bx:bx+bw]
            cv2.imwrite(os.path.join(posts_dir, img_fn), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            new_rows.append({"name":"post","latitude":lat,"longitude":lon,
                              "source_frame":fname,"saved_image":img_fn,
                              "box_x":bx,"box_y":by,"box_w":bw,"box_h":bh})
write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=["name","latitude","longitude","source_frame",
                                       "saved_image","box_x","box_y","box_w","box_h"])
    if write_header:
        w.writeheader()
    w.writerows(new_rows)
print("DETECTIONS:" + json.dumps(new_rows), flush=True)
"""

_CULVERT_SCRIPT = r"""
import sys, os, json, itertools

params_file = sys.argv[1]
with open(params_file) as f:
    p = json.load(f)

frames_dir = p['frames_dir']
model_path = p['model_path']
conf       = float(p['conf'])
pixel_sep  = int(p['pixel_sep'])
tolerance  = float(p.get('tolerance', 0.30))
points     = p['points']   # list of [lat, lon, fname]

fname_to_latlon = {item[2]: (item[0], item[1]) for item in points}
frames = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg'))

print(f"Scanning {len(frames)} frames for culverts (sep={pixel_sep}px ±{int(tolerance*100)}%)...", flush=True)

from ultralytics import YOLO
model = YOLO(model_path)

lo = pixel_sep * (1 - tolerance)
hi = pixel_sep * (1 + tolerance)
found = 0

for i, fname in enumerate(frames):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(frames)} scanned...", flush=True)
    fpath = os.path.join(frames_dir, fname)
    results = model(fpath, conf=conf, verbose=False, stream=True, half=True, device=0)
    boxes = []
    for r in results:
        if r.boxes:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                boxes.append((cx, cy))
    if len(boxes) < 2:
        continue
    for b1, b2 in itertools.combinations(boxes, 2):
        sep = abs(b1[0] - b2[0])
        if lo <= sep <= hi:
            lat, lon = fname_to_latlon.get(fname, (0.0, 0.0))
            print("CULVERT:" + json.dumps({
                "frame": fname, "lat": lat, "lon": lon, "sep": round(sep, 1)
            }), flush=True)
            found += 1
            break

print(f"Found {found} culvert candidate(s).", flush=True)
"""

_VEHICLE_SCRIPT = r"""
import sys, os, json, math

params_file = sys.argv[1]
with open(params_file) as f:
    p = json.load(f)

frames_dir     = p['frames_dir']
model_path     = p['model_path']
conf           = float(p['conf'])
frame_interval = float(p['frame_interval'])
segment_km     = float(p['segment_km'])
points         = p['points']   # [[lat, lon, fname], ...] sorted by filename

VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck (COCO)

# ── Haversine distance (km) ───────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# ── Assign frames to 1km segments ────────────────────────────────────────────
cum = 0.0
distances = [0.0]
for i in range(1, len(points)):
    cum += haversine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
    distances.append(cum)

total_km = distances[-1]
frame_segs = [int(d / segment_km) for d in distances]
n_segs = max(frame_segs) + 1 if frame_segs else 1

# Collect GPS path points per segment (subsampled for map display)
seg_paths = {}
for i, (lat, lon, fname) in enumerate(points):
    seg = frame_segs[i]
    seg_paths.setdefault(seg, []).append([lat, lon])

print(f"Route: {total_km:.1f}km -> {n_segs} segment(s) of {segment_km}km", flush=True)

# ── YOLO vehicle detection + centroid tracker ─────────────────────────────────
from ultralytics import YOLO
model = YOLO(model_path)

MAX_GAP     = 3      # frames a track can vanish before being closed
DIST_THRESH = 0.15   # max normalised centroid move between frames
GROW_THRESH = 1.2    # area must grow by this factor to count as oncoming

active_tracks = []   # {cx, cy, area_first, area_last, first_frame, last_frame, segs}
closed_counts = {}   # seg_id -> oncoming vehicle count
next_id = 0
total = len(points)

def close_stale(tracks, frame_idx, force=False):
    keep, closed = [], []
    for t in tracks:
        if force or (frame_idx - t['last_frame']) > MAX_GAP:
            closed.append(t)
        else:
            keep.append(t)
    return keep, closed

def record_closed(closed, counts):
    for t in closed:
        # Oncoming: area grew (vehicle approached camera)
        if t['area_last'] >= t['area_first'] * GROW_THRESH:
            for seg in t['segs']:
                counts[seg] = counts.get(seg, 0) + 1

for idx, (lat, lon, fname) in enumerate(points):
    pct = int((idx + 1) / total * 100)
    if (idx + 1) % 25 == 0 or idx == total - 1:
        active_v = sum(closed_counts.values())
        print(f"PROGRESS:{pct}:{idx+1}/{total} frames | {len(active_tracks)} active tracks | {active_v} counted", flush=True)

    fpath = os.path.join(frames_dir, fname)
    if not os.path.exists(fpath):
        if idx < 5:
            print(f"DBG frame missing: {fpath}", flush=True)
        continue

    seg = frame_segs[idx]
    results = model(fpath, conf=conf, verbose=False, stream=True, half=True, device=0)
    detections = []
    img_w = img_h = None
    dbg = idx < 10  # verbose on first 10 frames

    all_boxes_in_frame = []
    for r in results:
        if img_w is None and r.orig_shape:
            img_h, img_w = r.orig_shape[:2]
            if dbg:
                print(f"DBG frame {idx} ({fname}): size={img_w}x{img_h} conf={conf}", flush=True)
        if r.boxes:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                all_boxes_in_frame.append((cls, conf_score, cx, cy))
                if cls not in VEHICLE_CLASSES:
                    if dbg:
                        print(f"  skip cls={cls} conf={conf_score:.2f}", flush=True)
                    continue
                # Oncoming candidate: centre 60% of width, upper 75% of height
                if img_w and not (0.2 * img_w < cx < 0.8 * img_w):
                    if dbg:
                        print(f"  skip cls={cls} cx={cx:.0f} outside centre (img_w={img_w})", flush=True)
                    continue
                if img_h and cy > 0.75 * img_h:
                    if dbg:
                        print(f"  skip cls={cls} cy={cy:.0f} too low (img_h={img_h})", flush=True)
                    continue
                if dbg:
                    print(f"  PASS cls={cls} conf={conf_score:.2f} cx={cx:.0f} cy={cy:.0f} area={area:.0f}", flush=True)
                detections.append((cx, cy, area))
    if dbg and not all_boxes_in_frame:
        print(f"  no boxes at all in frame {idx}", flush=True)

    # Close stale tracks
    active_tracks, stale = close_stale(active_tracks, idx)
    record_closed(stale, closed_counts)

    # Match detections to active tracks
    matched = set()
    for cx, cy, area in detections:
        best, best_d = None, float('inf')
        for t in active_tracks:
            if id(t) in matched:
                continue
            dx = (cx - t['cx']) / (img_w or 1920)
            dy = (cy - t['cy']) / (img_h or 1080)
            d  = math.sqrt(dx*dx + dy*dy)
            if d < best_d and d < DIST_THRESH:
                best_d, best = d, t
        if best:
            best['cx'] = cx
            best['cy'] = cy
            best['area_last'] = area
            best['last_frame'] = idx
            best['segs'].add(seg)
            matched.add(id(best))
        else:
            active_tracks.append({'cx': cx, 'cy': cy,
                                   'area_first': area, 'area_last': area,
                                   'first_frame': idx, 'last_frame': idx,
                                   'segs': {seg}})

# Close remaining tracks
_, remaining = close_stale(active_tracks, total, force=True)
record_closed(remaining, closed_counts)

# ── Build output ──────────────────────────────────────────────────────────────
max_count = max(closed_counts.values()) if closed_counts else 1
segments_out = []
for seg_id in range(n_segs):
    path = seg_paths.get(seg_id, [])
    if not path:
        continue
    # Subsample path to at most 20 points
    step = max(1, len(path) // 20)
    segments_out.append({
        'seg_id':   seg_id,
        'count':    closed_counts.get(seg_id, 0),
        'start_km': round(seg_id * segment_km, 2),
        'path':     path[::step],
    })

total_vehicles = sum(closed_counts.values())
print(f"Tracks closed: {next_id} total | {total_vehicles} passed oncoming filter (grew >{GROW_THRESH}x)", flush=True)
print(f"SEGMENTS:" + json.dumps({'segments': segments_out, 'max_count': max_count}), flush=True)
print(f"Done. {total_vehicles} oncoming vehicle(s) across {len(closed_counts)} segment(s).", flush=True)
"""

# ─── YOLO Workers ─────────────────────────────────────────────────────────────

class YoloTrainWorker(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, project_name):
        super().__init__()
        self.project_name = project_name

    def run(self):
        try:
            import subprocess, sys, json, tempfile
            # Collect posts.csv from every project that has one
            sources = []
            for proj in list_projects():
                csv_path   = os.path.join(PROJECTS_DIR, proj, "posts.csv")
                frames_dir = project_frames_dir(proj)
                if os.path.exists(csv_path) and os.path.isdir(frames_dir):
                    fd_csv = project_false_detections_csv(proj)
                    false_frames = []
                    if os.path.exists(fd_csv):
                        with open(fd_csv, newline='') as ff:
                            false_frames = [r['source_frame'] for r in csv.DictReader(ff)]
                    sources.append({"csv": csv_path, "frames_dir": frames_dir,
                                    "false_detections": false_frames})
                    self.log.emit(f"  Found posts.csv in project: {proj}"
                                  + (f" ({len(false_frames)} false detections)" if false_frames else ""))

            if not sources:
                self.log.emit("No posts.csv found in any project. Box some posts first.")
                self.finished.emit(False)
                return

            base   = POST_MODEL_PATH if os.path.exists(POST_MODEL_PATH) else 'yolov8n.pt'
            epochs = 50
            params = {
                "sources":     sources,
                "dataset_dir": POST_DATASET_DIR,
                "model_dir":   POST_MODEL_DIR,
                "base_model":  base,
                "epochs":      epochs,
            }
            params_file = os.path.join(tempfile.gettempdir(), 'dcv_train_params.json')
            with open(params_file, 'w') as f:
                json.dump(params, f)

            self.log.emit(f"Training across {len(sources)} project(s), base: {os.path.basename(base)}, epochs: {epochs}")
            script_file = os.path.join(tempfile.gettempdir(), 'dcv_train_script.py')
            with open(script_file, 'w', encoding='utf-8') as sf:
                sf.write(_TRAIN_SCRIPT)
            proc = subprocess.Popen(
                [sys.executable, script_file, params_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='replace',
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.log.emit(line)
            proc.wait()
            if proc.returncode == 0:
                self.log.emit(f"Training complete. Model: {POST_MODEL_PATH}")
                self.finished.emit(True)
            else:
                self.log.emit(f"Training subprocess exited with code {proc.returncode}")
                self.finished.emit(False)
        except Exception as e:
            self.log.emit(f"Training error: {e}")
            self.finished.emit(False)



class YoloDetectWorker(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, project_name, confidence, points_with_files):
        super().__init__()
        self.project_name      = project_name
        self.confidence        = confidence
        self.points_with_files = points_with_files
        self.new_rows          = []

    def run(self):
        if not os.path.exists(POST_MODEL_PATH):
            self.log.emit("No trained model found. Run Train first.")
            self.finished.emit(False)
            return

        frames_dir   = project_frames_dir(self.project_name)
        csv_path     = project_posts_csv(self.project_name)
        posts_dir = os.path.join(PROJECTS_DIR, self.project_name, "posts")

        already = set()
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    already.add(row.get('source_frame', ''))

        coord_map = {fname: [lat, lon] for lat, lon, fname in self.points_with_files}

        import subprocess, sys, tempfile
        params_file = os.path.join(tempfile.gettempdir(), 'dcv_detect_params.json')
        with open(params_file, 'w') as f:
            json.dump({'coord_map': coord_map, 'already': list(already)}, f)

        self.log.emit(f"Launching detection subprocess (conf={self.confidence:.2f})...")
        proc = subprocess.Popen(
            [sys.executable, '-c', _DETECT_SCRIPT,
             frames_dir, POST_MODEL_PATH, str(self.confidence),
             posts_dir, csv_path, params_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("DETECTIONS:"):
                try:
                    self.new_rows = json.loads(line[11:])
                except Exception:
                    pass
            elif line.startswith("PROGRESS:"):
                # Format: PROGRESS:pct:message
                parts = line.split(":", 2)
                msg = parts[2] if len(parts) == 3 else line[9:]
                pct = parts[1] if len(parts) >= 2 else "?"
                self.log.emit(f"Detecting posts... {pct}%  ({msg})")
            elif line:
                self.log.emit(line)
        proc.wait()
        self.log.emit(f"Detection complete. {len(self.new_rows)} new post(s) found.")
        self.finished.emit(proc.returncode == 0)


class VehicleCountWorker(QThread):
    log      = pyqtSignal(str)
    done     = pyqtSignal(dict)   # {'segments': [...], 'max_count': int}
    finished = pyqtSignal(bool)

    def __init__(self, project_name, points_with_files, conf, frame_interval, segment_km):
        super().__init__()
        self.project_name      = project_name
        self.points_with_files = points_with_files
        self.conf              = conf
        self.frame_interval    = frame_interval
        self.segment_km        = segment_km

    def run(self):
        import subprocess, sys, json, tempfile
        base_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt')
        if not os.path.exists(base_model):
            self.log.emit("yolov8n.pt not found in project folder. Place it alongside this script.")
            self.finished.emit(False)
            return

        frames_dir = project_frames_dir(self.project_name)
        params = {
            'frames_dir':     frames_dir,
            'model_path':     base_model,
            'conf':           self.conf,
            'frame_interval': self.frame_interval,
            'segment_km':     self.segment_km,
            'points':         [[lat, lon, fname] for lat, lon, fname in self.points_with_files],
        }
        params_file = os.path.join(tempfile.gettempdir(), 'dcv_vehicle_params.json')
        with open(params_file, 'w') as f:
            json.dump(params, f)

        script_file = os.path.join(tempfile.gettempdir(), 'dcv_vehicle_script.py')
        with open(script_file, 'w', encoding='utf-8') as sf:
            sf.write(_VEHICLE_SCRIPT)

        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dcv_vehicle_log.txt')
        self.log.emit(f"Vehicle log: {log_file}")

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        proc = subprocess.Popen(
            [sys.executable, script_file, params_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
            env=env,
        )
        result = {}
        with open(log_file, 'w', encoding='utf-8') as lf:
            for line in proc.stdout:
                line = line.rstrip()
                lf.write(line + '\n')
                lf.flush()
                if not line:
                    continue
                if line.startswith('SEGMENTS:'):
                    try:
                        result = json.loads(line[9:])
                    except Exception:
                        pass
                elif line.startswith('PROGRESS:'):
                    parts = line.split(':', 2)
                    pct = parts[1] if len(parts) >= 2 else '?'
                    msg = parts[2] if len(parts) == 3 else ''
                    self.log.emit(f"Counting vehicles... {pct}%  ({msg})")
                else:
                    self.log.emit(line)
        proc.wait()
        self.done.emit(result)
        self.finished.emit(proc.returncode == 0)


class CulvertFindWorker(QThread):
    log      = pyqtSignal(str)
    found    = pyqtSignal(list)   # list of {"frame", "lat", "lon", "sep"}
    finished = pyqtSignal(bool)

    def __init__(self, project_name, points_with_files, conf, pixel_sep):
        super().__init__()
        self.project_name      = project_name
        self.points_with_files = points_with_files
        self.conf              = conf
        self.pixel_sep         = pixel_sep

    def run(self):
        import subprocess, sys, json, tempfile
        frames_dir = project_frames_dir(self.project_name)
        if not os.path.exists(POST_MODEL_PATH):
            self.log.emit("No trained model. Run Train YOLO first.")
            self.finished.emit(False)
            return

        params = {
            "frames_dir": frames_dir,
            "model_path": POST_MODEL_PATH,
            "conf":       self.conf,
            "pixel_sep":  self.pixel_sep,
            "points":     [[lat, lon, fname] for lat, lon, fname in self.points_with_files],
        }
        params_file = os.path.join(tempfile.gettempdir(), 'dcv_culvert_params.json')
        with open(params_file, 'w') as f:
            json.dump(params, f)

        script_file = os.path.join(tempfile.gettempdir(), 'dcv_culvert_script.py')
        with open(script_file, 'w', encoding='utf-8') as sf:
            sf.write(_CULVERT_SCRIPT)

        proc = subprocess.Popen(
            [sys.executable, script_file, params_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
        )
        candidates = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("CULVERT:"):
                try:
                    d = json.loads(line[8:])
                    candidates.append(d)
                    self.log.emit(f"Culvert: {d['frame']}  (sep={d['sep']}px)")
                except Exception:
                    pass
            else:
                self.log.emit(line)
        proc.wait()
        self.found.emit(candidates)
        self.finished.emit(proc.returncode == 0)


# ─── Selectable Frame Label ───────────────────────────────────────────────────

class BoxFrameLabel(QLabel):
    """QLabel that lets the user draw boxes and shows them as overlays."""

    box_drawn = pyqtSignal(int, int, int, int)   # x1, y1, x2, y2 in original image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self._boxes = []        # list of (x1,y1,x2,y2) in original image coords
        self._orig_size = None  # (orig_w, orig_h) of last displayed image
        self._draw_mode = False
        self._selecting = False
        self._origin = QPoint()
        self._rb = QRubberBand(QRubberBand.Rectangle, self)

    def set_draw_mode(self, enabled):
        self._draw_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_orig_size(self, orig_w, orig_h):
        self._orig_size = (orig_w, orig_h)

    def clear_boxes(self):
        self._boxes = []
        self.update()

    def _widget_to_orig(self, wx, wy):
        """Convert widget pixel coords to original image pixel coords."""
        if not self._orig_size:
            return wx, wy
        orig_w, orig_h = self._orig_size
        lw, lh = self.width(), self.height()
        scale = min(lw / orig_w, lh / orig_h)
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        ox, oy = (lw - sw) // 2, (lh - sh) // 2
        x = max(0, min(int((wx - ox) / scale), orig_w))
        y = max(0, min(int((wy - oy) / scale), orig_h))
        return x, y

    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton:
            self._selecting = True
            self._origin = event.pos()
            self._rb.setGeometry(QRect(self._origin, QSize()))
            self._rb.show()

    def mouseMoveEvent(self, event):
        if self._selecting:
            self._rb.setGeometry(QRect(self._origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self._selecting and event.button() == Qt.LeftButton:
            self._selecting = False
            self._rb.hide()
            rect = QRect(self._origin, event.pos()).normalized()
            if rect.width() > 5 and rect.height() > 5:
                x1, y1 = self._widget_to_orig(rect.left(), rect.top())
                x2, y2 = self._widget_to_orig(rect.right(), rect.bottom())
                if x2 > x1 and y2 > y1:
                    self._boxes.append((x1, y1, x2, y2))
                    self.update()
                    self.box_drawn.emit(x1, y1, x2, y2)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._boxes or not self._orig_size:
            return
        orig_w, orig_h = self._orig_size
        lw, lh = self.width(), self.height()
        if lw <= 0 or lh <= 0:
            return
        scale = min(lw / orig_w, lh / orig_h)
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        ox, oy = (lw - sw) // 2, (lh - sh) // 2
        p = QPainter(self)
        p.setPen(QPen(QColor(255, 50, 50), 2))
        for (x1, y1, x2, y2) in self._boxes:
            wx = ox + int(x1 * scale)
            wy = oy + int(y1 * scale)
            ww = int((x2 - x1) * scale)
            wh = int((y2 - y1) * scale)
            p.drawRect(wx, wy, ww, wh)


# ─── Single-frame YOLO inference worker ───────────────────────────────────────

class YoloInferWorker(QThread):
    """Run YOLO inference on one frame and return bounding boxes."""
    done = pyqtSignal(list)   # list of (x1,y1,x2,y2)

    def __init__(self, frame_path, model_path, conf):
        super().__init__()
        self.frame_path = frame_path
        self.model_path = model_path
        self.conf = conf

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            results = model(self.frame_path, conf=self.conf, verbose=False, stream=True, half=True, device=0)
            boxes = []
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                        boxes.append((x1, y1, x2, y2))
            self.done.emit(boxes)
        except Exception as e:
            self.done.emit([])


# ─── Map Communicator ─────────────────────────────────────────────────────────

class Communicator(QObject):
    frame_changed = pyqtSignal(int)   # emitted when a new frame is shown (index)

    def __init__(self, coords_label, image_label, points_with_files, web_view, map_var, spin_box, image_dir,
                 marker_color='yellow', marker_radius=10):
        super().__init__()
        self.coords_label = coords_label
        self.image_label = image_label
        self.points_with_files = sorted(points_with_files, key=lambda x: x[2])
        self.image_dir = image_dir
        self.current_index = None
        self.web_view = web_view
        self.map_var = map_var
        self.marker_color = marker_color
        self.marker_radius = marker_radius
        self.auto_center_enabled = False
        self.spin_box = spin_box
        self.prev_timer = QTimer()
        self.prev_timer.timeout.connect(self.show_prev)
        self.next_timer = QTimer()
        self.next_timer.timeout.connect(self.show_next)
        self._road_cache = {}
        self._road_pending = None   # (lat, lon, fname) awaiting lookup
        self._road_reply = None
        self._nam = QNetworkAccessManager()
        self._road_timer = QTimer()
        self._road_timer.setSingleShot(True)
        self._road_timer.timeout.connect(self._do_road_lookup)

    @pyqtSlot(str)
    def update_coords(self, coords):
        lines = coords.split('\n')
        if len(lines) >= 2:
            try:
                clicked_lat = float(lines[0].split(': ', 1)[1])
                clicked_lon = float(lines[1].split(': ', 1)[1])
                min_dist = float('inf')
                closest_index = None
                for i, (lat, lon, fname) in enumerate(self.points_with_files):
                    dist = math.sqrt((lat - clicked_lat) ** 2 + (lon - clicked_lon) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_index = i
                if closest_index is not None:
                    self.current_index = closest_index
                    self._show_frame(closest_index)
                else:
                    self.coords_label.setText(coords)
                    self.image_label.clear()
            except (IndexError, ValueError):
                self.coords_label.setText(coords)
                self.image_label.clear()

    def _show_frame(self, index):
        self.frame_changed.emit(index)
        lat, lon, fname = self.points_with_files[index]
        full_path = os.path.join(self.image_dir, fname)
        if os.path.exists(full_path):
            pixmap = QPixmap(full_path)
            if hasattr(self.image_label, 'set_orig_size'):
                self.image_label.set_orig_size(pixmap.width(), pixmap.height())
            size = self.image_label.size()
            if size.width() > 0 and size.height() > 0:
                pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)
        else:
            self.image_label.setText(f"Image not found:\n{full_path}")

        cache_key = (round(lat, 3), round(lon, 3))
        road = self._road_cache.get(cache_key)
        self._update_coords_label(lat, lon, fname, road)
        self.update_marker(lat, lon)
        if self.auto_center_enabled:
            self.web_view.page().runJavaScript(
                f"{self.map_var}.panTo([{lat}, {lon}]);"
            )

        if road is None:
            self._road_pending = (lat, lon, fname)
            self._road_timer.start(500)  # debounce — wait 500ms before requesting

    def _update_coords_label(self, lat, lon, fname, road):
        road_str = f'  |  {road}' if road else ''
        self.coords_label.setText(
            f"Latitude: {lat:.6f}  Longitude: {lon:.6f}{road_str}\nClosest Image: {fname}"
        )

    def _do_road_lookup(self):
        if not self._road_pending:
            return
        lat, lon, fname = self._road_pending
        if self._road_reply:
            self._road_reply.abort()
            self._road_reply = None
        url = QUrl(f'https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json')
        request = QNetworkRequest(url)
        request.setRawHeader(b'User-Agent', b'DashCamViewer/3.0')
        self._road_reply = self._nam.get(request)
        self._road_reply.finished.connect(
            lambda: self._on_reply_finished(round(lat, 3), round(lon, 3), lat, lon, fname)
        )

    def _on_reply_finished(self, cache_key_lat, cache_key_lon, lat, lon, fname):
        reply = self._road_reply
        road = ''
        try:
            data = json.loads(bytes(reply.readAll()))
            addr = data.get('address', {})
            road = (addr.get('road') or addr.get('path') or
                    addr.get('track') or addr.get('hamlet') or '')
        except Exception:
            pass
        reply.deleteLater()
        self._road_reply = None
        self._road_cache[(cache_key_lat, cache_key_lon)] = road
        if self.current_index is not None:
            _, _, cur_fname = self.points_with_files[self.current_index]
            if cur_fname == fname:
                self._update_coords_label(lat, lon, fname, road)

    @pyqtSlot()
    def show_prev(self):
        if self.current_index is None:
            QMessageBox.information(None, "No Selection", "Click the map to select a starting point.")
            return
        if self.current_index <= 0:
            self.prev_timer.stop()
            QMessageBox.information(None, "No Image", "Already at the first frame.")
            return
        self.current_index -= 1
        self._show_frame(self.current_index)

    @pyqtSlot()
    def show_next(self):
        if self.current_index is None:
            QMessageBox.information(None, "No Selection", "Click the map to select a starting point.")
            return
        if self.current_index >= len(self.points_with_files) - 1:
            self.next_timer.stop()
            QMessageBox.information(None, "No Image", "Already at the last frame.")
            return
        self.current_index += 1
        self._show_frame(self.current_index)

    @pyqtSlot()
    def toggle_auto_center(self):
        self.auto_center_enabled = not self.auto_center_enabled

    def update_marker(self, lat, lon):
        js = f"""
        if (window.marker) {{ {self.map_var}.removeLayer(window.marker); }}
        window.marker = L.circleMarker([{lat}, {lon}], {{
            color: '{self.marker_color}', fillColor: '{self.marker_color}', fillOpacity: 0.8, radius: {self.marker_radius}
        }}).addTo({self.map_var});
        """
        self.web_view.page().runJavaScript(js)

    @pyqtSlot()
    def toggle_auto_prev(self):
        if self.prev_timer.isActive():
            self.prev_timer.stop()
        else:
            self.next_timer.stop()
            self.prev_timer.start(self.spin_box.value())

    @pyqtSlot()
    def toggle_auto_next(self):
        if self.next_timer.isActive():
            self.next_timer.stop()
        else:
            self.prev_timer.stop()
            self.next_timer.start(self.spin_box.value())

    @pyqtSlot(int)
    def update_interval(self, value):
        if self.prev_timer.isActive():
            self.prev_timer.setInterval(value)
        if self.next_timer.isActive():
            self.next_timer.setInterval(value)


# ─── Process Panel ────────────────────────────────────────────────────────────

class ProcessPanel(QWidget):
    processing_done = pyqtSignal()

    def __init__(self, get_project_fn, global_settings, save_settings_fn):
        super().__init__()
        self.get_project = get_project_fn
        self.global_settings = global_settings
        self.save_settings = save_settings_fn
        self.worker = None
        self._loading = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Video processing settings ──
        form_group = QGroupBox("Settings")
        form = QFormLayout(form_group)
        form.setSpacing(8)

        src_row = QHBoxLayout()
        self.src_edit = QLineEdit()
        src_browse = QPushButton("Browse")
        src_browse.clicked.connect(self._browse_src)
        src_row.addWidget(self.src_edit)
        src_row.addWidget(src_browse)
        form.addRow("Video Folder:", src_row)

        exif_row = QHBoxLayout()
        self.exif_edit = QLineEdit(self.global_settings.get("exiftool_path", ""))
        exif_browse = QPushButton("Browse")
        exif_browse.clicked.connect(self._browse_exif)
        exif_row.addWidget(self.exif_edit)
        exif_row.addWidget(exif_browse)
        form.addRow("ExifTool Path:", exif_row)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.5, 60.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" sec")
        self.interval_spin.setFixedWidth(100)
        form.addRow("Frame Interval:", self.interval_spin)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(92)
        self.quality_spin.setFixedWidth(100)
        form.addRow("JPEG Quality:", self.quality_spin)

        layout.addWidget(form_group)

        # ── Survey data settings ──
        survey_group = QGroupBox("Survey Data (Pre-Plot)")
        survey_form = QFormLayout(survey_group)
        survey_form.setSpacing(8)

        csv_row = QHBoxLayout()
        self.preplot_edit = QLineEdit()
        self.preplot_edit.setPlaceholderText("Optional: pre-plot CSV file")
        preplot_browse = QPushButton("Browse")
        preplot_browse.clicked.connect(self._browse_preplot)
        csv_row.addWidget(self.preplot_edit)
        csv_row.addWidget(preplot_browse)
        survey_form.addRow("Pre-Plot CSV:", csv_row)

        self.zone_combo = QComboBox()
        for z in range(46, 60):
            self.zone_combo.addItem(f"MGA Zone {z}", z)
        # Default to zone 55
        default_idx = self.zone_combo.findData(55)
        if default_idx >= 0:
            self.zone_combo.setCurrentIndex(default_idx)
        survey_form.addRow("MGA Zone:", self.zone_combo)

        layout.addWidget(survey_group)

        self.preplot_edit.editingFinished.connect(self._save_survey_settings)
        self.zone_combo.currentIndexChanged.connect(self._save_survey_settings)

        # ── Map Display ──
        display_group = QGroupBox("Map Display")
        disp_form = QFormLayout(display_group)
        disp_form.setSpacing(8)

        def _color_row(default_color, spin_label, default_val):
            row = QHBoxLayout()
            btn = self._mk_color_btn(default_color)
            spin = QSpinBox()
            spin.setRange(1, 20)
            spin.setValue(default_val)
            spin.setFixedWidth(60)
            row.addWidget(btn)
            row.addSpacing(6)
            row.addWidget(QLabel(f"{spin_label}:"))
            row.addWidget(spin)
            row.addStretch()
            return row, btn, spin

        dc_row, self.dc_color_btn, self.dc_width_spin = _color_row('#ffff00', 'Width', 5)
        self.dc_color_btn.clicked.connect(lambda: self._pick_color(self.dc_color_btn))
        disp_form.addRow("Dash cam line:", dc_row)

        sv_row, self.sv_color_btn, self.sv_width_spin = _color_row('#00ccff', 'Width', 2)
        self.sv_color_btn.clicked.connect(lambda: self._pick_color(self.sv_color_btn))
        disp_form.addRow("Survey line:", sv_row)

        sp_row, self.sp_color_btn, self.sp_radius_spin = _color_row('#00ccff', 'Radius', 4)
        self.sp_color_btn.clicked.connect(lambda: self._pick_color(self.sp_color_btn))
        disp_form.addRow("Survey points:", sp_row)

        sm_row, self.sm_color_btn, self.sm_radius_spin = _color_row('#ff3300', 'Radius', 6)
        self.sm_color_btn.clicked.connect(lambda: self._pick_color(self.sm_color_btn))
        disp_form.addRow("Station markers:", sm_row)

        cm_row, self.cm_color_btn, self.cm_radius_spin = _color_row('#ffff00', 'Radius', 10)
        self.cm_color_btn.clicked.connect(lambda: self._pick_color(self.cm_color_btn))
        disp_form.addRow("Cam position:", cm_row)

        sl_row = QHBoxLayout()
        self.sl_check = QCheckBox("Show")
        self.sl_check.setChecked(True)
        self.sl_size_spin = QSpinBox()
        self.sl_size_spin.setRange(6, 24)
        self.sl_size_spin.setValue(11)
        self.sl_size_spin.setFixedWidth(60)
        sl_row.addWidget(self.sl_check)
        sl_row.addSpacing(6)
        sl_row.addWidget(QLabel("Size:"))
        sl_row.addWidget(self.sl_size_spin)
        sl_row.addStretch()
        disp_form.addRow("Station labels:", sl_row)

        slc_row = QHBoxLayout()
        self.sl_text_color_btn = self._mk_color_btn('#ffffff')
        self.sl_text_color_btn.clicked.connect(lambda: self._pick_color(self.sl_text_color_btn))
        self.sl_bg_check = QCheckBox("BG")
        self.sl_bg_check.setChecked(True)
        self.sl_bg_color_btn = self._mk_color_btn('#333333')
        self.sl_bg_color_btn.clicked.connect(lambda: self._pick_color(self.sl_bg_color_btn))
        slc_row.addWidget(QLabel("Text:"))
        slc_row.addWidget(self.sl_text_color_btn)
        slc_row.addSpacing(10)
        slc_row.addWidget(self.sl_bg_check)
        slc_row.addWidget(self.sl_bg_color_btn)
        slc_row.addStretch()
        disp_form.addRow("  └ Colors:", slc_row)

        self.basemap_combo = QComboBox()
        self.basemap_combo.addItem("Satellite", "satellite")
        self.basemap_combo.addItem("OpenStreetMap", "osm")
        self.basemap_combo.addItem("Topo", "topo")
        disp_form.addRow("Base map:", self.basemap_combo)

        layout.addWidget(display_group)

        for w in [self.dc_width_spin, self.sv_width_spin, self.sp_radius_spin,
                  self.sm_radius_spin, self.cm_radius_spin, self.sl_size_spin]:
            w.valueChanged.connect(self._save_display_settings)
        self.sl_check.stateChanged.connect(self._save_display_settings)
        self.sl_bg_check.stateChanged.connect(self._save_display_settings)
        self.basemap_combo.currentIndexChanged.connect(self._save_display_settings)

        # ── Action buttons ──
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedHeight(32)
        self.run_btn.clicked.connect(self._run)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setFixedHeight(32)
        self.abort_btn.clicked.connect(self._abort)
        self.abort_btn.setEnabled(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.abort_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        log_label = QLabel("Log:")
        layout.addWidget(log_label)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_edit, stretch=1)

    def load_project(self, project_data):
        self._loading = True

        self.src_edit.setText(project_data.get("video_source", ""))
        self.interval_spin.setValue(float(project_data.get("frame_interval", 1.0)))
        self.quality_spin.setValue(project_data.get("jpeg_quality", 92))
        self.preplot_edit.setText(project_data.get("preplot_csv", ""))
        zone = project_data.get("mga_zone", 55)
        idx = self.zone_combo.findData(zone)
        if idx >= 0:
            self.zone_combo.setCurrentIndex(idx)

        def set_color(btn, key, default):
            c = project_data.get(key, default)
            btn.setStyleSheet(f"background: {c}; border: 1px solid #aaa;")
            btn.setProperty('hex_color', c)

        set_color(self.dc_color_btn, 'dc_color', '#ffff00')
        self.dc_width_spin.setValue(project_data.get('dc_width', 5))
        set_color(self.sv_color_btn, 'sv_color', '#00ccff')
        self.sv_width_spin.setValue(project_data.get('sv_width', 2))
        set_color(self.sp_color_btn, 'sp_color', '#00ccff')
        self.sp_radius_spin.setValue(project_data.get('sp_radius', 4))
        set_color(self.sm_color_btn, 'sm_color', '#ff3300')
        self.sm_radius_spin.setValue(project_data.get('sm_radius', 6))
        set_color(self.cm_color_btn, 'cm_color', '#ffff00')
        self.cm_radius_spin.setValue(project_data.get('cm_radius', 10))
        self.sl_check.setChecked(project_data.get('sl_show', True))
        self.sl_size_spin.setValue(project_data.get('sl_size', 11))
        set_color(self.sl_text_color_btn, 'sl_text_color', '#ffffff')
        self.sl_bg_check.setChecked(project_data.get('sl_bg_show', True))
        set_color(self.sl_bg_color_btn, 'sl_bg_color', '#333333')
        bm_idx = self.basemap_combo.findData(project_data.get('base_map', 'satellite'))
        if bm_idx >= 0:
            self.basemap_combo.setCurrentIndex(bm_idx)

        self._loading = False
        self.log_edit.clear()

    def _browse_src(self):
        d = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if d:
            self.src_edit.setText(d)

    def _browse_exif(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select ExifTool Executable",
            filter="Executables (*.exe);;All Files (*)"
        )
        if f:
            self.exif_edit.setText(f)
            self.global_settings["exiftool_path"] = f
            self.save_settings(self.global_settings)

    def _browse_preplot(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-Plot CSV",
            filter="CSV Files (*.csv);;All Files (*)"
        )
        if f:
            self.preplot_edit.setText(f)
            self._save_survey_settings()

    def _save_survey_settings(self):
        project = self.get_project()
        if not project:
            return
        project["preplot_csv"] = self.preplot_edit.text().strip()
        project["mga_zone"] = self.zone_combo.currentData()
        save_project(project["name"], project)

    @staticmethod
    def _mk_color_btn(color):
        btn = QPushButton()
        btn.setFixedSize(44, 26)
        btn.setStyleSheet(f"background: {color}; border: 1px solid #aaa;")
        btn.setProperty('hex_color', color)
        return btn

    def _pick_color(self, btn):
        current = btn.property('hex_color') or '#ffffff'
        color = QColorDialog.getColor(QColor(current), self)
        if color.isValid():
            h = color.name()
            btn.setStyleSheet(f"background: {h}; border: 1px solid #aaa;")
            btn.setProperty('hex_color', h)
            self._save_display_settings()

    def _save_display_settings(self):
        if self._loading:
            return
        project = self.get_project()
        if not project:
            return
        project.update({
            'dc_color':  self.dc_color_btn.property('hex_color'),
            'dc_width':  self.dc_width_spin.value(),
            'sv_color':  self.sv_color_btn.property('hex_color'),
            'sv_width':  self.sv_width_spin.value(),
            'sp_color':  self.sp_color_btn.property('hex_color'),
            'sp_radius': self.sp_radius_spin.value(),
            'sm_color':  self.sm_color_btn.property('hex_color'),
            'sm_radius': self.sm_radius_spin.value(),
            'cm_color':  self.cm_color_btn.property('hex_color'),
            'cm_radius': self.cm_radius_spin.value(),
            'sl_show':       self.sl_check.isChecked(),
            'sl_size':       self.sl_size_spin.value(),
            'sl_text_color': self.sl_text_color_btn.property('hex_color'),
            'sl_bg_show':    self.sl_bg_check.isChecked(),
            'sl_bg_color':   self.sl_bg_color_btn.property('hex_color'),
            'base_map':  self.basemap_combo.currentData(),
        })
        save_project(project['name'], project)

    def _run(self):
        project = self.get_project()
        if not project:
            QMessageBox.warning(self, "No Project", "Please select or create a project first.")
            return

        video_dir = self.src_edit.text().strip()
        exiftool = self.exif_edit.text().strip()

        if not video_dir or not os.path.isdir(video_dir):
            QMessageBox.warning(self, "Error", "Please select a valid video folder.")
            return
        if not exiftool or not os.path.isfile(exiftool):
            QMessageBox.warning(self, "Error", "Please select a valid ExifTool path.")
            return

        project["video_source"] = video_dir
        project["frame_interval"] = self.interval_spin.value()
        project["jpeg_quality"] = self.quality_spin.value()
        save_project(project["name"], project)

        self.log_edit.clear()
        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)

        self.worker = ProcessWorker(
            video_dir,
            project_frames_dir(project["name"]),
            project_txt_path(project["name"]),
            self.interval_spin.value(),
            self.quality_spin.value(),
            exiftool,
        )
        self.worker.log.connect(self.log_edit.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _abort(self):
        if self.worker:
            self.worker.abort()

    def _on_finished(self, success):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        if success:
            self.log_edit.append("\nProcessing complete.")
            self.processing_done.emit()


# ─── View Panel ───────────────────────────────────────────────────────────────

class ViewPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.communicator = None
        self.channel = None
        self._tmp_html = None
        self._project_name = None
        self._yolo_worker = None
        self._detect_worker = None
        self._infer_worker = None
        self._detected_boxes = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.coords_label = QLabel("Select a project and run processing, or load an existing project.")
        self.coords_label.setFont(QFont("Arial", 13))
        self.coords_label.setAlignment(Qt.AlignCenter)
        self.coords_label.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.coords_label)

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter, stretch=1)

        self.map_container = QWidget()
        self.map_layout = QVBoxLayout(self.map_container)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = None
        self.splitter.addWidget(self.map_container)

        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = BoxFrameLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.box_drawn.connect(self._on_box_drawn)
        image_layout.addWidget(self.image_label, stretch=1)

        # ── Frame navigation row (light blue background) ──────────────────────
        nav_container = QWidget()
        nav_container.setStyleSheet("background-color: #ddeeff; border-radius: 4px;")
        btn_row = QHBoxLayout(nav_container)
        btn_row.setContentsMargins(6, 6, 6, 6)
        btn_row.setSpacing(6)
        btn_row.addStretch()
        self.auto_left = QPushButton("Auto ←")
        self.left_btn = QPushButton("← Prev")
        self.spin_box = QSpinBox()
        self.spin_box.setRange(100, 5000)
        self.spin_box.setValue(1000)
        self.spin_box.setSingleStep(100)
        self.spin_box.setSuffix(" ms")
        self.right_btn = QPushButton("Next →")
        self.auto_right = QPushButton("Auto →")
        self.center_toggle = QPushButton("Auto Center")
        self.center_toggle.setCheckable(True)

        _nav_btn_style = (
            "QPushButton {"
            "  padding: 4px 14px;"
            "  min-height: 26px;"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f5f5f5,stop:1 #dcdcdc);"
            "  border: 1px solid #aaa;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #fff,stop:1 #e8e8e8); }"
            "QPushButton:pressed { background: #c8c8c8; border: 1px solid #888; }"
        )
        self.center_toggle.setStyleSheet(
            _nav_btn_style +
            "QPushButton:checked { background: #4a9eff; color: white; border-color: #2277cc; }"
        )
        for w in [self.auto_left, self.left_btn, self.right_btn, self.auto_right]:
            w.setStyleSheet(_nav_btn_style)
            w.setMinimumHeight(30)
        self.spin_box.setMinimumHeight(30)
        self.center_toggle.setMinimumHeight(30)

        for w in [self.auto_left, self.left_btn, self.spin_box, self.right_btn, self.auto_right, self.center_toggle]:
            btn_row.addWidget(w)
        btn_row.addStretch()
        image_layout.addWidget(nav_container)

        # ── Separator ─────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        image_layout.addWidget(sep)

        # ── Post detection row 1: boxing + model controls ─────────────────────
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Post Detection"))
        self.box_chk = QCheckBox("Box Posts")
        self.box_chk.setToolTip("Check then drag boxes around posts on the image — each box saves automatically")
        self.box_chk.toggled.connect(self._toggle_box_mode)
        id_row.addWidget(self.box_chk)
        self.prev_post_btn = QPushButton("← Post")
        self.prev_post_btn.setToolTip("Jump to previous frame with a boxed post")
        self.prev_post_btn.clicked.connect(self._prev_post_frame)
        id_row.addWidget(self.prev_post_btn)
        self.next_post_btn = QPushButton("Post →")
        self.next_post_btn.setToolTip("Jump to next frame with a boxed post")
        self.next_post_btn.clicked.connect(self._next_post_frame)
        id_row.addWidget(self.next_post_btn)
        self.train_btn = QPushButton("Train YOLO")
        self.train_btn.clicked.connect(self._train_yolo)
        id_row.addWidget(self.train_btn)
        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self._run_detection)
        id_row.addWidget(self.detect_btn)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setPrefix("conf ")
        self.conf_spin.setFixedWidth(154)
        id_row.addWidget(self.conf_spin)
        id_row.addStretch()
        image_layout.addLayout(id_row)

        # ── Post detection row 2: marker style + remove ───────────────────────
        marker_row = QHBoxLayout()
        self.show_posts_chk = QCheckBox("Show Posts")
        self.show_posts_chk.setChecked(True)
        self.show_posts_chk.toggled.connect(self._toggle_posts_layer)
        marker_row.addWidget(self.show_posts_chk)
        marker_row.addWidget(QLabel("Marker:"))
        self.id_shape_combo = QComboBox()
        self.id_shape_combo.addItems(["Circle", "Square", "Triangle"])
        marker_row.addWidget(self.id_shape_combo)
        self.id_color_combo = QComboBox()
        self.id_color_combo.addItems(["red", "blue", "green", "orange", "purple", "yellow"])
        marker_row.addWidget(self.id_color_combo)
        self.id_size_combo = QComboBox()
        self.id_size_combo.addItems(["Small", "Medium", "Large"])
        self.id_size_combo.setCurrentText("Medium")
        marker_row.addWidget(self.id_size_combo)
        self.remove_post_btn = QPushButton("Remove")
        self.remove_post_btn.clicked.connect(self._remove_post)
        marker_row.addWidget(self.remove_post_btn)
        self.remove_all_btn = QPushButton("Remove All")
        self.remove_all_btn.clicked.connect(self._remove_all_markers)
        marker_row.addWidget(self.remove_all_btn)
        marker_row.addStretch()
        image_layout.addLayout(marker_row)

        # ── Culvert detection row ─────────────────────────────────────────────
        culvert_row = QHBoxLayout()
        self.show_culverts_chk = QCheckBox("Show Culverts")
        self.show_culverts_chk.setChecked(True)
        self.show_culverts_chk.toggled.connect(self._toggle_culverts_layer)
        culvert_row.addWidget(self.show_culverts_chk)
        self.find_culverts_btn = QPushButton("Find Culverts")
        self.find_culverts_btn.setToolTip("Scan all frames with YOLO and flag frames where 2 posts are the given pixel distance apart")
        self.find_culverts_btn.clicked.connect(self._find_culverts)
        culvert_row.addWidget(self.find_culverts_btn)

        self.culvert_sep_spin = QSpinBox()
        self.culvert_sep_spin.setRange(10, 2000)
        self.culvert_sep_spin.setValue(200)
        self.culvert_sep_spin.setSingleStep(10)
        self.culvert_sep_spin.setPrefix("sep ")
        self.culvert_sep_spin.setSuffix(" px")
        self.culvert_sep_spin.setFixedWidth(110)
        culvert_row.addWidget(self.culvert_sep_spin)

        culvert_row.addWidget(QLabel("Culvert:"))
        self.culvert_shape_combo = QComboBox()
        self.culvert_shape_combo.addItems(["Circle", "Square", "Diamond"])
        culvert_row.addWidget(self.culvert_shape_combo)
        self.culvert_color_combo = QComboBox()
        self.culvert_color_combo.addItems(["cyan", "orange", "yellow", "red", "blue", "green", "purple"])
        culvert_row.addWidget(self.culvert_color_combo)
        self.culvert_size_combo = QComboBox()
        self.culvert_size_combo.addItems(["Small", "Medium", "Large"])
        self.culvert_size_combo.setCurrentText("Medium")
        culvert_row.addWidget(self.culvert_size_combo)

        self.prev_culvert_btn = QPushButton("← Culvert")
        self.prev_culvert_btn.setToolTip("Jump to previous culvert frame")
        self.prev_culvert_btn.clicked.connect(self._prev_culvert_frame)
        culvert_row.addWidget(self.prev_culvert_btn)

        self.next_culvert_btn = QPushButton("Culvert →")
        self.next_culvert_btn.setToolTip("Jump to next culvert frame")
        self.next_culvert_btn.clicked.connect(self._next_culvert_frame)
        culvert_row.addWidget(self.next_culvert_btn)

        self.add_culvert_btn = QPushButton("Add Culvert")
        self.add_culvert_btn.setToolTip("Mark the current frame as a culvert location")
        self.add_culvert_btn.clicked.connect(self._add_culvert)
        culvert_row.addWidget(self.add_culvert_btn)

        self.remove_culvert_btn = QPushButton("Remove")
        self.remove_culvert_btn.clicked.connect(self._remove_culvert)
        culvert_row.addWidget(self.remove_culvert_btn)

        self.remove_all_culverts_btn = QPushButton("Remove All")
        self.remove_all_culverts_btn.clicked.connect(self._remove_all_culverts)
        culvert_row.addWidget(self.remove_all_culverts_btn)

        culvert_row.addStretch()
        image_layout.addLayout(culvert_row)

        # ── Vehicle count / traffic density row ───────────────────────────────
        traffic_row = QHBoxLayout()

        self.count_vehicles_btn = QPushButton("Count Vehicles")
        self.count_vehicles_btn.setToolTip("Run YOLO on all frames to count oncoming vehicles per segment")
        self.count_vehicles_btn.clicked.connect(self._count_vehicles)
        traffic_row.addWidget(self.count_vehicles_btn)

        self.vehicle_interval_spin = QDoubleSpinBox()
        self.vehicle_interval_spin.setRange(0.5, 60.0)
        self.vehicle_interval_spin.setSingleStep(0.5)
        self.vehicle_interval_spin.setDecimals(1)
        self.vehicle_interval_spin.setValue(0.5)
        self.vehicle_interval_spin.setSuffix(" sec")
        self.vehicle_interval_spin.setFixedWidth(90)
        self.vehicle_interval_spin.setToolTip("Frame interval used when frames were extracted")
        traffic_row.addWidget(QLabel("Interval:"))
        traffic_row.addWidget(self.vehicle_interval_spin)

        self.vehicle_segment_spin = QDoubleSpinBox()
        self.vehicle_segment_spin.setRange(0.1, 10.0)
        self.vehicle_segment_spin.setSingleStep(0.1)
        self.vehicle_segment_spin.setDecimals(1)
        self.vehicle_segment_spin.setValue(1.0)
        self.vehicle_segment_spin.setSuffix(" km")
        self.vehicle_segment_spin.setFixedWidth(90)
        self.vehicle_segment_spin.setToolTip("Road segment length for grouping vehicle counts")
        traffic_row.addWidget(QLabel("Segment:"))
        traffic_row.addWidget(self.vehicle_segment_spin)

        self.vehicle_conf_spin = QDoubleSpinBox()
        self.vehicle_conf_spin.setRange(0.05, 0.95)
        self.vehicle_conf_spin.setSingleStep(0.05)
        self.vehicle_conf_spin.setDecimals(2)
        self.vehicle_conf_spin.setValue(0.15)
        self.vehicle_conf_spin.setPrefix("conf ")
        self.vehicle_conf_spin.setFixedWidth(100)
        self.vehicle_conf_spin.setToolTip("YOLO confidence threshold for vehicle detection")
        traffic_row.addWidget(self.vehicle_conf_spin)

        self.show_traffic_chk = QCheckBox("Show Heatmap")
        self.show_traffic_chk.setChecked(True)
        self.show_traffic_chk.toggled.connect(self._toggle_traffic_layer)
        traffic_row.addWidget(self.show_traffic_chk)

        self.clear_traffic_btn = QPushButton("Clear")
        self.clear_traffic_btn.clicked.connect(self._clear_vehicle_heatmap)
        traffic_row.addWidget(self.clear_traffic_btn)

        traffic_row.addStretch()
        image_layout.addLayout(traffic_row)

        self.splitter.addWidget(image_widget)
        self._load_post_settings()
        self._load_culvert_settings()
        self.splitter.setSizes([500, 900])

    # ── Post detection helpers ────────────────────────────────────────────────

    def _load_post_settings(self):
        s = load_post_settings()
        if 'shape' in s:
            self.id_shape_combo.setCurrentText(s['shape'])
        if 'color' in s:
            self.id_color_combo.setCurrentText(s['color'])
        if 'size' in s:
            self.id_size_combo.setCurrentText(s['size'])

    def _save_post_settings(self):
        s = {
            'shape': self.id_shape_combo.currentText(),
            'color': self.id_color_combo.currentText(),
            'size':  self.id_size_combo.currentText(),
        }
        save_post_settings(s)

    def _remove_post(self):
        if not self.communicator or self.communicator.current_index is None:
            QMessageBox.warning(self, "No Frame", "Select a frame on the map first.")
            return

        _, _, fname = self.communicator.points_with_files[self.communicator.current_index]
        csv_path = project_posts_csv(self._project_name)

        if not os.path.exists(csv_path):
            self.coords_label.setText("No posts recorded for this project.")
            return

        # Read all rows, find matching ones
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            all_rows = list(reader)

        matching = [r for r in all_rows if r.get('source_frame') == fname]
        if not matching:
            self.coords_label.setText(f"No post recorded for current frame.")
            return

        # Delete saved images
        posts_dir = os.path.join(PROJECTS_DIR, self._project_name, "posts")
        for row in matching:
            img_file = os.path.join(posts_dir, row.get('saved_image', ''))
            if os.path.exists(img_file):
                os.remove(img_file)

        # Rewrite CSV without matching rows
        remaining = [r for r in all_rows if r.get('source_frame') != fname]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(remaining)

        # Log as false detection (hard negative for future training)
        fd_csv = project_false_detections_csv(self._project_name)
        existing_fd = set()
        if os.path.exists(fd_csv):
            with open(fd_csv, newline='') as f:
                existing_fd = {r['source_frame'] for r in csv.DictReader(f)}
        if fname not in existing_fd:
            write_header = not os.path.exists(fd_csv)
            with open(fd_csv, 'a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['source_frame'])
                w.writerow([fname])

        # Remove marker from map
        safe_frame = fname.replace("'", "\\'")
        js = (f"if (window._postLayerGroup && window._postMarkers && window._postMarkers['{safe_frame}']) {{"
              f"  window._postLayerGroup.removeLayer(window._postMarkers['{safe_frame}']);"
              f"  delete window._postMarkers['{safe_frame}'];"
              f"}}")
        self.web_view.page().runJavaScript(js)
        self.coords_label.setText(f"Post removed and logged as false detection: {fname}")

    def _remove_all_markers(self):
        if not self._project_name:
            return
        reply = QMessageBox.question(
            self, "Remove All Posts",
            "Delete all boxed posts for this project?\n\nThis will remove posts.csv and the posts folder.",
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if reply != QMessageBox.Yes:
            return

        import shutil
        csv_path  = project_posts_csv(self._project_name)
        posts_dir = os.path.join(PROJECTS_DIR, self._project_name, "posts")

        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.isdir(posts_dir):
            shutil.rmtree(posts_dir)

        # Clear map markers
        if self.web_view and self.communicator:
            js = """
            if (window._postLayerGroup) { window._postLayerGroup.clearLayers(); }
            window._postMarkers = {};
            """
            self.web_view.page().runJavaScript(js)

        self.coords_label.setText("All posts removed (posts.csv and posts folder deleted).")

    # ── Culvert helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_marker_js(lat, lon, popup, color, radius, shape, layer_var, markers_var, key, map_var):
        """Return JS string that adds a shaped marker to a Leaflet layer."""
        init = (f"window.{markers_var} = window.{markers_var} || {{}};"
                f"if (!window.{layer_var}) {{ window.{layer_var} = L.layerGroup().addTo({map_var}); }}")
        if shape == 'Circle':
            marker = (f"L.circleMarker([{lat},{lon}],"
                      f"{{radius:{radius},color:'white',weight:1.5,"
                      f"fillColor:'{color}',fillOpacity:0.9}})"
                      f".bindPopup({popup})")
        else:
            sz = radius * 2
            if shape == 'Square':
                shape_svg = f'<rect width=\\"{sz}\\" height=\\"{sz}\\" fill=\\"{color}\\" stroke=\\"white\\" stroke-width=\\"1.5\\"/>'
            elif shape == 'Triangle':
                pts = f"0,{sz} {sz},{sz} {sz//2},0"
                shape_svg = f'<polygon points=\\"{pts}\\" fill=\\"{color}\\" stroke=\\"white\\" stroke-width=\\"1.5\\"/>'
            else:  # Diamond
                h = sz // 2
                shape_svg = f'<polygon points=\\"{h},0 {sz},{h} {h},{sz} 0,{h}\\" fill=\\"{color}\\" stroke=\\"white\\" stroke-width=\\"1.5\\"/>'
            svg = f'<svg width=\\"{sz}\\" height=\\"{sz}\\" xmlns=\\"http://www.w3.org/2000/svg\\">{shape_svg}</svg>'
            marker = (f"L.marker([{lat},{lon}],{{icon:L.divIcon({{"
                      f"html:'{svg}',iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                      f").bindPopup({popup})")
        return f"{init} window.{markers_var}['{key}'] = {marker}.addTo(window.{layer_var});"

    def _toggle_posts_layer(self, visible):
        if not self.web_view:
            return
        mv = self.communicator.map_var if self.communicator else 'map'
        if visible:
            js = f"if (window._postLayerGroup) {{ window._postLayerGroup.addTo({mv}); }}"
        else:
            js = "if (window._postLayerGroup) { window._postLayerGroup.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _toggle_culverts_layer(self, visible):
        if not self.web_view:
            return
        mv = self.communicator.map_var if self.communicator else 'map'
        if visible:
            js = f"if (window._culvertLayerGroup) {{ window._culvertLayerGroup.addTo({mv}); }}"
        else:
            js = "if (window._culvertLayerGroup) { window._culvertLayerGroup.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _load_culvert_settings(self):
        s = load_culvert_settings()
        if 'shape' in s:
            self.culvert_shape_combo.setCurrentText(s['shape'])
        if 'color' in s:
            self.culvert_color_combo.setCurrentText(s['color'])
        if 'size' in s:
            self.culvert_size_combo.setCurrentText(s['size'])
        if 'pixel_sep' in s:
            self.culvert_sep_spin.setValue(s['pixel_sep'])

    def _save_culvert_settings(self):
        save_culvert_settings({
            'shape':     self.culvert_shape_combo.currentText(),
            'color':     self.culvert_color_combo.currentText(),
            'size':      self.culvert_size_combo.currentText(),
            'pixel_sep': self.culvert_sep_spin.value(),
        })

    def _add_culvert_marker_js(self, lat, lon, fname):
        cs     = load_culvert_settings()
        color  = cs.get('color', 'cyan')
        radius = {'Small': 6, 'Medium': 10, 'Large': 14}.get(cs.get('size', 'Medium'), 10)
        shape  = cs.get('shape', 'Circle')
        safe   = fname.replace("'", "\\'")
        popup  = f"'culvert<br>{lat:.5f}, {lon:.5f}'"
        return self._make_marker_js(lat, lon, popup, color, radius, shape,
                                    '_culvertLayerGroup', '_culvertMarkers',
                                    safe, self.communicator.map_var)

    def _find_culverts(self):
        if not self._project_name or not self.communicator:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        if not os.path.exists(POST_MODEL_PATH):
            QMessageBox.warning(self, "No Model", "Train YOLO first.")
            return
        self._save_culvert_settings()
        self.find_culverts_btn.setEnabled(False)
        self._culvert_worker = CulvertFindWorker(
            self._project_name,
            self.communicator.points_with_files,
            self.conf_spin.value(),
            self.culvert_sep_spin.value(),
        )
        self._culvert_worker.log.connect(self.coords_label.setText)
        self._culvert_worker.found.connect(self._on_culverts_found)
        self._culvert_worker.finished.connect(lambda _: self.find_culverts_btn.setEnabled(True))
        self._culvert_worker.start()

    def _on_culverts_found(self, candidates):
        if not candidates:
            self.coords_label.setText("No culvert candidates found.")
            return
        for d in candidates:
            js = self._add_culvert_marker_js(d['lat'], d['lon'], d['frame'])
            self.web_view.page().runJavaScript(js)
            self._write_culvert_row(d['frame'], d['lat'], d['lon'], d['sep'])
        self.coords_label.setText(f"Found {len(candidates)} culvert candidate(s) — markers added to map.")
        # Navigate to first candidate
        fname = candidates[0]['frame']
        for i, (_, _, fn) in enumerate(self.communicator.points_with_files):
            if fn == fname:
                self.communicator.current_index = i
                self.communicator._show_frame(i)
                break

    def _write_culvert_row(self, fname, lat, lon, sep=0):
        if not self._project_name:
            return
        csv_path = project_culverts_csv(self._project_name)
        # Avoid duplicates
        existing = set()
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    existing.add(row.get('source_frame', ''))
        if fname in existing:
            return
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['name', 'latitude', 'longitude', 'source_frame', 'pixel_sep'])
            w.writerow(['culvert', lat, lon, fname, sep])

    def _add_culvert(self):
        if not self.communicator or self.communicator.current_index is None:
            QMessageBox.warning(self, "No Frame", "Select a frame on the map first.")
            return
        if not self._project_name:
            return
        lat, lon, fname = self.communicator.points_with_files[self.communicator.current_index]
        self._write_culvert_row(fname, lat, lon)
        js = self._add_culvert_marker_js(lat, lon, fname)
        self.web_view.page().runJavaScript(js)
        self.coords_label.setText(f"Culvert added: {fname}")

    def _remove_culvert(self):
        if not self.communicator or self.communicator.current_index is None:
            QMessageBox.warning(self, "No Frame", "Select a frame on the map first.")
            return
        _, _, fname = self.communicator.points_with_files[self.communicator.current_index]
        csv_path = project_culverts_csv(self._project_name)
        if not os.path.exists(csv_path):
            self.coords_label.setText("No culverts recorded for this project.")
            return
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if r.get('source_frame') != fname]
        if len(kept) == len(rows):
            self.coords_label.setText("No culvert entry for this frame.")
            return
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'latitude', 'longitude', 'source_frame', 'pixel_sep'])
            w.writeheader()
            w.writerows(kept)
        safe = fname.replace("'", "\\'")
        js = (f"if (window._culvertMarkers && window._culvertMarkers['{safe}']) {{"
              f"  window._culvertLayerGroup.removeLayer(window._culvertMarkers['{safe}']);"
              f"  delete window._culvertMarkers['{safe}']; }}")
        self.web_view.page().runJavaScript(js)
        self.coords_label.setText(f"Culvert removed: {fname}")

    def _remove_all_culverts(self):
        if not self._project_name:
            return
        reply = QMessageBox.question(
            self, "Remove All Culverts",
            "Delete all culvert records for this project?\n\nThis will remove culverts.csv.",
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if reply != QMessageBox.Yes:
            return
        csv_path = project_culverts_csv(self._project_name)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if self.web_view:
            js = ("if (window._culvertLayerGroup) { window._culvertLayerGroup.clearLayers(); }"
                  "window._culvertMarkers = {};")
            self.web_view.page().runJavaScript(js)
        self.coords_label.setText("All culvert records removed.")

    # ── Vehicle / traffic density helpers ────────────────────────────────────

    def _count_vehicles(self):
        if not self._project_name or not self.communicator:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        self.count_vehicles_btn.setEnabled(False)
        self._vehicle_worker = VehicleCountWorker(
            self._project_name,
            self.communicator.points_with_files,
            self.vehicle_conf_spin.value(),
            self.vehicle_interval_spin.value(),
            self.vehicle_segment_spin.value(),
        )
        self._vehicle_worker.log.connect(self.coords_label.setText)
        self._vehicle_worker.done.connect(self._on_vehicles_counted)
        self._vehicle_worker.finished.connect(lambda _: self.count_vehicles_btn.setEnabled(True))
        self._vehicle_worker.start()

    def _on_vehicles_counted(self, result):
        segments = result.get('segments', [])
        max_count = result.get('max_count', 1) or 1
        if not segments:
            self.coords_label.setText("No vehicles detected.")
            return

        def _heat_color(count, max_c):
            if max_c == 0 or count == 0:
                return '#888888'
            t = min(count / max_c, 1.0)
            if t < 0.5:
                r = int(255 * t * 2)
                g = 255
            else:
                r = 255
                g = int(255 * (1 - (t - 0.5) * 2))
            return f'#{r:02x}{g:02x}00'

        mv = self.communicator.map_var
        lines = [
            "window._trafficLayer = window._trafficLayer || L.layerGroup().addTo(" + mv + ");",
            "window._trafficLayer.clearLayers();",
        ]
        for seg in segments:
            path  = seg['path']
            count = seg['count']
            if len(path) < 2:
                continue
            color   = _heat_color(count, max_count)
            weight  = 4 + int(count / max_count * 6)
            latlngs = json.dumps(path)
            sid     = seg['seg_id'] + 1
            skm     = seg['start_km']
            popup   = f"'Segment {sid}  ({skm:.1f}km)<br>{count} oncoming vehicle(s)'"
            lines.append(
                f"L.polyline({latlngs},{{color:'{color}',weight:{weight},opacity:0.85}})"
                f".bindPopup({popup}).addTo(window._trafficLayer);"
            )

        js = "setTimeout(function(){\n" + "\n".join(lines) + "\n}, 100);"
        self.web_view.page().runJavaScript(js)
        total = sum(s['count'] for s in segments)
        self.coords_label.setText(
            f"Traffic density drawn — {total} oncoming vehicles across {len(segments)} segment(s). "
            f"Max per segment: {max_count}."
        )

    def _toggle_traffic_layer(self, visible):
        if not self.web_view:
            return
        mv = self.communicator.map_var if self.communicator else 'map'
        if visible:
            js = f"if (window._trafficLayer) {{ window._trafficLayer.addTo({mv}); }}"
        else:
            js = "if (window._trafficLayer) { window._trafficLayer.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _clear_vehicle_heatmap(self):
        if not self.web_view:
            return
        self.web_view.page().runJavaScript(
            "if (window._trafficLayer) { window._trafficLayer.clearLayers(); }"
        )
        self.coords_label.setText("Traffic heatmap cleared.")

    def _train_yolo(self):
        if not self._project_name:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        self.train_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self._yolo_worker = YoloTrainWorker(self._project_name)
        self._yolo_worker.log.connect(self.coords_label.setText)
        self._yolo_worker.finished.connect(self._on_train_finished)
        self._yolo_worker.start()

    def _on_train_finished(self, success):
        self.train_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        if success:
            self.coords_label.setText(
                f"Training complete — model ready at post_model/weights/best.pt"
            )

    def _run_detection(self):
        if not self._project_name or not self.communicator:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        if not os.path.exists(POST_MODEL_PATH):
            QMessageBox.warning(self, "No Model", "Train YOLO first.")
            return
        self.detect_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self._detect_worker = YoloDetectWorker(
            self._project_name,
            self.conf_spin.value(),
            self.communicator.points_with_files,
        )
        self._detect_worker.log.connect(self.coords_label.setText)
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.start()

    def _on_detect_finished(self, success):
        self.detect_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        rows = getattr(self._detect_worker, 'new_rows', [])
        if not rows:
            return
        cs     = load_post_settings()
        color  = cs.get('color', 'red')
        radius = {'Small': 6, 'Medium': 10, 'Large': 14}.get(cs.get('size', 'Medium'), 10)
        shape  = cs.get('shape', 'Circle')
        mv     = self.communicator.map_var
        for d in rows:
            fname = d['source_frame']
            lat, lon = d['latitude'], d['longitude']
            safe  = fname.replace("'", "\\'")
            popup = f"'post (YOLO)<br>{lat:.5f}, {lon:.5f}'"
            js = self._make_marker_js(lat, lon, popup, color, radius, shape,
                                      '_postLayerGroup', '_postMarkers', safe, mv)
            self.web_view.page().runJavaScript(js)

    def _on_frame_changed(self, index):
        """Clear drawn boxes when navigating to a new frame (keep draw mode state)."""
        self._detected_boxes = []
        self.image_label.clear_boxes()
        if self.box_chk.isChecked():
            self.image_label.set_draw_mode(True)

    def _post_frame_indices(self):
        """Return sorted list of points_with_files indices that have a posts.csv entry."""
        if not self.communicator or not self._project_name:
            return []
        csv_path = project_posts_csv(self._project_name)
        if not os.path.exists(csv_path):
            return []
        try:
            with open(csv_path, newline='') as f:
                post_frames = {row['source_frame'] for row in csv.DictReader(f)}
        except Exception:
            return []
        return [i for i, (_, _, fname) in enumerate(self.communicator.points_with_files)
                if fname in post_frames]

    def _prev_post_frame(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        indices = self._post_frame_indices()
        if not indices:
            self.coords_label.setText("No boxed post frames found in this project.")
            return
        cur = self.communicator.current_index
        before = [i for i in indices if i < cur]
        target = before[-1] if before else indices[-1]   # wrap to last
        self.communicator.current_index = target
        self.communicator._show_frame(target)

    def _next_post_frame(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        indices = self._post_frame_indices()
        if not indices:
            self.coords_label.setText("No boxed post frames found in this project.")
            return
        cur = self.communicator.current_index
        after = [i for i in indices if i > cur]
        target = after[0] if after else indices[0]       # wrap to first
        self.communicator.current_index = target
        self.communicator._show_frame(target)

    def _culvert_frame_indices(self):
        if not self.communicator or not self._project_name:
            return []
        csv_path = project_culverts_csv(self._project_name)
        if not os.path.exists(csv_path):
            return []
        try:
            with open(csv_path, newline='') as f:
                culvert_frames = {row['source_frame'] for row in csv.DictReader(f)}
        except Exception:
            return []
        return [i for i, (_, _, fname) in enumerate(self.communicator.points_with_files)
                if fname in culvert_frames]

    def _prev_culvert_frame(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        indices = self._culvert_frame_indices()
        if not indices:
            self.coords_label.setText("No culvert frames found in this project.")
            return
        cur = self.communicator.current_index
        before = [i for i in indices if i < cur]
        target = before[-1] if before else indices[-1]
        self.communicator.current_index = target
        self.communicator._show_frame(target)

    def _next_culvert_frame(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        indices = self._culvert_frame_indices()
        if not indices:
            self.coords_label.setText("No culvert frames found in this project.")
            return
        cur = self.communicator.current_index
        after = [i for i in indices if i > cur]
        target = after[0] if after else indices[0]
        self.communicator.current_index = target
        self.communicator._show_frame(target)

    def _toggle_box_mode(self, checked):
        self.image_label.set_draw_mode(checked)
        if checked:
            self.coords_label.setText("Box mode ON — drag to draw a box around a post. Each box saves automatically.")
        else:
            self.coords_label.setText("Box mode OFF.")

    def _on_box_drawn(self, x1, y1, x2, y2):
        """Called immediately when user finishes drawing a box. Saves crop + CSV row."""
        if not self.communicator or self.communicator.current_index is None:
            return
        if not self._project_name:
            return

        lat, lon, fname = self.communicator.points_with_files[self.communicator.current_index]
        name = "post"

        posts_dir = os.path.join(PROJECTS_DIR, self._project_name, "posts")
        os.makedirs(posts_dir, exist_ok=True)
        img_path = os.path.join(self.communicator.image_dir, fname)
        orig = QPixmap(img_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"{name}_{ts}.jpg"
        if not orig.isNull():
            crop = orig.copy(QRect(x1, y1, x2 - x1, y2 - y1))
            crop.save(os.path.join(posts_dir, img_filename), "JPEG", 90)

        csv_path = os.path.join(PROJECTS_DIR, self._project_name, "posts.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["name", "latitude", "longitude", "source_frame", "saved_image",
                             "box_x", "box_y", "box_w", "box_h"])
            w.writerow([name, lat, lon, fname, img_filename,
                        x1, y1, x2 - x1, y2 - y1])

        n = len(self.image_label._boxes)
        self.coords_label.setText(f"Saved box {n} for {fname}  ({x2-x1}x{y2-y1}px)  — posts/")

    def load_project(self, project_name):
        import traceback
        try:
            self._load_project_inner(project_name)
        except Exception as e:
            msg = traceback.format_exc()
            self.coords_label.setText(f"Viewer error: {e}")
            QMessageBox.critical(self, "Viewer Error", msg)

    def _load_project_inner(self, project_name):
        if self.communicator:
            self.communicator.prev_timer.stop()
            self.communicator.next_timer.stop()
            self.communicator = None
        self.image_label.clear()
        self.coords_label.setText("Click on the map to get coordinates")
        if self.web_view is not None:
            self.map_layout.removeWidget(self.web_view)
            self.web_view.setParent(None)
            self.web_view.deleteLater()
        self.web_view = QWebEngineView()
        self.map_layout.addWidget(self.web_view)
        self.splitter.setSizes([500, 900])

        self._project_name = project_name
        txt_path = project_txt_path(project_name)
        frames_dir = project_frames_dir(project_name)

        # Load project config for display settings
        try:
            project_data = load_project(project_name)
        except Exception:
            project_data = {}

        dc_color  = project_data.get('dc_color',  '#ffff00')
        dc_width  = project_data.get('dc_width',  5)
        sv_color  = project_data.get('sv_color',  '#00ccff')
        sv_width  = project_data.get('sv_width',  2)
        sp_color  = project_data.get('sp_color',  '#00ccff')
        sp_radius = project_data.get('sp_radius', 4)
        sm_color  = project_data.get('sm_color',  '#ff3300')
        sm_radius = project_data.get('sm_radius', 6)
        sl_show      = 'true' if project_data.get('sl_show', True) else 'false'
        sl_size      = project_data.get('sl_size',       11)
        sl_text_color = project_data.get('sl_text_color', '#ffffff')
        sl_bg_show   = project_data.get('sl_bg_show',    True)
        sl_bg_color  = project_data.get('sl_bg_color',   '#333333')
        base_map  = project_data.get('base_map',  'satellite')

        if not os.path.exists(txt_path):
            self.coords_label.setText("No data found for this project. Run processing first.")
            self.web_view.setHtml(
                "<h3 style='text-align:center;margin-top:60px;color:#888'>No map data available</h3>"
            )
            return

        try:
            grouped_points, points_with_files = self._read_points(txt_path)
        except Exception as e:
            self.coords_label.setText(f"Error loading data: {e}")
            return

        all_points = [p for pts in grouped_points.values() for p in pts]
        if not all_points:
            self.coords_label.setText("No GPS points found in data.")
            return

        center_lat = statistics.mean(lat for lat, lon in all_points)
        center_lon = statistics.mean(lon for lat, lon in all_points)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)

        if base_map == 'osm':
            folium.TileLayer('OpenStreetMap', name='OpenStreetMap', overlay=False, control=True).add_to(m)
        elif base_map == 'topo':
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri', name='Topo Map', overlay=False, control=True
            ).add_to(m)
        else:
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri', name='Satellite Imagery', overlay=False, control=True
            ).add_to(m)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri', name='Roads', overlay=True, control=True
        ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri', name='Places & Boundaries', overlay=True, control=True
        ).add_to(m)

        for section_name, points in grouped_points.items():
            if len(points) > 1:
                folium.PolyLine(
                    locations=[(lat, lon) for lat, lon in points],
                    color=dc_color, weight=dc_width, opacity=0.8,
                    popup=f"Section: {section_name}"
                ).add_to(m)

        folium.LayerControl().add_to(m)
        map_var = m.get_name()

        # ── Load pre-plot survey data if configured ──
        preplot_points = []
        try:
            preplot_csv = project_data.get("preplot_csv", "")
            mga_zone = project_data.get("mga_zone", 55)
            if preplot_csv and os.path.exists(preplot_csv):
                preplot_points = load_preplot_csv(preplot_csv, mga_zone)
                print(f"[preplot] {len(preplot_points)} points loaded (zone {mga_zone})")
        except Exception as e:
            print(f"[preplot] Load error: {e}")

        # ── Set up web channel ──
        self.channel = QWebChannel()
        self.web_view.page().setWebChannel(self.channel)

        self.communicator = Communicator(
            self.coords_label, self.image_label,
            points_with_files, self.web_view,
            map_var, self.spin_box, frames_dir,
            marker_color=project_data.get('cm_color', 'yellow'),
            marker_radius=project_data.get('cm_radius', 10),
        )
        self.channel.registerObject('communicator', self.communicator)
        self.communicator.frame_changed.connect(self._on_frame_changed)

        for btn in [self.auto_left, self.left_btn, self.right_btn, self.auto_right, self.center_toggle]:
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
        try:
            self.spin_box.valueChanged.disconnect()
        except TypeError:
            pass

        self.auto_left.clicked.connect(self.communicator.toggle_auto_prev)
        self.left_btn.clicked.connect(self.communicator.show_prev)
        self.right_btn.clicked.connect(self.communicator.show_next)
        self.auto_right.clicked.connect(self.communicator.toggle_auto_next)
        self.spin_box.valueChanged.connect(self.communicator.update_interval)
        self.center_toggle.setChecked(False)
        self.center_toggle.clicked.connect(self.communicator.toggle_auto_center)

        m.get_root().header.add_child(
            folium.Element('<script type="text/javascript" src="qrc:///qtwebchannel/qwebchannel.js"></script>')
        )
        js_code = f"""
        <script>
        new QWebChannel(qt.webChannelTransport, function(channel) {{
            var communicator = channel.objects.communicator;
            {map_var}.on('click', function(e) {{
                var lat = e.latlng.lat.toFixed(6);
                var lng = e.latlng.lng.toFixed(6);
                communicator.update_coords('Latitude: ' + lat + '\\nLongitude: ' + lng);
            }});
        }});
        </script>
        """
        m.get_root().html.add_child(folium.Element(js_code))

        # ── Inject pre-plot survey layer ──
        if preplot_points:
            # CSS for permanent station labels
            bg_css = f'background: {sl_bg_color};' if sl_bg_show else 'background: transparent;'
            m.get_root().html.add_child(folium.Element(
                f'<style>.preplot-label {{ font-size: {sl_size}px; font-weight: bold; '
                f'{bg_css} color: {sl_text_color}; border: none; '
                f'padding: 1px 4px; border-radius: 3px; }}</style>'
            ))

            pts_json = json.dumps(
                [[round(lat, 6), round(lon, 6), stn] for lat, lon, stn in preplot_points]
            )

            preplot_js = f"""
            <script>
            setTimeout(function() {{
                try {{
                    var pp = {pts_json};
                    var n = pp.length;

                    // Polyline always visible through all points
                    var latlngs = pp.map(function(p) {{ return [p[0], p[1]]; }});
                    L.polyline(latlngs, {{color: '{sv_color}', weight: {sv_width}, opacity: 0.8}}).addTo({map_var});

                    // Canvas renderer for fast drawing of large point sets
                    var renderer = L.canvas({{padding: 0.5}});
                    var mkLayer = L.layerGroup().addTo({map_var});

                    function addMarker(i, showLabel) {{
                        var isStation = (i % 10 === 0);
                        var mk = L.circleMarker([pp[i][0], pp[i][1]], {{
                            renderer:    renderer,
                            color:       isStation ? '{sm_color}' : '{sp_color}',
                            fillColor:   isStation ? '{sm_color}' : '{sp_color}',
                            fillOpacity: 0.85,
                            radius:      isStation ? {sm_radius} : {sp_radius},
                            weight: 1
                        }}).bindPopup('Station: ' + pp[i][2]);
                        if (isStation && showLabel && {sl_show}) {{
                            mk.bindTooltip(pp[i][2], {{
                                permanent: true, direction: 'right',
                                offset: [5, 0], className: 'preplot-label'
                            }});
                        }}
                        mk.addTo(mkLayer);
                    }}

                    function redraw() {{
                        var z = {map_var}.getZoom();
                        mkLayer.clearLayers();
                        if (z < 7) return;
                        var showLabel = (z >= 7);
                        var step = (z < 14) ? Math.max(1, Math.floor(n / 50))
                                 : (z < 16) ? Math.max(1, Math.floor(n / 200))
                                 : 1;
                        for (var i = 0; i < n; i += step) {{
                            addMarker(i, showLabel);
                        }}
                    }}

                    // Only redraw on zoom change — no moveend, so labels stay stable during pan
                    {map_var}.on('zoomend', redraw);
                    redraw();
                }} catch(e) {{
                    console.error('[preplot] JS error: ' + e);
                }}
            }}, 500);
            </script>
            """
            m.get_root().html.add_child(folium.Element(preplot_js))

        # ── Inject saved post markers ─────────────────────────────────────────
        posts_csv = project_posts_csv(project_name)
        if os.path.exists(posts_csv):
            cs = load_post_settings()
            cv_shape  = cs.get('shape', 'Circle')
            cv_color  = cs.get('color', 'red')
            cv_radius = {'Small': 6, 'Medium': 10, 'Large': 14}.get(cs.get('size', 'Medium'), 10)
            post_entries = []
            try:
                with open(posts_csv, newline='') as f:
                    for row in csv.DictReader(f):
                        try:
                            post_entries.append((
                                float(row['latitude']), float(row['longitude']),
                                row['name'], row['source_frame']
                            ))
                        except (KeyError, ValueError):
                            pass
            except Exception:
                pass
            if post_entries:
                markers_js = [
                    "window._postMarkers = {};",
                    f"window._postLayerGroup = L.layerGroup().addTo({map_var});",
                ]
                for clat, clon, cname, src_frame in post_entries:
                    safe = cname.replace("'", "\\'")
                    safe_frame = src_frame.replace("'", "\\'")
                    popup = f"'{safe}<br>{clat:.5f}, {clon:.5f}'"
                    if cv_shape == 'Circle':
                        stmt = (f"window._postMarkers['{safe_frame}'] = "
                                f"L.circleMarker([{clat},{clon}],"
                                f"{{radius:{cv_radius},color:'white',weight:1.5,"
                                f"fillColor:'{cv_color}',fillOpacity:0.9}})"
                                f".bindPopup({popup}).addTo(window._postLayerGroup);")
                    else:
                        sz = cv_radius * 2
                        if cv_shape == 'Square':
                            shape_svg = f'<rect width="{sz}" height="{sz}" fill="{cv_color}" stroke="white" stroke-width="1.5"/>'
                        else:
                            pts = f"0,{sz} {sz},{sz} {sz//2},0"
                            shape_svg = f'<polygon points="{pts}" fill="{cv_color}" stroke="white" stroke-width="1.5"/>'
                        svg = f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg">{shape_svg}</svg>'
                        stmt = (f"window._postMarkers['{safe_frame}'] = "
                                f"L.marker([{clat},{clon}],{{icon:L.divIcon({{"
                                f"html:'{svg}',iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                                f").bindPopup({popup}).addTo(window._postLayerGroup);")
                    markers_js.append(stmt)

                post_js = (
                    "<script>\nsetTimeout(function() {\n"
                    + "\n".join(markers_js)
                    + "\n}, 600);\n</script>"
                )
                m.get_root().html.add_child(folium.Element(post_js))

        # ── Inject saved culvert markers ──────────────────────────────────────
        culverts_csv = project_culverts_csv(project_name)
        if os.path.exists(culverts_csv):
            cs = load_culvert_settings()
            cv_color  = cs.get('color', 'cyan')
            cv_radius = {'Small': 6, 'Medium': 10, 'Large': 14}.get(cs.get('size', 'Medium'), 10)
            cv_shape  = cs.get('shape', 'Circle')
            culvert_entries = []
            try:
                with open(culverts_csv, newline='') as f:
                    for row in csv.DictReader(f):
                        try:
                            culvert_entries.append((
                                float(row['latitude']), float(row['longitude']),
                                row['source_frame']
                            ))
                        except (KeyError, ValueError):
                            pass
            except Exception:
                pass
            if culvert_entries:
                markers_js = [
                    "window._culvertMarkers = {};",
                    f"window._culvertLayerGroup = L.layerGroup().addTo({map_var});",
                ]
                for clat, clon, src_frame in culvert_entries:
                    safe_frame = src_frame.replace("'", "\\'")
                    popup = f"'culvert<br>{clat:.5f}, {clon:.5f}'"
                    if cv_shape == 'Circle':
                        stmt = (f"window._culvertMarkers['{safe_frame}'] = "
                                f"L.circleMarker([{clat},{clon}],"
                                f"{{radius:{cv_radius},color:'white',weight:2,"
                                f"fillColor:'{cv_color}',fillOpacity:0.9}})"
                                f".bindPopup({popup}).addTo(window._culvertLayerGroup);")
                    else:
                        sz = cv_radius * 2
                        if cv_shape == 'Square':
                            shape_svg = f'<rect width="{sz}" height="{sz}" fill="{cv_color}" stroke="white" stroke-width="2"/>'
                        else:  # Diamond
                            h = sz // 2
                            shape_svg = f'<polygon points="{h},0 {sz},{h} {h},{sz} 0,{h}" fill="{cv_color}" stroke="white" stroke-width="2"/>'
                        svg = f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg">{shape_svg}</svg>'
                        stmt = (f"window._culvertMarkers['{safe_frame}'] = "
                                f"L.marker([{clat},{clon}],{{icon:L.divIcon({{"
                                f"html:'{svg}',iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                                f").bindPopup({popup}).addTo(window._culvertLayerGroup);")
                    markers_js.append(stmt)

                culvert_js = (
                    "<script>\nsetTimeout(function() {\n"
                    + "\n".join(markers_js)
                    + "\n}, 700);\n</script>"
                )
                m.get_root().html.add_child(folium.Element(culvert_js))

        # Write to temp file to avoid setHtml() size limits
        if self._tmp_html and os.path.exists(self._tmp_html):
            try:
                os.unlink(self._tmp_html)
            except Exception:
                pass
        out = BytesIO()
        m.save(out, close_file=False)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(out.getvalue().decode('utf-8'))
            self._tmp_html = f.name
        self.web_view.setUrl(QUrl.fromLocalFile(self._tmp_html))

        if preplot_points:
            self.coords_label.setText(
                f"Click on the map to get coordinates  |  Survey: {len(preplot_points)} points loaded"
            )
        else:
            self.coords_label.setText("Click on the map to get coordinates")

    def _read_points(self, txt_path):
        grouped_points = defaultdict(list)
        points_with_files = []
        with open(txt_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    filename = parts[0]
                    lat = float(parts[1])
                    lon = float(parts[2])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and lat != -180 and lon != -180):
                        continue
                    if '_frame_' in filename:
                        section_name = filename.split('_frame_')[0]
                        grouped_points[section_name].append((lat, lon))
                        points_with_files.append((lat, lon, filename))
                except ValueError:
                    pass
        if not grouped_points:
            raise ValueError("No valid GPS points found.")
        return grouped_points, points_with_files


# ─── Main Window ──────────────────────────────────────────────────────────────

# ─── Post Review Panel ────────────────────────────────────────────────────────

class PostThumb(QLabel):
    """Clickable thumbnail — click to toggle selected for removal."""
    toggled = pyqtSignal(str, bool)   # saved_image filename, is_selected

    def __init__(self, img_path, row, thumb_size=150):
        super().__init__()
        self.img_file = row.get('saved_image', '')
        self.row = row
        self._selected = False
        self._thumb_size = thumb_size

        px = QPixmap(img_path)
        if not px.isNull():
            self.setPixmap(px.scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.setText("(missing)")
            self.setAlignment(Qt.AlignCenter)

        self.setFixedSize(thumb_size + 10, thumb_size + 30)
        self.setAlignment(Qt.AlignCenter)
        src = row.get('source_frame', '')
        self.setToolTip(f"{self.img_file}\n{src}")

        # Small label under image
        self._lbl = QLabel(src[-30:] if len(src) > 30 else src, self)
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet("font-size: 9px; color: #aaa;")
        self._lbl.setGeometry(0, thumb_size + 4, thumb_size + 10, 20)

        self._apply_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected = not self._selected
            self._apply_style()
            self.toggled.emit(self.img_file, self._selected)

    def _apply_style(self):
        if self._selected:
            self.setStyleSheet("border: 3px solid #ff4444; background: rgba(255,50,50,40);")
        else:
            self.setStyleSheet("border: 2px solid #555; background: #2b2b2b;")


class ReviewPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._project_name = None
        self._selected = set()   # set of saved_image filenames marked for removal
        self._thumbs = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Post Image Review"))
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        top.addWidget(self.refresh_btn)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setStyleSheet("color: white; background: #993333;")
        self.remove_btn.clicked.connect(self._remove_selected)
        top.addWidget(self.remove_btn)
        self.count_lbl = QLabel("")
        top.addWidget(self.count_lbl)
        top.addStretch()
        layout.addLayout(top)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(6)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll)

    def load_project(self, project_name):
        self._project_name = project_name
        self._refresh()

    def _refresh(self):
        # Clear grid
        self._thumbs.clear()
        self._selected.clear()
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._project_name:
            return

        csv_path  = project_posts_csv(self._project_name)
        posts_dir = os.path.join(PROJECTS_DIR, self._project_name, "posts")

        if not os.path.exists(csv_path):
            self.count_lbl.setText("No posts.csv found.")
            return

        try:
            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            self.count_lbl.setText(f"Error reading CSV: {e}")
            return

        COLS = 12
        for i, row in enumerate(rows):
            img_path = os.path.join(posts_dir, row.get('saved_image', ''))
            thumb = PostThumb(img_path, row)
            thumb.toggled.connect(self._on_thumb_toggled)
            self.grid.addWidget(thumb, i // COLS, i % COLS)
            self._thumbs.append(thumb)

        self.count_lbl.setText(f"{len(rows)} images")

    def _on_thumb_toggled(self, img_file, selected):
        if selected:
            self._selected.add(img_file)
        else:
            self._selected.discard(img_file)
        n = len(self._selected)
        total = len(self._thumbs)
        self.count_lbl.setText(f"{total} images  |  {n} selected" if n else f"{total} images")

    def _remove_selected(self):
        if not self._selected:
            return
        reply = QMessageBox.question(
            self, "Remove Selected",
            f"Delete {len(self._selected)} image(s) from disk and posts.csv?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        posts_dir = os.path.join(PROJECTS_DIR, self._project_name, "posts")
        csv_path  = project_posts_csv(self._project_name)

        # Delete image files
        for img_file in self._selected:
            path = os.path.join(posts_dir, img_file)
            if os.path.exists(path):
                os.remove(path)

        # Rewrite CSV without removed rows
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                kept = [r for r in reader if r.get('saved_image') not in self._selected]
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept)

        self._refresh()


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dash Cam Viewer")
        screen = QDesktopWidget().screenGeometry()
        self.resize(int(screen.width() * 0.85), int(screen.height() * 0.85))
        self.setMinimumSize(900, 650)
        self._center()

        self.global_settings = load_global_settings()
        self.current_project = None
        self._build_ui()
        self._refresh_projects()
        last = self.global_settings.get("last_project", "")
        if last:
            idx = self.project_combo.findText(last)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        top_bar = QWidget()
        top_bar.setFixedHeight(72)
        top_bar.setStyleSheet("background:#2b2b2b;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 4, 12, 4)

        project_lbl = QLabel("Project:")
        project_lbl.setStyleSheet("color:white; font-size:20px;")
        top_layout.addWidget(project_lbl)

        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(280)
        self.project_combo.setFixedHeight(44)
        self.project_combo.setStyleSheet("font-size:26px; padding: 2px 8px; color: black; background: white;")
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        self.project_combo.activated.connect(lambda idx: self._on_project_changed(self.project_combo.currentText()))
        top_layout.addWidget(self.project_combo)

        for label, slot in [("New", self._new_project), ("Delete", self._delete_project)]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.setStyleSheet("font-size:18px; padding: 0 10px; color: white; background: #555555; border: 1px solid #777;")
            btn.clicked.connect(slot)
            top_layout.addWidget(btn)

        top_layout.addStretch()
        root.addWidget(top_bar)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(120)
        sidebar.setStyleSheet("background:#3c3f41;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 12, 0, 0)
        sidebar_layout.setSpacing(2)

        btn_style = """
            QPushButton {
                background: transparent; color: #bbbbbb;
                border: none; font-size: 23px; padding: 10px 0;
            }
            QPushButton:hover { background: #4c5052; color: white; }
            QPushButton:checked {
                background: #4c5052; color: white;
                border-left: 3px solid #4a9eff;
            }
        """
        self.process_btn = QPushButton("Process")
        self.view_btn_side = QPushButton("View")
        self.review_btn = QPushButton("Review")
        for btn in [self.process_btn, self.view_btn_side, self.review_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            sidebar_layout.addWidget(btn)
        sidebar_layout.addStretch()
        body_layout.addWidget(sidebar)

        self.stack = QStackedWidget()

        self.process_panel = ProcessPanel(
            lambda: self.current_project,
            self.global_settings,
            save_global_settings,
        )
        self.process_panel.processing_done.connect(self._switch_to_view)

        self.view_panel = ViewPanel()
        self.review_panel = ReviewPanel()

        self.stack.addWidget(self.process_panel)  # index 0
        self.stack.addWidget(self.view_panel)      # index 1
        self.stack.addWidget(self.review_panel)    # index 2
        body_layout.addWidget(self.stack, stretch=1)
        root.addWidget(body, stretch=1)

        self.process_btn.clicked.connect(lambda: self._switch_panel(0))
        self.view_btn_side.clicked.connect(self._switch_to_view)
        self.review_btn.clicked.connect(self._switch_to_review)
        self._switch_panel(0)

    def _switch_panel(self, index):
        self.stack.setCurrentIndex(index)
        self.process_btn.setChecked(index == 0)
        self.view_btn_side.setChecked(index == 1)
        self.review_btn.setChecked(index == 2)

    def _switch_to_view(self):
        if self.current_project:
            self.view_panel.load_project(self.current_project["name"])
        self._switch_panel(1)

    def _switch_to_review(self):
        if self.current_project:
            self.review_panel.load_project(self.current_project["name"])
        self._switch_panel(2)

    def _refresh_projects(self):
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        self.project_combo.addItem("-- Select Project --")
        for name in list_projects():
            self.project_combo.addItem(name)
        self.project_combo.blockSignals(False)

    def _on_project_changed(self, name):
        if not name or name == "-- Select Project --":
            self.current_project = None
            return
        try:
            self.current_project = load_project(name)
            self.process_panel.load_project(self.current_project)
            self.global_settings["last_project"] = name
            save_global_settings(self.global_settings)
            if self.stack.currentIndex() == 1:
                self.view_panel.load_project(name)
            elif self.stack.currentIndex() == 2:
                self.review_panel.load_project(name)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load project: {e}")

    def _new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if os.path.exists(os.path.join(PROJECTS_DIR, name)):
            QMessageBox.warning(self, "Error", f"Project '{name}' already exists.")
            return
        save_project(name, {
            "name": name,
            "video_source": "",
            "frame_interval": 1,
            "jpeg_quality": 92,
            "preplot_csv": "",
            "mga_zone": 55,
            "dc_color": "#ffff00",
            "dc_width": 5,
            "sv_color": "#00ccff",
            "sv_width": 2,
            "sp_color": "#00ccff",
            "sp_radius": 4,
            "sm_color": "#ff3300",
            "sm_radius": 6,
            "cm_color": "#ffff00",
            "cm_radius": 10,
            "sl_show": True,
            "sl_size": 11,
            "sl_text_color": "#ffffff",
            "sl_bg_show": True,
            "sl_bg_color": "#333333",
            "base_map": "satellite",
        })
        self._refresh_projects()
        idx = self.project_combo.findText(name)
        if idx >= 0:
            self.project_combo.setCurrentIndex(idx)

    def _delete_project(self):
        if not self.current_project:
            QMessageBox.information(self, "No Project", "No project selected.")
            return
        name = self.current_project["name"]
        reply = QMessageBox.question(
            self, "Delete Project",
            f"Delete '{name}'? This will permanently remove all frames and data.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            shutil.rmtree(os.path.join(PROJECTS_DIR, name), ignore_errors=True)
            self.current_project = None
            self._refresh_projects()

    def _center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
