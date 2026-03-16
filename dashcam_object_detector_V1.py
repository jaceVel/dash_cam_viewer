#!/usr/bin/env python3
"""
Dashcam Object Detector V1
PyQt5 desktop app: GPS frame viewer + YOLO object detection training pipeline.
Five panels: Process | View | Label | Models | Review
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
import tempfile
from datetime import timedelta, datetime
from pathlib import Path
from collections import defaultdict
from io import BytesIO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QMessageBox,
    QDesktopWidget, QSplitter, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QFileDialog, QTextEdit,
    QStackedWidget, QInputDialog, QFormLayout, QGroupBox, QCheckBox,
    QColorDialog, QRubberBand, QScrollArea, QGridLayout, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy, QButtonGroup, QRadioButton,
    QTabWidget, QDialog, QDialogButtonBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import (
    pyqtSlot, QObject, Qt, QTimer, QThread, pyqtSignal,
    QUrl, QRect, QPoint, QSize
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPainter, QPen

import folium

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECTS_DIR        = str(Path(__file__).parent / "dashcam_projects")
MODELS_DIR          = os.path.join(PROJECTS_DIR, "models")
TRAINING_SETS_DIR   = os.path.join(PROJECTS_DIR, "training_sets")
CLASSES_PATH        = os.path.join(PROJECTS_DIR, "classes.json")
GLOBAL_SETTINGS_PATH = os.path.join(PROJECTS_DIR, "settings.json")
FPS_FALLBACK        = 30.0


def jpeg_size(path):
    """Read JPEG width/height from file header without loading pixel data."""
    try:
        with open(path, 'rb') as f:
            f.read(2)                       # skip SOI
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    break
                code = marker[1]
                if 0xC0 <= code <= 0xC3:    # SOF0-SOF3
                    f.read(3)               # length + precision
                    h = int.from_bytes(f.read(2), 'big')
                    w = int.from_bytes(f.read(2), 'big')
                    return w, h
                length = int.from_bytes(f.read(2), 'big')
                f.seek(length - 2, 1)
    except Exception:
        pass
    return None


# ─── Global Settings ──────────────────────────────────────────────────────────

def load_global_settings():
    if os.path.exists(GLOBAL_SETTINGS_PATH):
        with open(GLOBAL_SETTINGS_PATH) as f:
            return json.load(f)
    return {"exiftool_path": "", "last_project": ""}


def save_global_settings(s):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(GLOBAL_SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)

# ─── Class Registry ───────────────────────────────────────────────────────────

def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            return json.load(f)
    return []


def save_classes(classes):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)


def add_class(name, color='#ff4444'):
    classes = load_classes()
    new_id = max((c['id'] for c in classes), default=-1) + 1
    classes.append({"id": new_id, "name": name, "color": color})
    save_classes(classes)
    return new_id

# ─── Project Helpers ──────────────────────────────────────────────────────────

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
    with open(cfg, encoding='utf-8') as f:
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


def project_labels_dir(name):
    return os.path.join(PROJECTS_DIR, name, "labels")


def project_split_path(name):
    return os.path.join(PROJECTS_DIR, name, "split.json")


def project_detections_csv(name):
    return os.path.join(PROJECTS_DIR, name, "detections.csv")


def project_false_negatives_csv(name):
    return os.path.join(PROJECTS_DIR, name, "false_negatives.csv")

# ─── Label Helpers ────────────────────────────────────────────────────────────

def load_label(project_name, frame_filename):
    """Returns list of (class_id, cx, cy, w, h) floats."""
    labels_dir = project_labels_dir(project_name)
    stem = os.path.splitext(frame_filename)[0]
    txt = os.path.join(labels_dir, stem + ".txt")
    if not os.path.exists(txt):
        return []
    boxes = []
    with open(txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
                except ValueError:
                    pass
    return boxes


def save_label(project_name, frame_filename, boxes):
    """boxes = [(class_id, cx, cy, w, h), ...]"""
    labels_dir = project_labels_dir(project_name)
    os.makedirs(labels_dir, exist_ok=True)
    stem = os.path.splitext(frame_filename)[0]
    txt = os.path.join(labels_dir, stem + ".txt")
    with open(txt, "w") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def delete_label(project_name, frame_filename):
    labels_dir = project_labels_dir(project_name)
    stem = os.path.splitext(frame_filename)[0]
    txt = os.path.join(labels_dir, stem + ".txt")
    if os.path.exists(txt):
        os.remove(txt)

# ─── Split Helpers ────────────────────────────────────────────────────────────

def load_split(project_name):
    p = project_split_path(project_name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"train": [], "val": []}


def save_split(project_name, split):
    with open(project_split_path(project_name), "w") as f:
        json.dump(split, f, indent=2)

# ─── Model Helpers ────────────────────────────────────────────────────────────

def load_model_config(model_name):
    cfg = os.path.join(MODELS_DIR, model_name, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return json.load(f)
    return {}


def save_model_config(model_name, config):
    d = os.path.join(MODELS_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def list_models():
    if not os.path.exists(MODELS_DIR):
        return []
    models = []
    for name in sorted(os.listdir(MODELS_DIR)):
        cfg = os.path.join(MODELS_DIR, name, "config.json")
        if os.path.isfile(cfg):
            models.append(name)
    return models


def model_weights_path(model_name):
    return os.path.join(MODELS_DIR, model_name, "weights", "best.pt")


def model_dataset_dir(model_name):
    return os.path.join(MODELS_DIR, model_name, "dataset")


# ─── Training Set Helpers ──────────────────────────────────────────────────────

def training_set_dir(set_id):
    return os.path.join(TRAINING_SETS_DIR, set_id)


def training_set_labels_dir(set_id):
    return os.path.join(TRAINING_SETS_DIR, set_id, "labels")


def load_training_set_meta(set_id):
    p = os.path.join(TRAINING_SETS_DIR, set_id, "meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_training_set_meta(set_id, meta):
    d = training_set_dir(set_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_tset_split(set_id):
    p = os.path.join(training_set_dir(set_id), "split.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"train": [], "val": []}


def save_tset_split(set_id, split):
    d = training_set_dir(set_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "split.json"), "w") as f:
        json.dump(split, f, indent=2)


def list_training_sets():
    if not os.path.isdir(TRAINING_SETS_DIR):
        return []
    result = []
    for name in sorted(os.listdir(TRAINING_SETS_DIR), reverse=True):
        if os.path.exists(os.path.join(TRAINING_SETS_DIR, name, "meta.json")):
            result.append(name)
    return result


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

# ─── Processing Worker ────────────────────────────────────────────────────────

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

            cv2.setNumThreads(1)

            log_queue = _queue.Queue()
            all_entries = []
            total_saved = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_map = {executor.submit(self._extract_video, mp4, prefix, log_queue): mp4
                              for mp4, prefix in jobs}
                pending = set(future_map)

                while pending:
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

                while True:
                    try:
                        self.log.emit(log_queue.get_nowait())
                    except _queue.Empty:
                        break

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


# ─── Train Script (subprocess) ────────────────────────────────────────────────

_TRAIN_SCRIPT = r"""
import sys, os, json, shutil, yaml

params_file = sys.argv[1]
with open(params_file) as f:
    p = json.load(f)

sources     = p['sources']
class_ids   = p['class_ids']
class_names = p['class_names']
dataset_dir = p['dataset_dir']
model_dir   = p['model_dir']
base_model  = p['base_model']
epochs      = p['epochs']

# Build id remap: global class_id -> 0-based index
id_to_idx = {cid: i for i, cid in enumerate(class_ids)}

img_train = os.path.join(dataset_dir, 'images', 'train')
lbl_train = os.path.join(dataset_dir, 'labels', 'train')
img_val   = os.path.join(dataset_dir, 'images', 'val')
lbl_val   = os.path.join(dataset_dir, 'labels', 'val')
for d in [img_train, lbl_train, img_val, lbl_val]:
    os.makedirs(d, exist_ok=True)

total_train = 0
total_val   = 0

for src in sources:
    frames_dir   = src['frames_dir']
    labels_dir   = src['labels_dir']
    train_frames = src.get('train_frames', [])
    val_frames   = src.get('val_frames', [])
    fn_frames    = src.get('false_neg_frames', [])

    def copy_frame(fname, img_dst, lbl_dst, is_hard_neg=False):
        src_img = os.path.join(frames_dir, fname)
        if not os.path.exists(src_img):
            return
        shutil.copy2(src_img, os.path.join(img_dst, fname))
        stem = os.path.splitext(fname)[0]
        src_lbl = os.path.join(labels_dir, stem + '.txt')
        dst_lbl = os.path.join(lbl_dst, stem + '.txt')
        if is_hard_neg or not os.path.exists(src_lbl):
            open(dst_lbl, 'w').close()
            return
        # Filter and remap labels
        lines = []
        with open(src_lbl) as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    if cid in id_to_idx:
                        lines.append(f"{id_to_idx[cid]} {' '.join(parts[1:])}\n")
        with open(dst_lbl, 'w') as lf:
            lf.writelines(lines)

    for fname in train_frames:
        copy_frame(fname, img_train, lbl_train)
        total_train += 1
    for fname in val_frames:
        copy_frame(fname, img_val, lbl_val)
        total_val += 1
    for fname in fn_frames:
        copy_frame(fname, img_train, lbl_train, is_hard_neg=True)
        total_train += 1

print(f"Dataset: {total_train} train, {total_val} val frames", flush=True)

yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
with open(yaml_path, 'w') as yf:
    yf.write(f"path: {dataset_dir}\n")
    yf.write("train: images/train\n")
    yf.write("val: images/val\n")
    yf.write(f"nc: {len(class_names)}\n")
    yf.write(f"names: {class_names}\n")

print(f"Training with base: {base_model}, epochs: {epochs}", flush=True)

from ultralytics import YOLO
model = YOLO(base_model)
model.train(data=yaml_path, epochs=epochs, imgsz=640,
            project=model_dir, name='.', exist_ok=True,
            workers=0, cache=False, device=0)
print("TRAINING_DONE", flush=True)
"""

# ─── Detect Script (subprocess) ───────────────────────────────────────────────

_DETECT_SCRIPT = r"""
import sys, os, json, csv

params_file = sys.argv[1]
with open(params_file) as f:
    p = json.load(f)

frames_dir   = p['frames_dir']
model_path   = p['model_path']
conf         = float(p['conf'])
class_filter = p.get('class_filter', None)   # model class indices to keep, None=all
class_names  = p['class_names']
coord_map    = p['coord_map']
already      = set(p.get('already', []))

from ultralytics import YOLO
model  = YOLO(model_path)
frames = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg'))
todo   = [f for f in frames if f not in already]
total  = len(todo)

for idx, fname in enumerate(todo):
    pct = int((idx + 1) / total * 100) if total else 100
    print(f"PROGRESS:{pct}:{idx+1}/{total} — {fname}", flush=True)
    fpath = os.path.join(frames_dir, fname)
    results = model(fpath, conf=conf, verbose=False, stream=True, half=True, device=0)
    lat, lon = coord_map.get(fname, [0.0, 0.0])
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            if class_filter is not None and cls_idx not in class_filter:
                continue
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cname = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
            det = {
                "frame": fname, "class_id": cls_idx, "class_name": cname,
                "conf": round(conf_score, 4),
                "box_x": x1, "box_y": y1,
                "box_w": x2 - x1, "box_h": y2 - y1,
                "lat": lat, "lon": lon
            }
            print("DETECTION:" + json.dumps(det), flush=True)

print("DETECT_DONE", flush=True)
"""


# ─── YOLO Train Worker ────────────────────────────────────────────────────────

class YoloTrainWorker(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            config = load_model_config(self.model_name)
            if not config:
                self.log.emit(f"No config found for model '{self.model_name}'")
                self.finished.emit(False)
                return

            class_ids     = config.get('classes', [])
            epochs        = config.get('epochs', 50)
            train_set_ids = config.get('training_sets', [])
            fixed_val     = config.get('fixed_val', False)

            classes = load_classes()
            class_names = [next((c['name'] for c in classes if c['id'] == cid), str(cid))
                           for cid in class_ids]

            import random as _random
            sources = []
            for set_id in train_set_ids:
                meta = load_training_set_meta(set_id)
                if not meta:
                    self.log.emit(f"  Skipping '{set_id}': meta not found")
                    continue

                proj       = meta.get('project', '')
                frames_dir = project_frames_dir(proj)
                lbl_dir    = training_set_labels_dir(set_id)
                tp_frames  = meta.get('tp_frames', [])
                fp_frames  = meta.get('fp_frames', [])

                if not os.path.isdir(frames_dir):
                    self.log.emit(f"  Skipping '{set_id}': frames dir missing for project '{proj}'")
                    continue
                if not tp_frames:
                    self.log.emit(f"  Skipping '{set_id}': no TP frames")
                    continue

                # 80/20 split of TP frames
                saved_split = load_tset_split(set_id)
                if fixed_val and saved_split.get('val'):
                    val_frames   = [f for f in saved_split['val'] if f in tp_frames]
                    already      = set(saved_split.get('train', [])) | set(val_frames)
                    new_frames   = [f for f in tp_frames if f not in already]
                    train_frames = [f for f in saved_split.get('train', []) if f in tp_frames] + new_frames
                    if new_frames:
                        save_tset_split(set_id, {"train": train_frames, "val": val_frames})
                else:
                    shuffled = tp_frames[:]
                    _random.shuffle(shuffled)
                    cut = max(1, round(len(shuffled) * 0.8))
                    train_frames = shuffled[:cut]
                    val_frames   = shuffled[cut:] or shuffled[:1]
                    if fixed_val:
                        save_tset_split(set_id, {"train": train_frames, "val": val_frames})

                self.log.emit(f"  Set '{set_id}' ({meta.get('source_type','?')}): "
                              f"{len(train_frames)} train / {len(val_frames)} val / "
                              f"{len(fp_frames)} hard-neg")

                sources.append({
                    "frames_dir":       frames_dir,
                    "labels_dir":       lbl_dir,
                    "train_frames":     train_frames,
                    "val_frames":       val_frames,
                    "false_neg_frames": fp_frames,
                })

            if not sources:
                self.log.emit("No valid training sets selected. Create training sets first.")
                self.finished.emit(False)
                return

            mwp       = model_weights_path(self.model_name)
            model_dir = os.path.join(MODELS_DIR, self.model_name)
            dataset_dir = model_dataset_dir(self.model_name)
            base = mwp if os.path.exists(mwp) else 'yolov8n.pt'

            params = {
                "sources":      sources,
                "class_ids":    class_ids,
                "class_names":  class_names,
                "dataset_dir":  dataset_dir,
                "model_dir":    model_dir,
                "base_model":   base,
                "epochs":       epochs,
            }

            params_file = os.path.join(tempfile.gettempdir(), 'dcv_v1_train_params.json')
            with open(params_file, 'w') as f:
                json.dump(params, f)

            script_file = os.path.join(tempfile.gettempdir(), 'dcv_v1_train_script.py')
            with open(script_file, 'w', encoding='utf-8') as sf:
                sf.write(_TRAIN_SCRIPT)

            self.log.emit(f"Training '{self.model_name}' | base: {os.path.basename(base)} | "
                          f"epochs: {epochs} | classes: {class_names}")

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
                config['trained_date'] = datetime.now().strftime("%Y-%m-%d")
                save_model_config(self.model_name, config)
                self.log.emit(f"Training complete. Weights: {mwp}")
                self.finished.emit(True)
            else:
                self.log.emit(f"Training subprocess exited with code {proc.returncode}")
                self.finished.emit(False)

        except Exception as e:
            self.log.emit(f"Training error: {e}")
            self.finished.emit(False)


# ─── YOLO Detect Worker ───────────────────────────────────────────────────────

class YoloDetectWorker(QThread):
    log            = pyqtSignal(str)
    progress       = pyqtSignal(int, str)
    finished       = pyqtSignal(bool)
    new_detections = pyqtSignal(list)

    def __init__(self, project_name, model_name, confidence, points_with_files):
        super().__init__()
        self.project_name      = project_name
        self.model_name        = model_name
        self.confidence        = confidence
        self.points_with_files = points_with_files

    def run(self):
        try:
            mwp = model_weights_path(self.model_name)
            if not os.path.exists(mwp):
                self.log.emit(f"No weights found for '{self.model_name}'. Train first.")
                self.finished.emit(False)
                return

            config       = load_model_config(self.model_name)
            class_ids    = config.get('classes', [])
            classes      = load_classes()
            class_names  = [next((c['name'] for c in classes if c['id'] == cid), str(cid))
                            for cid in class_ids]
            class_filter = list(range(len(class_ids)))  # model outputs 0-based indices

            frames_dir = project_frames_dir(self.project_name)
            det_csv    = project_detections_csv(self.project_name)

            # Fresh run — remove previous detections so view and review are clean
            if os.path.exists(det_csv):
                os.remove(det_csv)

            coord_map = {fname: [lat, lon] for lat, lon, fname in self.points_with_files}

            params = {
                "frames_dir":   frames_dir,
                "model_path":   mwp,
                "conf":         self.confidence,
                "class_filter": class_filter,
                "class_names":  class_names,
                "coord_map":    coord_map,
                "already":      [],
            }

            params_file = os.path.join(tempfile.gettempdir(), 'dcv_v1_detect_params.json')
            with open(params_file, 'w') as f:
                json.dump(params, f)

            script_file = os.path.join(tempfile.gettempdir(), 'dcv_v1_detect_script.py')
            with open(script_file, 'w', encoding='utf-8') as sf:
                sf.write(_DETECT_SCRIPT)

            self.log.emit(f"Detecting with '{self.model_name}' (conf={self.confidence:.2f})...")

            proc = subprocess.Popen(
                [sys.executable, script_file, params_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='replace',
            )

            new_rows = []
            write_header = not os.path.exists(det_csv)
            fieldnames = ["frame", "class_id", "class_name", "conf",
                          "box_x", "box_y", "box_w", "box_h", "lat", "lon",
                          "model_name", "reviewed", "result"]

            csv_f = open(det_csv, 'a', newline='')
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("DETECTION:"):
                    try:
                        det = json.loads(line[10:])
                        det['model_name'] = self.model_name
                        det['reviewed']   = ''
                        det['result']     = ''
                        writer.writerow(det)
                        csv_f.flush()
                        new_rows.append(det)
                    except Exception:
                        pass
                elif line.startswith("PROGRESS:"):
                    parts = line.split(":", 2)
                    pct = int(parts[1]) if len(parts) >= 2 else 0
                    msg = parts[2] if len(parts) == 3 else ""
                    self.progress.emit(pct, msg)
                    self.log.emit(f"  {pct}% {msg}")
                elif line:
                    self.log.emit(line)

            csv_f.close()
            proc.wait()
            self.log.emit(f"Detection complete. {len(new_rows)} new detection(s).")
            self.new_detections.emit(new_rows)
            self.finished.emit(proc.returncode == 0)

        except Exception as e:
            self.log.emit(f"Detect error: {e}")
            self.finished.emit(False)


# ─── Box Frame Label ──────────────────────────────────────────────────────────

class BoxFrameLabel(QLabel):
    """QLabel with box-drawing and overlay display. Per-class colors supported."""

    box_drawn            = pyqtSignal(int, int, int, int)   # x1,y1,x2,y2 orig coords
    hard_negative_marked = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._boxes        = []   # manual: list of (x1,y1,x2,y2,class_id)
        self._detect_boxes = []   # yolo: list of {'rect':(x1,y1,x2,y2),'neg':bool,'cls_name':str,'conf':float}
        self._class_colors = {}   # class_id -> QColor
        self._orig_size    = None
        self._draw_mode    = False
        self._selecting    = False
        self._origin       = QPoint()
        self._rb           = QRubberBand(QRubberBand.Rectangle, self)
        self.setContextMenuPolicy(Qt.PreventContextMenu)

    def set_draw_mode(self, enabled):
        self._draw_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_orig_size(self, orig_w, orig_h):
        self._orig_size = (orig_w, orig_h)

    def set_class_colors(self, color_map):
        """color_map: dict of class_id -> hex color string."""
        self._class_colors = {k: QColor(v) for k, v in color_map.items()}

    def clear_boxes(self):
        self._boxes = []
        self._detect_boxes = []
        self.update()

    def set_detection_boxes(self, boxes):
        """boxes = list of (x1,y1,x2,y2) or dicts with extra info."""
        self._detect_boxes = []
        for b in boxes:
            if isinstance(b, dict):
                self._detect_boxes.append({
                    'rect': (b['box_x'], b['box_y'],
                             b['box_x'] + b['box_w'], b['box_y'] + b['box_h']),
                    'neg': False,
                    'cls_name': b.get('class_name', ''),
                    'conf': b.get('conf', 0.0),
                })
            else:
                x1, y1, x2, y2 = b
                self._detect_boxes.append({'rect': (x1, y1, x2, y2), 'neg': False,
                                           'cls_name': '', 'conf': 0.0})
        self.update()

    def _widget_to_orig(self, wx, wy):
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

    def _orig_rect_to_widget(self, x1, y1, x2, y2):
        if not self._orig_size:
            return QRect(x1, y1, x2 - x1, y2 - y1)
        orig_w, orig_h = self._orig_size
        lw, lh = self.width(), self.height()
        scale = min(lw / orig_w, lh / orig_h)
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        ox, oy = (lw - sw) // 2, (lh - sh) // 2
        return QRect(ox + int(x1 * scale), oy + int(y1 * scale),
                     int((x2 - x1) * scale), int((y2 - y1) * scale))

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self._detect_boxes:
            pos = event.pos()
            for i, db in enumerate(self._detect_boxes):
                if db['neg']:
                    continue
                x1, y1, x2, y2 = db['rect']
                if self._orig_rect_to_widget(x1, y1, x2, y2).contains(pos):
                    self._detect_boxes[i]['neg'] = True
                    self.update()
                    self.hard_negative_marked.emit(x1, y1, x2, y2)
                    return
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
                    self._boxes.append((x1, y1, x2, y2, -1))
                    self.update()
                    self.box_drawn.emit(x1, y1, x2, y2)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._orig_size:
            return
        orig_w, orig_h = self._orig_size
        lw, lh = self.width(), self.height()
        if lw <= 0 or lh <= 0:
            return
        scale = min(lw / orig_w, lh / orig_h)
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        ox, oy = (lw - sw) // 2, (lh - sh) // 2
        p = QPainter(self)

        # Manual boxes — per-class color or blue
        if self._boxes:
            for (x1, y1, x2, y2, cls_id) in self._boxes:
                color = self._class_colors.get(cls_id, QColor(50, 120, 255))
                p.setPen(QPen(color, 2))
                p.drawRect(ox + int(x1 * scale), oy + int(y1 * scale),
                           int((x2 - x1) * scale), int((y2 - y1) * scale))

        # Detection boxes — green or red (hard negative)
        for db in self._detect_boxes:
            x1, y1, x2, y2 = db['rect']
            p.setPen(QPen(QColor(220, 50, 50) if db['neg'] else QColor(50, 200, 80), 2))
            p.drawRect(ox + int(x1 * scale), oy + int(y1 * scale),
                       int((x2 - x1) * scale), int((y2 - y1) * scale))
            if db.get('cls_name'):
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.drawText(ox + int(x1 * scale) + 2, oy + int(y1 * scale) - 2,
                           f"{db['cls_name']} {db['conf']:.2f}")


# ─── Communicator ─────────────────────────────────────────────────────────────

class Communicator(QObject):
    frame_changed = pyqtSignal(int)

    def __init__(self, coords_label, image_label, points_with_files, web_view,
                 map_var, spin_box, image_dir, marker_color='yellow', marker_radius=10):
        super().__init__()
        self.coords_label      = coords_label
        self.image_label       = image_label
        self.points_with_files = sorted(points_with_files, key=lambda x: x[2])
        self.image_dir         = image_dir
        self.current_index     = None
        self.web_view          = web_view
        self.map_var           = map_var
        self.marker_color      = marker_color
        self.marker_radius     = marker_radius
        self.auto_center_enabled = False
        self.spin_box          = spin_box
        self.prev_timer        = QTimer()
        self.prev_timer.timeout.connect(self.show_prev)
        self.next_timer        = QTimer()
        self.next_timer.timeout.connect(self.show_next)
        self._road_cache       = {}
        self._road_pending     = None
        self._road_reply       = None
        self._nam              = QNetworkAccessManager()
        self._road_timer       = QTimer()
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
            except (IndexError, ValueError):
                self.coords_label.setText(coords)

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
            self.web_view.page().runJavaScript(f"{self.map_var}.panTo([{lat}, {lon}]);")
        if road is None:
            self._road_pending = (lat, lon, fname)
            self._road_timer.start(500)

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
        request.setRawHeader(b'User-Agent', b'DashCamViewer/V1')
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
            return
        if self.current_index <= 0:
            self.prev_timer.stop()
            return
        self.current_index -= 1
        self._show_frame(self.current_index)

    @pyqtSlot()
    def show_next(self):
        if self.current_index is None:
            return
        if self.current_index >= len(self.points_with_files) - 1:
            self.next_timer.stop()
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
            color: '{self.marker_color}', fillColor: '{self.marker_color}',
            fillOpacity: 0.8, radius: {self.marker_radius}
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


# ─── Shared Map Helpers ───────────────────────────────────────────────────────

def _read_points(txt_path):
    """Parse frames_latlon.txt. Returns (grouped_points, points_with_files)."""
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
                section_name = filename.split('_frame_')[0] if '_frame_' in filename else 'default'
                grouped_points[section_name].append((lat, lon))
                points_with_files.append((lat, lon, filename))
            except ValueError:
                pass
    if not grouped_points:
        raise ValueError("No valid GPS points found.")
    return grouped_points, points_with_files


def build_folium_map(project_name, project_data, points_with_files):
    """Build and return a folium.Map for the given project."""
    grouped_points = defaultdict(list)
    for lat, lon, fname in points_with_files:
        section = fname.split('_frame_')[0] if '_frame_' in fname else 'default'
        grouped_points[section].append((lat, lon))

    all_points = [p for pts in grouped_points.values() for p in pts]
    if not all_points:
        return None

    center_lat = statistics.mean(lat for lat, lon in all_points)
    center_lon = statistics.mean(lon for lat, lon in all_points)
    base_map   = project_data.get('base_map', 'satellite')
    dc_color   = project_data.get('dc_color', '#ffff00')
    dc_width   = project_data.get('dc_width', 5)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)

    if base_map == 'osm':
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap',
                         overlay=False, control=True).add_to(m)
    elif base_map == 'topo':
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles © Esri', name='Topo Map', overlay=False, control=True
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles © Esri', name='Satellite', overlay=False, control=True
        ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles © Esri', name='Roads', overlay=True, control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles © Esri', name='Places', overlay=True, control=True
    ).add_to(m)

    for section_name, points in grouped_points.items():
        if len(points) > 1:
            folium.PolyLine(
                locations=[(lat, lon) for lat, lon in points],
                color=dc_color, weight=dc_width, opacity=0.8,
                popup=f"Section: {section_name}"
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _inject_webchannel(m, map_var):
    """Add QWebChannel JS bridge to folium map."""
    m.get_root().header.add_child(
        folium.Element('<script type="text/javascript" src="qrc:///qtwebchannel/qwebchannel.js"></script>')
    )
    js_code = (
        "<script>\n"
        "new QWebChannel(qt.webChannelTransport, function(channel) {\n"
        "    var communicator = channel.objects.communicator;\n"
        f"    {map_var}.on('click', function(e) {{\n"
        "        var lat = e.latlng.lat.toFixed(6);\n"
        "        var lng = e.latlng.lng.toFixed(6);\n"
        "        communicator.update_coords('Latitude: ' + lat + '\\nLongitude: ' + lng);\n"
        "    });\n"
        "});\n"
        "</script>\n"
    )
    m.get_root().html.add_child(folium.Element(js_code))


def _save_map_to_tempfile(m, old_path=None):
    """Save folium map to a temp HTML file. Returns new path."""
    if old_path and os.path.exists(old_path):
        try:
            os.unlink(old_path)
        except Exception:
            pass
    out = BytesIO()
    m.save(out, close_file=False)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(out.getvalue().decode('utf-8'))
        return f.name


# ─── Process Panel ────────────────────────────────────────────────────────────

class ProcessPanel(QWidget):
    processing_done = pyqtSignal()

    def __init__(self, get_project_fn, global_settings, save_settings_fn):
        super().__init__()
        self.get_project     = get_project_fn
        self.global_settings = global_settings
        self.save_settings   = save_settings_fn
        self.worker          = None
        self._loading        = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

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
        default_idx = self.zone_combo.findData(55)
        if default_idx >= 0:
            self.zone_combo.setCurrentIndex(default_idx)
        survey_form.addRow("MGA Zone:", self.zone_combo)
        layout.addWidget(survey_group)

        self.preplot_edit.editingFinished.connect(self._save_survey_settings)
        self.zone_combo.currentIndexChanged.connect(self._save_survey_settings)

        display_group = QGroupBox("Map Display")
        disp_form = QFormLayout(display_group)
        disp_form.setSpacing(8)

        def _color_row(default_color, spin_label, default_val):
            row = QHBoxLayout()
            btn = ProcessPanel._mk_color_btn(default_color)
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

        self.basemap_combo = QComboBox()
        self.basemap_combo.addItem("Satellite", "satellite")
        self.basemap_combo.addItem("OpenStreetMap", "osm")
        self.basemap_combo.addItem("Topo", "topo")
        disp_form.addRow("Base map:", self.basemap_combo)
        layout.addWidget(display_group)

        for w in [self.dc_width_spin, self.sv_width_spin, self.sp_radius_spin, self.sm_radius_spin]:
            w.valueChanged.connect(self._save_display_settings)
        self.basemap_combo.currentIndexChanged.connect(self._save_display_settings)

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

        layout.addWidget(QLabel("Log:"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_edit, stretch=1)

    def load_project(self, project_name):
        self._loading = True
        try:
            project_data = load_project(project_name)
        except Exception:
            project_data = {}

        self.src_edit.setText(project_data.get("video_source", ""))
        self.interval_spin.setValue(float(project_data.get("frame_interval", 1.0)))
        self.quality_spin.setValue(project_data.get("jpeg_quality", 92))
        self.preplot_edit.setText(project_data.get("preplot_csv", ""))

        zone = project_data.get("mga_zone", 55)
        idx = self.zone_combo.findData(zone)
        if idx >= 0:
            self.zone_combo.setCurrentIndex(idx)

        def _sc(btn, key, default):
            c = project_data.get(key, default)
            btn.setStyleSheet(f"background: {c}; border: 1px solid #aaa;")
            btn.setProperty('hex_color', c)

        _sc(self.dc_color_btn, 'dc_color', '#ffff00')
        self.dc_width_spin.setValue(project_data.get('dc_width', 5))
        _sc(self.sv_color_btn, 'sv_color', '#00ccff')
        self.sv_width_spin.setValue(project_data.get('sv_width', 2))
        _sc(self.sp_color_btn, 'sp_color', '#00ccff')
        self.sp_radius_spin.setValue(project_data.get('sp_radius', 4))
        _sc(self.sm_color_btn, 'sm_color', '#ff3300')
        self.sm_radius_spin.setValue(project_data.get('sm_radius', 6))

        bm_idx = self.basemap_combo.findData(project_data.get('base_map', 'satellite'))
        if bm_idx >= 0:
            self.basemap_combo.setCurrentIndex(bm_idx)

        self._loading = False
        self.log_edit.clear()

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
            'base_map':  self.basemap_combo.currentData(),
        })
        save_project(project['name'], project)

    def _save_survey_settings(self):
        if self._loading:
            return
        project = self.get_project()
        if not project:
            return
        project["preplot_csv"] = self.preplot_edit.text().strip()
        project["mga_zone"]    = self.zone_combo.currentData()
        save_project(project["name"], project)

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

    def _run(self):
        project = self.get_project()
        print(f"DEBUG _run: project={project}")
        if not project:
            QMessageBox.warning(self, "No Project", "Please select or create a project first.")
            return
        video_dir = self.src_edit.text().strip()
        exiftool  = self.exif_edit.text().strip()
        if not video_dir or not os.path.isdir(video_dir):
            QMessageBox.warning(self, "Error", "Please select a valid video folder.")
            return
        if not exiftool or not os.path.isfile(exiftool):
            QMessageBox.warning(self, "Error", "Please select a valid ExifTool path.")
            return

        project["video_source"]   = video_dir
        project["frame_interval"] = self.interval_spin.value()
        project["jpeg_quality"]   = self.quality_spin.value()
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
        self.communicator  = None
        self.channel       = None
        self._tmp_html     = None
        self._project_name = None
        self._points       = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.coords_label = QLabel("Select a project, then click the map to view frames.")
        self.coords_label.setFont(QFont("Arial", 11))
        self.coords_label.setAlignment(Qt.AlignCenter)
        self.coords_label.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.coords_label)

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter, stretch=1)

        self.map_container = QWidget()
        self.map_container.setMinimumWidth(400)
        self.map_layout = QVBoxLayout(self.map_container)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = None
        self.splitter.addWidget(self.map_container)

        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = BoxFrameLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label, stretch=1)

        nav_container = QWidget()
        nav_container.setFixedHeight(52)
        nav_container.setStyleSheet("background-color: #3c3f41; border-radius: 4px;")
        btn_row = QHBoxLayout(nav_container)
        btn_row.setContentsMargins(6, 6, 6, 6)
        btn_row.setSpacing(6)

        self.auto_left     = QPushButton("Auto ←")
        self.auto_left.setCheckable(True)
        self.left_btn      = QPushButton("← Prev")
        self.spin_box      = QSpinBox()
        self.spin_box.setRange(100, 5000)
        self.spin_box.setValue(1000)
        self.spin_box.setSingleStep(100)
        self.spin_box.setSuffix(" ms")
        self.right_btn     = QPushButton("Next →")
        self.auto_right    = QPushButton("Auto →")
        self.auto_right.setCheckable(True)
        self.center_toggle = QPushButton("Auto Center")
        self.center_toggle.setCheckable(True)

        _btn_style = (
            "QPushButton {"
            "  padding: 4px 14px; min-height: 28px;"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5d5f,stop:1 #3c3f41);"
            "  border: 1px solid #777; border-radius: 4px; color: #eeeeee;"
            "}"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6e7173,stop:1 #4c5052); }"
            "QPushButton:pressed { background: #2b2b2b; border-color: #999; }"
            "QPushButton:checked { background: #4a9eff; color: white; border-color: #2277cc; }"
        )
        for w in [self.auto_left, self.left_btn, self.right_btn,
                  self.auto_right, self.center_toggle]:
            w.setStyleSheet(_btn_style)

        btn_row.addStretch()
        for w in [self.auto_left, self.left_btn, self.spin_box,
                  self.right_btn, self.auto_right, self.center_toggle]:
            btn_row.addWidget(w)
        btn_row.addStretch()
        image_layout.addWidget(nav_container)

        self.splitter.addWidget(image_widget)
        self.splitter.setSizes([500, 900])

    def load_project(self, project_name):
        import traceback
        try:
            self._load_inner(project_name)
        except Exception as e:
            self.coords_label.setText(f"View error: {e}")
            QMessageBox.critical(self, "View Error", traceback.format_exc())

    def _load_inner(self, project_name):
        if self.communicator:
            self.communicator.prev_timer.stop()
            self.communicator.next_timer.stop()
            self.communicator = None

        self.image_label.clear()
        self.image_label.clear_boxes()

        if self.web_view is not None:
            self.map_layout.removeWidget(self.web_view)
            self.web_view.setParent(None)
            self.web_view.deleteLater()
        self.web_view = QWebEngineView()
        self.map_layout.addWidget(self.web_view)
        self.splitter.setSizes([500, 900])

        self._project_name = project_name
        txt_path   = project_txt_path(project_name)
        frames_dir = project_frames_dir(project_name)

        if not os.path.exists(txt_path):
            self.coords_label.setText("No data found. Run processing first.")
            self.web_view.setHtml(
                "<h3 style='text-align:center;margin-top:60px;color:#888'>No map data</h3>"
            )
            return

        try:
            project_data = load_project(project_name)
        except Exception:
            project_data = {}

        _, points_with_files = _read_points(txt_path)
        self._points = points_with_files

        m = build_folium_map(project_name, project_data, points_with_files)
        if m is None:
            self.coords_label.setText("No GPS points found.")
            return

        map_var = m.get_name()

        preplot_points = []
        try:
            preplot_csv = project_data.get("preplot_csv", "")
            mga_zone    = project_data.get("mga_zone", 55)
            if preplot_csv and os.path.exists(preplot_csv):
                preplot_points = load_preplot_csv(preplot_csv, mga_zone)
        except Exception:
            pass

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

        for btn in [self.auto_left, self.left_btn, self.right_btn,
                    self.auto_right, self.center_toggle]:
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
        try:
            self.spin_box.valueChanged.disconnect()
        except TypeError:
            pass

        self.auto_left.setChecked(False)
        self.auto_right.setChecked(False)
        self.center_toggle.setChecked(False)

        self.auto_left.clicked.connect(self.communicator.toggle_auto_prev)
        self.auto_left.clicked.connect(
            lambda: self.auto_right.setChecked(self.communicator.next_timer.isActive()))
        self.left_btn.clicked.connect(self.communicator.show_prev)
        self.right_btn.clicked.connect(self.communicator.show_next)
        self.auto_right.clicked.connect(self.communicator.toggle_auto_next)
        self.auto_right.clicked.connect(
            lambda: self.auto_left.setChecked(self.communicator.prev_timer.isActive()))
        self.spin_box.valueChanged.connect(self.communicator.update_interval)
        self.center_toggle.clicked.connect(self.communicator.toggle_auto_center)

        _inject_webchannel(m, map_var)

        if preplot_points:
            sv_color  = project_data.get('sv_color', '#00ccff')
            sv_width  = project_data.get('sv_width', 2)
            sp_color  = project_data.get('sp_color', '#00ccff')
            sp_radius = project_data.get('sp_radius', 4)
            sm_color  = project_data.get('sm_color', '#ff3300')
            sm_radius = project_data.get('sm_radius', 6)
            pts_json = json.dumps(
                [[round(lat, 6), round(lon, 6), stn] for lat, lon, stn in preplot_points]
            )
            preplot_js = (
                "<script>\nsetTimeout(function() {\n"
                "  try {\n"
                f"    var pp = {pts_json};\n"
                "    var latlngs = pp.map(function(p) { return [p[0], p[1]]; });\n"
                f"    L.polyline(latlngs, {{color: '{sv_color}', weight: {sv_width}, opacity: 0.8}}).addTo({map_var});\n"
                "    var renderer = L.canvas({padding: 0.5});\n"
                "    for (var i = 0; i < pp.length; i++) {\n"
                "      var isStation = (i % 10 === 0);\n"
                f"      L.circleMarker([pp[i][0], pp[i][1]], {{renderer: renderer,\n"
                f"        color: isStation ? '{sm_color}' : '{sp_color}',\n"
                f"        fillColor: isStation ? '{sm_color}' : '{sp_color}',\n"
                f"        fillOpacity: 0.85, radius: isStation ? {sm_radius} : {sp_radius}, weight: 1\n"
                f"      }}).bindPopup('Station: ' + pp[i][2]).addTo({map_var});\n"
                "    }\n"
                "  } catch(e) { console.error('[preplot] ' + e); }\n"
                "}, 500);\n"
                "</script>\n"
            )
            m.get_root().html.add_child(folium.Element(preplot_js))

        self._tmp_html = _save_map_to_tempfile(m, self._tmp_html)
        self.web_view.setUrl(QUrl.fromLocalFile(self._tmp_html))
        self.coords_label.setText("Click on the map to view frames")

    def _on_frame_changed(self, index):
        if not self._project_name or not self.communicator:
            return
        pts = self.communicator.points_with_files
        if index >= len(pts):
            return
        det_csv = project_detections_csv(self._project_name)
        if not os.path.exists(det_csv):
            self.image_label.clear_boxes()
            return
        _, _, fname = pts[index]
        boxes = []
        try:
            with open(det_csv, newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('frame') == fname:
                        boxes.append({
                            'box_x': int(row['box_x']), 'box_y': int(row['box_y']),
                            'box_w': int(row['box_w']), 'box_h': int(row['box_h']),
                            'class_name': row.get('class_name', ''),
                            'conf': float(row.get('conf', 0)),
                        })
        except Exception:
            pass
        self.image_label.clear_boxes()
        self.image_label.set_detection_boxes(boxes)


# ─── Label Panel ──────────────────────────────────────────────────────────────

class LabelPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._project_name   = None
        self._frames         = []      # sorted list of filenames
        self._current_index  = 0
        self._current_boxes  = []      # [(class_id, cx, cy, w, h), ...]
        self._classes        = []      # [{id, name, color}, ...]
        self._communicator   = None
        self._channel        = None
        self._tmp_html       = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Top toolbar ──
        top = QHBoxLayout()
        top.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(140)
        top.addWidget(self.class_combo)

        add_cls_btn = QPushButton("Add Class")
        add_cls_btn.clicked.connect(self._add_class_dialog)
        top.addWidget(add_cls_btn)

        top.addSpacing(16)

        self.prev_unlabeled_btn = QPushButton("← Prev Unlabeled")
        self.prev_unlabeled_btn.clicked.connect(self._prev_unlabeled)
        top.addWidget(self.prev_unlabeled_btn)

        self.next_unlabeled_btn = QPushButton("Next Unlabeled →")
        self.next_unlabeled_btn.clicked.connect(self._next_unlabeled)
        top.addWidget(self.next_unlabeled_btn)

        self.prev_labeled_btn = QPushButton("← Prev Labeled")
        self.prev_labeled_btn.clicked.connect(self._prev_labeled)
        top.addWidget(self.prev_labeled_btn)

        self.next_labeled_btn = QPushButton("Next Labeled →")
        self.next_labeled_btn.clicked.connect(self._next_labeled)
        top.addWidget(self.next_labeled_btn)

        top.addStretch()

        save_set_btn = QPushButton("💾 Save as Training Set")
        save_set_btn.setToolTip("Snapshot current labels as a named training set for use in Models tab")
        save_set_btn.clicked.connect(self._save_as_training_set)
        top.addWidget(save_set_btn)

        outer.addLayout(top)

        # ── Second toolbar: nav + split ──
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("← Prev")
        self.prev_btn.clicked.connect(self._prev_frame)
        nav.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._next_frame)
        nav.addWidget(self.next_btn)

        nav.addSpacing(16)
        self.frame_lbl = QLabel("Frame: -/-")
        nav.addWidget(self.frame_lbl)

        nav.addStretch()
        outer.addLayout(nav)

        # ── Main area: map + image+labels ──
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # Left: small map
        self.map_container = QWidget()
        self.map_container.setMinimumWidth(280)
        self.map_container.setMaximumWidth(400)
        self.map_layout = QVBoxLayout(self.map_container)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        self.map_web = None
        splitter.addWidget(self.map_container)

        # Right: image + labels list
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self.image_label = BoxFrameLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.set_draw_mode(True)
        self.image_label.box_drawn.connect(self._on_box_drawn)
        right_layout.addWidget(self.image_label, stretch=1)

        right_layout.addWidget(QLabel("Labels on this frame:"))
        self.labels_list = QListWidget()
        self.labels_list.setMaximumHeight(120)
        right_layout.addWidget(self.labels_list)

        del_btn = QPushButton("Delete Selected Label")
        del_btn.clicked.connect(self._delete_selected_label)
        right_layout.addWidget(del_btn)

        splitter.addWidget(right)
        splitter.setSizes([300, 900])

        # ── Status bar ──
        self.stats_lbl = QLabel("No project loaded")
        self.stats_lbl.setStyleSheet("font-size: 21px; color: #cccccc; padding: 6px 10px; background: #2b2b2b; border-top: 1px solid #555;")
        outer.addWidget(self.stats_lbl)

    def load_project(self, project_name):
        self._project_name = project_name
        self._classes      = load_classes()
        self._refresh_class_combo()

        frames_dir = project_frames_dir(project_name)
        if os.path.isdir(frames_dir):
            self._frames = sorted(
                f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')
            )
        else:
            self._frames = []

        self._current_index = 0
        self._load_map()
        if self._frames:
            self._show_frame(0)
        self._update_stats()

    def _refresh_class_combo(self):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for c in self._classes:
            self.class_combo.addItem(c['name'], c['id'])
        self.class_combo.blockSignals(False)

    def _load_map(self):
        if self.map_web is not None:
            self.map_layout.removeWidget(self.map_web)
            self.map_web.setParent(None)
            self.map_web.deleteLater()
        self.map_web = QWebEngineView()
        self.map_layout.addWidget(self.map_web)

        txt_path = project_txt_path(self._project_name)
        if not os.path.exists(txt_path):
            self.map_web.setHtml(
                "<h3 style='text-align:center;margin-top:60px;color:#888'>No map data</h3>"
            )
            return

        try:
            project_data = load_project(self._project_name)
        except Exception:
            project_data = {}

        try:
            _, points_with_files = _read_points(txt_path)
        except Exception:
            return

        m = build_folium_map(self._project_name, project_data, points_with_files)
        if m is None:
            return

        map_var = m.get_name()

        self._channel = QWebChannel()
        self.map_web.page().setWebChannel(self._channel)

        frames_dir = project_frames_dir(self._project_name)
        self._communicator = Communicator(
            QLabel(), self.image_label,
            points_with_files, self.map_web,
            map_var, QSpinBox(), frames_dir,
        )
        self._communicator.frame_changed.connect(self._on_map_click)
        self._channel.registerObject('communicator', self._communicator)
        _inject_webchannel(m, map_var)

        self._tmp_html = _save_map_to_tempfile(m, self._tmp_html)
        self.map_web.setUrl(QUrl.fromLocalFile(self._tmp_html))

    def _on_map_click(self, index):
        if index < len(self._frames):
            # communicator's points_with_files may be in different order;
            # find matching frame
            pwf = self._communicator.points_with_files
            if index < len(pwf):
                fname = pwf[index][2]
                try:
                    fi = self._frames.index(fname)
                    self._current_index = fi
                    self._show_frame(fi)
                except ValueError:
                    pass

    def _show_frame(self, index):
        if not self._frames or index < 0 or index >= len(self._frames):
            return
        self._current_index = index
        fname      = self._frames[index]
        frames_dir = project_frames_dir(self._project_name)
        img_path   = os.path.join(frames_dir, fname)

        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            self.image_label.set_orig_size(pixmap.width(), pixmap.height())
            sz = self.image_label.size()
            if sz.width() > 0 and sz.height() > 0:
                pixmap = pixmap.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

        # Load and show existing boxes
        self._current_boxes = load_label(self._project_name, fname)
        self._refresh_boxes_overlay()
        self._refresh_labels_list()

        # Frame counter
        self.frame_lbl.setText(f"Frame: {index+1}/{len(self._frames)}  {fname}")


    def _refresh_boxes_overlay(self):
        """Update BoxFrameLabel with current boxes using per-class colors."""
        color_map = {c['id']: c['color'] for c in self._classes}
        self.image_label.set_class_colors(color_map)
        # Rebuild manual boxes list in image_label
        self.image_label._boxes = []
        for cls_id, cx, cy, w, h in self._current_boxes:
            # Convert normalized coords back to pixel for display
            orig = self.image_label._orig_size
            if orig:
                ow, oh = orig
                x1 = int((cx - w / 2) * ow)
                y1 = int((cy - h / 2) * oh)
                x2 = int((cx + w / 2) * ow)
                y2 = int((cy + h / 2) * oh)
                self.image_label._boxes.append((x1, y1, x2, y2, cls_id))
        self.image_label.update()

    def _refresh_labels_list(self):
        self.labels_list.clear()
        classes_by_id = {c['id']: c['name'] for c in self._classes}
        for i, (cls_id, cx, cy, w, h) in enumerate(self._current_boxes):
            cls_name = classes_by_id.get(cls_id, str(cls_id))
            item = QListWidgetItem(
                f"[{i}] {cls_name}  cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
            )
            self.labels_list.addItem(item)

    def _on_box_drawn(self, x1, y1, x2, y2):
        if not self._project_name or not self._frames:
            return
        orig = self.image_label._orig_size
        if not orig:
            return
        ow, oh = orig
        cx = (x1 + x2) / 2 / ow
        cy = (y1 + y2) / 2 / oh
        w  = (x2 - x1) / ow
        h  = (y2 - y1) / oh
        cls_id = self.class_combo.currentData()
        if cls_id is None:
            cls_id = 0
        self._current_boxes.append((cls_id, cx, cy, w, h))
        fname = self._frames[self._current_index]
        save_label(self._project_name, fname, self._current_boxes)
        self._refresh_boxes_overlay()
        self._refresh_labels_list()
        self._update_stats()

    def _delete_selected_label(self):
        row = self.labels_list.currentRow()
        if row < 0 or row >= len(self._current_boxes):
            return
        self._current_boxes.pop(row)
        fname = self._frames[self._current_index]
        if self._current_boxes:
            save_label(self._project_name, fname, self._current_boxes)
        else:
            delete_label(self._project_name, fname)
        self._refresh_boxes_overlay()
        self._refresh_labels_list()
        self._update_stats()


    def _save_as_training_set(self):
        if not self._project_name:
            QMessageBox.warning(self, "No Project", "No project loaded.")
            return
        labeled = self._labeled_indices()
        if not labeled:
            QMessageBox.warning(self, "No Labels", "No labeled frames in this project.")
            return

        set_id  = f"manual_{self._project_name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        lbl_dir = training_set_labels_dir(set_id)
        os.makedirs(lbl_dir, exist_ok=True)

        src_labels_dir = project_labels_dir(self._project_name)
        tp_frames = []
        for i in labeled:
            fname = self._frames[i]
            stem  = os.path.splitext(fname)[0]
            src   = os.path.join(src_labels_dir, stem + '.txt')
            dst   = os.path.join(lbl_dir, stem + '.txt')
            if os.path.exists(src):
                shutil.copy2(src, dst)
                tp_frames.append(fname)

        meta = {
            "name":        set_id,
            "created":     datetime.now().isoformat(),
            "source_type": "manual",
            "project":     self._project_name,
            "tp_frames":   tp_frames,
            "fp_frames":   [],
        }
        save_training_set_meta(set_id, meta)
        QMessageBox.information(self, "Training Set Saved",
            f"'{set_id}'\n{len(tp_frames)} labeled frames saved.\n\n"
            f"Select it in the Models tab to train from.")

    def _prev_frame(self):
        if self._frames and self._current_index > 0:
            self._show_frame(self._current_index - 1)

    def _next_frame(self):
        if self._frames and self._current_index < len(self._frames) - 1:
            self._show_frame(self._current_index + 1)

    def _labeled_indices(self):
        if not self._project_name:
            return []
        labels_dir = project_labels_dir(self._project_name)
        result = []
        for i, fname in enumerate(self._frames):
            stem = os.path.splitext(fname)[0]
            if os.path.exists(os.path.join(labels_dir, stem + '.txt')):
                result.append(i)
        return result

    def _prev_labeled(self):
        labeled = self._labeled_indices()
        before = [i for i in labeled if i < self._current_index]
        if before:
            self._show_frame(before[-1])

    def _next_labeled(self):
        labeled = self._labeled_indices()
        after = [i for i in labeled if i > self._current_index]
        if after:
            self._show_frame(after[0])

    def _prev_unlabeled(self):
        labeled = set(self._labeled_indices())
        for i in range(self._current_index - 1, -1, -1):
            if i not in labeled:
                self._show_frame(i)
                return

    def _next_unlabeled(self):
        labeled = set(self._labeled_indices())
        for i in range(self._current_index + 1, len(self._frames)):
            if i not in labeled:
                self._show_frame(i)
                return

    def _update_stats(self):
        if not self._project_name:
            return
        total     = len(self._frames)
        labeled   = len(self._labeled_indices())
        n_boxes   = len(self._current_boxes)
        unlabeled = total - labeled
        label_pct = round(labeled / total * 100) if total else 0
        auto_train = round(labeled * 0.8)
        auto_val   = labeled - auto_train
        self.stats_lbl.setText(
            f"Frames: {total}  |  "
            f"Labeled: {labeled} ({label_pct}%)  |  "
            f"Unlabeled: {unlabeled}  |  "
            f"Boxes this frame: {n_boxes}  |  "
            f"Auto-split at train: ~{auto_train} train / ~{auto_val} val  (80/20)"
        )

    def _add_class_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Class")
        dlg.setMinimumWidth(300)
        form = QFormLayout(dlg)
        name_edit = QLineEdit()
        form.addRow("Name:", name_edit)

        color_btn = QPushButton()
        color_btn.setFixedSize(50, 26)
        color_btn.setStyleSheet("background: #ff4444; border: 1px solid #aaa;")
        color_btn.setProperty('hex_color', '#ff4444')
        def pick():
            c = QColorDialog.getColor(QColor(color_btn.property('hex_color')), dlg)
            if c.isValid():
                color_btn.setStyleSheet(f"background: {c.name()}; border: 1px solid #aaa;")
                color_btn.setProperty('hex_color', c.name())
        color_btn.clicked.connect(pick)
        form.addRow("Color:", color_btn)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            name = name_edit.text().strip()
            if name:
                color = color_btn.property('hex_color')
                add_class(name, color)
                self._classes = load_classes()
                self._refresh_class_combo()


# ─── Models Panel ─────────────────────────────────────────────────────────────

class ModelsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._project_name  = None
        self._selected_model = None
        self._train_worker  = None
        self._detect_worker = None
        self._points        = []
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Left: model list
        left = QWidget()
        left.setFixedWidth(160)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("Models:"))

        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self._on_model_selected)
        left_layout.addWidget(self.model_list, stretch=1)

        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._new_model)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_model)
        row = QHBoxLayout()
        row.addWidget(new_btn)
        row.addWidget(del_btn)
        left_layout.addLayout(row)
        outer.addWidget(left)

        # Right: config + evaluate + log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # Config group
        config_group = QGroupBox("Model Config")
        config_form  = QFormLayout(config_group)
        self.name_edit = QLineEdit()
        self.name_edit.setReadOnly(True)
        config_form.addRow("Name:", self.name_edit)

        # Classes checkboxes (built dynamically)
        self.classes_widget = QWidget()
        self.classes_layout = QVBoxLayout(self.classes_widget)
        self.classes_layout.setContentsMargins(0, 0, 0, 0)
        self.classes_layout.setSpacing(2)
        config_form.addRow("Classes:", self.classes_widget)

        # Training sets checkboxes (built dynamically)
        self.sets_scroll = QScrollArea()
        self.sets_scroll.setWidgetResizable(True)
        self.sets_scroll.setMaximumHeight(180)
        self.sets_inner = QWidget()
        self.sets_layout = QVBoxLayout(self.sets_inner)
        self.sets_layout.setContentsMargins(0, 0, 0, 0)
        self.sets_layout.setSpacing(2)
        self.sets_scroll.setWidget(self.sets_inner)
        refresh_sets_btn = QPushButton("⟳ Refresh Sets")
        refresh_sets_btn.setFixedWidth(110)
        refresh_sets_btn.clicked.connect(self._refresh_sets_checkboxes)
        sets_row = QHBoxLayout()
        sets_row.addWidget(self.sets_scroll)
        sets_row.addWidget(refresh_sets_btn, alignment=Qt.AlignTop)
        sets_container = QWidget()
        sets_container.setLayout(sets_row)
        config_form.addRow("Training sets:", sets_container)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setFixedWidth(80)
        config_form.addRow("Epochs:", self.epochs_spin)

        self.fixed_val_chk = QCheckBox("Fixed val set (lock 20% val frames after first train)")
        self.fixed_val_chk.setToolTip(
            "When checked, the val split is saved after the first train run.\n"
            "Subsequent runs reuse the same val frames so model versions are comparable."
        )
        config_form.addRow("", self.fixed_val_chk)

        cfg_btns = QHBoxLayout()
        save_cfg_btn = QPushButton("Save Config")
        save_cfg_btn.clicked.connect(self._save_config)
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self._start_training)
        cfg_btns.addWidget(save_cfg_btn)
        cfg_btns.addWidget(self.train_btn)
        cfg_btns.addStretch()
        config_form.addRow("", cfg_btns)

        right_layout.addWidget(config_group)

        # Evaluate group
        eval_group  = QGroupBox("Evaluate")
        eval_layout = QFormLayout(eval_group)

        self.eval_project_combo = QComboBox()
        self.eval_project_combo.setMinimumWidth(200)
        eval_layout.addRow("Project:", self.eval_project_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setFixedWidth(80)
        eval_layout.addRow("Confidence:", self.conf_spin)

        run_eval_btn = QPushButton("Run Detect on Val Frames")
        run_eval_btn.clicked.connect(self._run_evaluate)
        eval_layout.addRow("", run_eval_btn)

        self.eval_result_lbl = QLabel("—")
        eval_layout.addRow("Results:", self.eval_result_lbl)

        right_layout.addWidget(eval_group)

        # Detection group
        detect_group  = QGroupBox("Detect on Project")
        detect_layout = QFormLayout(detect_group)

        self.detect_project_combo = QComboBox()
        self.detect_project_combo.setMinimumWidth(200)
        detect_layout.addRow("Project:", self.detect_project_combo)

        self.detect_conf_spin = QDoubleSpinBox()
        self.detect_conf_spin.setRange(0.05, 0.95)
        self.detect_conf_spin.setSingleStep(0.05)
        self.detect_conf_spin.setValue(0.25)
        self.detect_conf_spin.setDecimals(2)
        self.detect_conf_spin.setFixedWidth(80)
        detect_layout.addRow("Confidence:", self.detect_conf_spin)

        run_detect_btn = QPushButton("Run Detection")
        run_detect_btn.clicked.connect(self._run_detect)
        detect_layout.addRow("", run_detect_btn)

        right_layout.addWidget(detect_group)

        # Log
        right_layout.addWidget(QLabel("Log:"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        right_layout.addWidget(self.log_edit, stretch=1)

        outer.addWidget(right)

    def load_project(self, project_name):
        self._project_name = project_name
        self._load_points()
        self._refresh_model_list()
        self._refresh_sets_checkboxes()
        self._refresh_eval_combos()

    def _load_points(self):
        if not self._project_name:
            return
        txt = project_txt_path(self._project_name)
        if os.path.exists(txt):
            try:
                _, self._points = _read_points(txt)
            except Exception:
                self._points = []

    def _refresh_model_list(self):
        self.model_list.clear()
        for name in list_models():
            self.model_list.addItem(name)

    def _refresh_sets_checkboxes(self, selected_ids=None):
        while self.sets_layout.count():
            item = self.sets_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        selected_ids = selected_ids or []
        for set_id in list_training_sets():
            meta = load_training_set_meta(set_id)
            src  = meta.get('source_type', '?')
            proj = meta.get('project', '?')
            tp   = len(meta.get('tp_frames', []))
            fp   = len(meta.get('fp_frames', []))
            date = meta.get('created', '')[:10]
            label = f"[{src}] {set_id}  — {proj} · {tp} TP / {fp} FP · {date}"
            cb = QCheckBox(label)
            cb.setProperty('set_id', set_id)
            cb.setChecked(set_id in selected_ids)
            self.sets_layout.addWidget(cb)

    def _refresh_classes_checkboxes(self, selected_ids=None):
        while self.classes_layout.count():
            item = self.classes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        selected_ids = selected_ids or []
        for c in load_classes():
            cb = QCheckBox(f"{c['name']} (id={c['id']})")
            cb.setProperty('class_id', c['id'])
            cb.setChecked(c['id'] in selected_ids)
            self.classes_layout.addWidget(cb)

    def _refresh_eval_combos(self):
        projects = list_projects()
        for combo in [self.eval_project_combo, self.detect_project_combo]:
            combo.clear()
            for p in projects:
                combo.addItem(p)
            if self._project_name and self._project_name in projects:
                combo.setCurrentText(self._project_name)

    def _on_model_selected(self, item, _prev):
        if item is None:
            return
        self._selected_model = item.text()
        config = load_model_config(self._selected_model)
        self.name_edit.setText(self._selected_model)
        self.epochs_spin.setValue(config.get('epochs', 50))
        self.fixed_val_chk.setChecked(config.get('fixed_val', False))
        self._refresh_classes_checkboxes(config.get('classes', []))
        self._refresh_sets_checkboxes(config.get('training_sets', []))

    def _get_checked_class_ids(self):
        ids = []
        for i in range(self.classes_layout.count()):
            cb = self.classes_layout.itemAt(i).widget()
            if isinstance(cb, QCheckBox) and cb.isChecked():
                ids.append(cb.property('class_id'))
        return ids

    def _get_checked_sets(self):
        ids = []
        for i in range(self.sets_layout.count()):
            cb = self.sets_layout.itemAt(i).widget()
            if isinstance(cb, QCheckBox) and cb.isChecked():
                ids.append(cb.property('set_id'))
        return ids

    def _new_model(self):
        name, ok = QInputDialog.getText(self, "New Model", "Model name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        config = {
            "name":               name,
            "classes":            [],
            "training_sets":      [],
            "epochs":             50,
            "trained_date":       "",
            "val_map50":          0.0,
        }
        save_model_config(name, config)
        self._refresh_model_list()
        items = self.model_list.findItems(name, Qt.MatchExactly)
        if items:
            self.model_list.setCurrentItem(items[0])

    def _delete_model(self):
        if not self._selected_model:
            return
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete model '{self._selected_model}' and all its files?",
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if reply == QMessageBox.Yes:
            model_dir = os.path.join(MODELS_DIR, self._selected_model)
            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir)
            self._selected_model = None
            self._refresh_model_list()

    def _save_config(self):
        if not self._selected_model:
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return
        config = load_model_config(self._selected_model)
        config['classes']        = self._get_checked_class_ids()
        config['training_sets']  = self._get_checked_sets()
        config['epochs']         = self.epochs_spin.value()
        config['fixed_val']      = self.fixed_val_chk.isChecked()
        save_model_config(self._selected_model, config)
        self.log_edit.append(f"Config saved for '{self._selected_model}'")

    def _start_training(self):
        if not self._selected_model:
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return
        self._save_config()
        config = load_model_config(self._selected_model)
        if not config.get('classes'):
            QMessageBox.warning(self, "No Classes", "Select at least one class for this model.")
            return
        if not config.get('training_sets'):
            QMessageBox.warning(self, "No Training Sets",
                                "Select at least one training set.\n\n"
                                "Create sets from the Label tab (Save as Training Set) "
                                "or the Review tab after a detection run.")
            return

        self.train_btn.setEnabled(False)
        self.log_edit.append(f"Starting training for '{self._selected_model}'...")

        self._train_worker = YoloTrainWorker(self._selected_model)
        self._train_worker.log.connect(self.log_edit.append)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.start()

    def _on_train_finished(self, success):
        self.train_btn.setEnabled(True)
        self.log_edit.append("Training done." if success else "Training failed.")

    def _run_evaluate(self):
        """Run YOLO detect on val frames, compare to labels, report metrics."""
        if not self._selected_model:
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return
        proj = self.eval_project_combo.currentText()
        if not proj:
            return
        mwp = model_weights_path(self._selected_model)
        if not os.path.exists(mwp):
            QMessageBox.warning(self, "No Weights", "Model has not been trained yet.")
            return

        split = load_split(proj)
        val_frames = split.get('val', [])
        if not val_frames:
            self.eval_result_lbl.setText("No val frames assigned in split.json")
            return

        config      = load_model_config(self._selected_model)
        class_ids   = config.get('classes', [])
        classes     = load_classes()
        class_names = [next((c['name'] for c in classes if c['id'] == cid), str(cid))
                       for cid in class_ids]

        frames_dir  = project_frames_dir(proj)
        labels_dir  = project_labels_dir(proj)
        id_to_idx   = {cid: i for i, cid in enumerate(class_ids)}
        conf        = self.conf_spin.value()

        self.log_edit.append(f"Evaluating on {len(val_frames)} val frames...")
        self.eval_result_lbl.setText("Running...")
        QApplication.processEvents()

        # Build a coord_map (no GPS needed for eval, just use dummy)
        coord_map = {f: [0.0, 0.0] for f in val_frames}

        params = {
            "frames_dir":   frames_dir,
            "model_path":   mwp,
            "conf":         conf,
            "class_filter": list(range(len(class_ids))),
            "class_names":  class_names,
            "coord_map":    coord_map,
            "already":      [],
        }
        params_file = os.path.join(tempfile.gettempdir(), 'dcv_v1_eval_params.json')
        with open(params_file, 'w') as pf:
            json.dump(params, pf)

        script_file = os.path.join(tempfile.gettempdir(), 'dcv_v1_detect_script.py')
        with open(script_file, 'w', encoding='utf-8') as sf:
            sf.write(_DETECT_SCRIPT)

        proc = subprocess.Popen(
            [sys.executable, script_file, params_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace',
        )
        detections_by_frame = defaultdict(list)
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("DETECTION:"):
                try:
                    det = json.loads(line[10:])
                    detections_by_frame[det['frame']].append(det)
                except Exception:
                    pass
            elif line:
                self.log_edit.append(line)
        proc.wait()

        # Compare detections to ground truth
        tp = fp = fn = 0
        IOU_THRESH = 0.5

        def iou(b1, b2):
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        for fname in val_frames:
            gt_boxes = []
            for cls_id, cx, cy, w, h in load_label(proj, fname):
                if cls_id not in id_to_idx:
                    continue
                # Convert normalized to absolute for IoU (use 1x1 space)
                gt_boxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))

            preds = detections_by_frame.get(fname, [])
            pred_boxes = [(d['box_x'], d['box_y'],
                           d['box_x']+d['box_w'], d['box_y']+d['box_h'])
                          for d in preds]

            # Try to get image size to normalize pred boxes
            img_path = os.path.join(frames_dir, fname)
            try:
                from PIL import Image as _PIL
                with _PIL.open(img_path) as im:
                    iw, ih = im.size
                pred_boxes_norm = [(x1/iw, y1/ih, x2/iw, y2/ih)
                                   for x1, y1, x2, y2 in pred_boxes]
            except Exception:
                pred_boxes_norm = pred_boxes

            matched_gt  = set()
            matched_pred = set()
            for pi, pb in enumerate(pred_boxes_norm):
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    if iou(pb, gb) >= IOU_THRESH:
                        matched_gt.add(gi)
                        matched_pred.add(pi)
                        tp += 1
                        break
            fp += len(pred_boxes) - len(matched_pred)
            fn += len(gt_boxes)   - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        map50     = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        result_text = (f"Val frames: {len(val_frames)}  TP={tp} FP={fp} FN={fn}  "
                       f"Precision={precision:.2f}  Recall={recall:.2f}  mAP50~{map50:.2f}")
        self.eval_result_lbl.setText(result_text)
        self.log_edit.append(result_text)

        # Save val_map50 to config
        config['val_map50'] = round(map50, 4)
        save_model_config(self._selected_model, config)

    def _run_detect(self):
        if not self._selected_model:
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return
        proj = self.detect_project_combo.currentText()
        if not proj:
            return

        txt = project_txt_path(proj)
        if os.path.exists(txt):
            try:
                _, points = _read_points(txt)
            except Exception:
                points = []
        else:
            points = []

        conf = self.detect_conf_spin.value()
        self._detect_worker = YoloDetectWorker(proj, self._selected_model, conf, points)
        self._detect_worker.log.connect(self.log_edit.append)
        self._detect_worker.finished.connect(
            lambda ok: self.log_edit.append("Detection done." if ok else "Detection failed.")
        )
        self._detect_worker.start()


# ─── Review Panel ─────────────────────────────────────────────────────────────

class DetectionThumb(QLabel):
    """Clickable thumbnail showing a detection crop."""
    toggled = pyqtSignal(int, bool)   # row_index, is_selected

    def __init__(self, row_index, pixmap, class_name, conf, result, thumb_size=150):
        super().__init__()
        self._row_index = row_index
        self._selected  = False
        self._result    = result   # 'tp', 'fp', or ''

        if pixmap and not pixmap.isNull():
            self.setPixmap(
                pixmap.scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.setText("(no image)")
            self.setAlignment(Qt.AlignCenter)

        self.setFixedSize(thumb_size + 10, thumb_size + 36)
        self.setAlignment(Qt.AlignCenter)

        self._lbl = QLabel(f"{class_name}\n{conf:.2f}", self)
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet("font-size: 9px; color: #ddd;")
        self._lbl.setGeometry(0, thumb_size + 4, thumb_size + 10, 28)

        self._apply_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected = not self._selected
            self._apply_style()
            self.toggled.emit(self._row_index, self._selected)

    def _apply_style(self):
        if self._result == 'tp':
            border = "border: 3px solid #44cc44;"
        elif self._result == 'fp':
            border = "border: 3px solid #cc4444;"
        elif self._selected:
            border = "border: 3px solid #4499ff;"
        else:
            border = "border: 2px solid #555;"
        self.setStyleSheet(f"{border} background: #2b2b2b;")

    def set_result(self, result):
        self._result = result
        self._apply_style()


class ReviewPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._project_name = None
        self._detections   = []    # list of row dicts from detections.csv
        self._selected     = set() # row indices
        self._thumbs       = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Project:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self._on_project_combo_changed)
        top.addWidget(self.project_combo)

        top.addWidget(QLabel("Model:"))
        self.model_filter_combo = QComboBox()
        self.model_filter_combo.addItem("All models")
        self.model_filter_combo.setMinimumWidth(160)
        top.addWidget(self.model_filter_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        top.addWidget(refresh_btn)
        top.addStretch()
        layout.addLayout(top)

        action_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Unreviewed", "TP", "FP"])
        self.filter_combo.currentIndexChanged.connect(self._refresh)
        action_row.addWidget(QLabel("Filter:"))
        action_row.addWidget(self.filter_combo)

        review_btn = QPushButton("Reject Selected (FP)  /  Confirm Rest (TP) →")
        review_btn.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #662222,stop:0.5 #444,stop:1 #226622); color: white; padding: 4px 16px;"
        )
        review_btn.setToolTip(
            "Selected (highlighted) crops → marked FP + added as hard negatives\n"
            "Unselected crops → marked TP + label files written for retraining"
        )
        review_btn.clicked.connect(self._review_batch)
        action_row.addWidget(review_btn)

        self.count_lbl = QLabel("")
        action_row.addWidget(self.count_lbl)
        action_row.addStretch()
        layout.addLayout(action_row)

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
        # Refresh project combo
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for p in list_projects():
            self.project_combo.addItem(p)
        if project_name in list_projects():
            self.project_combo.setCurrentText(project_name)
        self.project_combo.blockSignals(False)

        # Refresh model filter
        self.model_filter_combo.clear()
        self.model_filter_combo.addItem("All models")
        for m in list_models():
            self.model_filter_combo.addItem(m)

        self._refresh()

    def _on_project_combo_changed(self, name):
        if name:
            self._project_name = name
            self._refresh()

    def _refresh(self):
        if not self._project_name:
            return

        det_csv = project_detections_csv(self._project_name)
        self._detections = []
        if os.path.exists(det_csv):
            try:
                with open(det_csv, newline='') as f:
                    self._detections = list(csv.DictReader(f))
            except Exception:
                pass

        # Apply model filter
        model_filter = self.model_filter_combo.currentText()
        if model_filter != "All models":
            self._detections = [d for d in self._detections
                                if d.get('model_name') == model_filter]

        # Apply result filter
        result_filter = self.filter_combo.currentText()
        if result_filter == "Unreviewed":
            self._detections = [d for d in self._detections if not d.get('reviewed')]
        elif result_filter == "TP":
            self._detections = [d for d in self._detections if d.get('result') == 'tp']
        elif result_filter == "FP":
            self._detections = [d for d in self._detections if d.get('result') == 'fp']

        self._selected.clear()
        self._build_grid()

    def _build_grid(self):
        # Clear old thumbs
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbs = []

        THUMB_SIZE = 150
        COLS = 6
        frames_dir = project_frames_dir(self._project_name)

        for i, det in enumerate(self._detections):
            fname = det.get('frame', '')
            img_path = os.path.join(frames_dir, fname)

            # Crop the detection box from the source frame
            crop_pixmap = None
            try:
                bx = int(det['box_x']); by = int(det['box_y'])
                bw = int(det['box_w']); bh = int(det['box_h'])
                if os.path.exists(img_path) and bw > 0 and bh > 0:
                    full = QPixmap(img_path)
                    if not full.isNull():
                        crop_pixmap = full.copy(QRect(bx, by, bw, bh))
            except (KeyError, ValueError):
                pass

            thumb = DetectionThumb(
                i, crop_pixmap,
                det.get('class_name', ''), float(det.get('conf', 0)),
                det.get('result', ''),
                THUMB_SIZE
            )
            thumb.toggled.connect(self._on_thumb_toggled)
            self.grid.addWidget(thumb, i // COLS, i % COLS)
            self._thumbs.append(thumb)

        self.count_lbl.setText(f"{len(self._detections)} detections")

    def _on_thumb_toggled(self, row_index, selected):
        if selected:
            self._selected.add(row_index)
        else:
            self._selected.discard(row_index)

    def _review_batch(self):
        """Selected = FP (hard neg), unselected = TP. Saves a timestamped training set."""
        if not self._detections:
            return

        all_indices = set(range(len(self._detections)))
        fp_indices  = set(self._selected)       # copy before _set_result clears self._selected
        tp_indices  = all_indices - fp_indices

        # ── Mark results in detections.csv ──────────────────────────────────
        if fp_indices:
            self._set_result(fp_indices, 'fp')
        if tp_indices:
            self._set_result(tp_indices, 'tp')

        # ── Build training set ───────────────────────────────────────────────
        classes    = load_classes()
        frames_dir = project_frames_dir(self._project_name)
        set_id     = f"review_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        lbl_dir    = training_set_labels_dir(set_id)
        os.makedirs(lbl_dir, exist_ok=True)

        # Collect FP frame names
        fp_frames = []
        for idx in fp_indices:
            frame = self._detections[idx].get('frame', '')
            if frame and frame not in fp_frames:
                fp_frames.append(frame)

        # Write YOLO label files for TP detections
        tp_frames = []
        _size_cache = {}
        for idx in tp_indices:
            det   = self._detections[idx]
            frame = det.get('frame', '')
            if not frame:
                continue

            if frame not in _size_cache:
                _size_cache[frame] = jpeg_size(os.path.join(frames_dir, frame))
            size = _size_cache[frame]
            if not size:
                continue
            iw, ih = size

            try:
                bx = int(det['box_x']); by = int(det['box_y'])
                bw = int(det['box_w']); bh = int(det['box_h'])
            except (KeyError, ValueError):
                continue

            cname = det.get('class_name', '')
            cid   = next((c['id'] for c in classes if c['name'] == cname), None)
            if cid is None:
                continue

            cx = (bx + bw / 2) / iw
            cy = (by + bh / 2) / ih
            nw = bw / iw
            nh = bh / ih

            stem     = os.path.splitext(frame)[0]
            lbl_path = os.path.join(lbl_dir, stem + '.txt')
            line     = f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            # Append to label file (multiple detections per frame)
            with open(lbl_path, 'a') as lf:
                lf.write(line + '\n')

            if frame not in tp_frames:
                tp_frames.append(frame)

        # Save training set metadata
        meta = {
            "name":        set_id,
            "created":     datetime.now().isoformat(),
            "source_type": "review",
            "project":     self._project_name,
            "model":       self._detections[0].get('model_name', '') if self._detections else '',
            "tp_frames":   tp_frames,
            "fp_frames":   fp_frames,
        }
        save_training_set_meta(set_id, meta)

        msg = (f"Training set '{set_id}' created\n"
               f"{len(tp_frames)} TP frames / {len(fp_frames)} FP (hard neg) frames\n\n"
               f"Select it in the Models tab to include in training.")
        QMessageBox.information(self, "Review Saved", msg)
        self._selected.clear()
        self._refresh()

    def _set_result(self, indices, result):
        """Update detections.csv with reviewed/result for selected rows."""
        det_csv = project_detections_csv(self._project_name)
        if not os.path.exists(det_csv):
            return

        # Read all rows
        with open(det_csv, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            all_rows = list(reader)

        # Ensure reviewed/result columns exist
        if 'reviewed' not in fieldnames:
            fieldnames = list(fieldnames) + ['reviewed', 'result']
        if 'result' not in fieldnames:
            fieldnames = list(fieldnames) + ['result']

        # Map filtered detections back to all_rows
        # self._detections is a filtered view; we need original indices
        # Use (frame, box_x, box_y) as unique key
        def row_key(row):
            return (row.get('frame',''), row.get('box_x',''), row.get('box_y',''))

        keys_to_update = {row_key(self._detections[i]) for i in indices
                          if i < len(self._detections)}

        for row in all_rows:
            if row_key(row) in keys_to_update:
                row['reviewed'] = '1'
                row['result']   = result

        with open(det_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        # Update thumb visuals
        for idx in indices:
            if idx < len(self._thumbs):
                self._thumbs[idx].set_result(result)
                self._thumbs[idx]._selected = False
                self._thumbs[idx]._apply_style()
        self._selected.clear()
        self.count_lbl.setText(f"{len(self._detections)} detections")


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashcam Object Detector V1")
        self.resize(1400, 900)
        self._project_data = None
        self._settings     = load_global_settings()
        self._build_ui()
        self._load_projects_combo()
        last = self._settings.get('last_project', '')
        if last and last in list_projects():
            idx = self.project_combo.findText(last)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
                self._on_project_changed(last)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left sidebar ──
        sidebar = QWidget()
        sidebar.setFixedWidth(120)
        sidebar.setStyleSheet("background: #2a2a3a;")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(4, 8, 4, 8)
        sb_layout.setSpacing(4)

        self._panel_buttons = []
        panel_names = ["Process", "View", "Label", "Models", "Review"]
        for i, name in enumerate(panel_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            btn.setStyleSheet(
                "QPushButton { background: #3a3a5a; color: #ccc; border: none; border-radius: 4px; }"
                "QPushButton:checked { background: #5566cc; color: white; }"
                "QPushButton:hover { background: #4a4a7a; }"
            )
            btn.clicked.connect(lambda checked, idx=i: self._switch_panel(idx))
            sb_layout.addWidget(btn)
            self._panel_buttons.append(btn)

        sb_layout.addStretch()
        root.addWidget(sidebar)

        # ── Main area ──
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar: project selector
        top_bar = QWidget()
        top_bar.setFixedHeight(44)
        top_bar.setStyleSheet("background: #1a1a2a; border-bottom: 1px solid #444;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(12, 4, 12, 4)
        top_bar_layout.setSpacing(8)

        top_bar_layout.addWidget(QLabel("Project:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(260)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        top_bar_layout.addWidget(self.project_combo)

        new_proj_btn = QPushButton("New")
        new_proj_btn.setFixedHeight(28)
        new_proj_btn.clicked.connect(self._new_project)
        top_bar_layout.addWidget(new_proj_btn)

        del_proj_btn = QPushButton("Delete")
        del_proj_btn.setFixedHeight(28)
        del_proj_btn.clicked.connect(self._delete_project)
        top_bar_layout.addWidget(del_proj_btn)

        top_bar_layout.addStretch()
        main_layout.addWidget(top_bar)

        # Panel stack
        self.stack = QStackedWidget()

        self.process_panel = ProcessPanel(
            get_project_fn=lambda: self._project_data,
            global_settings=self._settings,
            save_settings_fn=save_global_settings,
        )
        self.process_panel.processing_done.connect(self._on_processing_done)

        self.view_panel    = ViewPanel()
        self.label_panel   = LabelPanel()
        self.models_panel  = ModelsPanel()
        self.review_panel  = ReviewPanel()

        self.stack.addWidget(self.process_panel)   # 0
        self.stack.addWidget(self.view_panel)       # 1
        self.stack.addWidget(self.label_panel)      # 2
        self.stack.addWidget(self.models_panel)     # 3
        self.stack.addWidget(self.review_panel)     # 4

        main_layout.addWidget(self.stack)
        root.addWidget(main_area)

        # Default to Process panel
        self._switch_panel(0)

    def _switch_panel(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._panel_buttons):
            btn.setChecked(i == index)
        # Reload current panel with current project
        if self._project_data:
            name = self._project_data['name']
            if index == 1:
                self.view_panel.load_project(name)
            elif index == 2:
                self.label_panel.load_project(name)
            elif index == 3:
                self.models_panel.load_project(name)
            elif index == 4:
                self.review_panel.load_project(name)

    def _load_projects_combo(self):
        self.project_combo.blockSignals(True)
        current = self.project_combo.currentText()
        self.project_combo.clear()
        for p in list_projects():
            self.project_combo.addItem(p)
        if current:
            idx = self.project_combo.findText(current)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
        self.project_combo.blockSignals(False)

    def _on_project_changed(self, name):
        if not name:
            return
        try:
            self._project_data = load_project(name)
        except Exception as e:
            print(f"_on_project_changed error: {e}")
            return
        self._settings['last_project'] = name
        save_global_settings(self._settings)

        # Load current panel
        idx = self.stack.currentIndex()
        if idx == 0:
            self.process_panel.load_project(name)
        elif idx == 1:
            self.view_panel.load_project(name)
        elif idx == 2:
            self.label_panel.load_project(name)
        elif idx == 3:
            self.models_panel.load_project(name)
        elif idx == 4:
            self.review_panel.load_project(name)

    def _on_processing_done(self):
        if self._project_data:
            # Auto-switch to view panel after processing
            self._switch_panel(1)

    def _new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in list_projects():
            QMessageBox.warning(self, "Exists", f"Project '{name}' already exists.")
            return

        data = {"name": name, "video_source": "", "frame_interval": 1.0, "jpeg_quality": 92}
        save_project(name, data)
        os.makedirs(project_frames_dir(name), exist_ok=True)
        os.makedirs(project_labels_dir(name), exist_ok=True)

        self._load_projects_combo()
        self.project_combo.setCurrentText(name)
        self._on_project_changed(name)
        print(f"DEBUG _new_project: _project_data={self._project_data}")

    def _delete_project(self):
        name = self.project_combo.currentText()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Delete Project",
            f"Delete project '{name}' and ALL its files?",
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if reply == QMessageBox.Yes:
            proj_dir = os.path.join(PROJECTS_DIR, name)
            if os.path.isdir(proj_dir):
                shutil.rmtree(proj_dir)
            self._project_data = None
            self._load_projects_combo()
            if self.project_combo.count() > 0:
                self.project_combo.setCurrentIndex(0)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = app.palette()
    from PyQt5.QtGui import QPalette
    palette.setColor(QPalette.Window,          QColor(30, 30, 42))
    palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Base,            QColor(22, 22, 32))
    palette.setColor(QPalette.AlternateBase,   QColor(40, 40, 55))
    palette.setColor(QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.Button,          QColor(55, 55, 78))
    palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight,       QColor(85, 102, 204))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # Pre-warm the Chromium web engine before showing the window.
    # Without this, the first QWebEngineView creation blocks the UI for ~5s.
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    _warmup = QWebEngineView()
    _warmup.setHtml("<html></html>")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
