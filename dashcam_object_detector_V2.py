#!/usr/bin/env python3
"""
Dashcam Object Detector V2
PyQt5 desktop app: GPS frame viewer + YOLO object detection training pipeline.
Five panels: Process | View | Label | Models | Review

V2 changes:
- View tab: draggable splitter between map and frame viewer (each can go nearly full width)
- Control rows (nav, culverts, traffic) moved to a full-width bottom strip outside the splitter
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
    QUrl, QRect, QPoint, QSize, QEvent
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen

import folium

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECTS_DIR        = str(Path(__file__).parent / "dashcam_projects")
MODELS_DIR          = os.path.join(PROJECTS_DIR, "models")
TRAINING_SETS_DIR   = os.path.join(PROJECTS_DIR, "training_sets")
CLASSES_PATH        = os.path.join(PROJECTS_DIR, "classes.json")
GLOBAL_SETTINGS_PATH  = os.path.join(PROJECTS_DIR, "settings.json")
CULVERT_SETTINGS_PATH = os.path.join(PROJECTS_DIR, "culvert_settings.json")
VIEW_SETTINGS_PATH    = os.path.join(PROJECTS_DIR, "view_settings.json")
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


def project_culverts_csv(name):
    return os.path.join(PROJECTS_DIR, name, "culverts.csv")


def project_vehicles_csv(name):
    return os.path.join(PROJECTS_DIR, name, "vehicles.csv")


def project_poi_csv(name):
    return os.path.join(PROJECTS_DIR, name, "poi.csv")


def load_culvert_settings():
    if os.path.exists(CULVERT_SETTINGS_PATH):
        with open(CULVERT_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def load_view_settings():
    if os.path.exists(VIEW_SETTINGS_PATH):
        with open(VIEW_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def save_view_settings(s):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(VIEW_SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)


def save_culvert_settings(settings):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(CULVERT_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)

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


# ─── Culvert Find Worker ──────────────────────────────────────────────────────

class CulvertFindWorker(QThread):
    log      = pyqtSignal(str)
    found    = pyqtSignal(list)   # list of {"frame", "lat", "lon", "sep"}
    finished = pyqtSignal(bool)

    def __init__(self, project_name, points_with_files, conf, pixel_sep, model_name=''):
        super().__init__()
        self.project_name      = project_name
        self.points_with_files = points_with_files
        self.conf              = conf
        self.pixel_sep         = pixel_sep
        self.model_name        = model_name

    def _resolve_model_path(self):
        if not self.model_name:
            return None
        mwp = model_weights_path(self.model_name)
        if os.path.exists(mwp):
            return mwp
        # Fall back to local file (pretrained .pt in script directory)
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_name)
        if os.path.exists(local):
            return local
        return None

    def run(self):
        frames_dir = project_frames_dir(self.project_name)
        model_path = self._resolve_model_path()
        if not model_path:
            self.log.emit("No model weights found. Train the model first or select a pretrained .pt file.")
            self.finished.emit(False)
            return

        params = {
            "frames_dir": frames_dir,
            "model_path": model_path,
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


# ─── Vehicle Count Worker ─────────────────────────────────────────────────────

class VehicleCountWorker(QThread):
    log      = pyqtSignal(str)
    done     = pyqtSignal(object)   # list of vehicle dicts
    finished = pyqtSignal(bool)

    def __init__(self, project_name, points_with_files, conf, frame_interval, model_name='yolov8n.pt'):
        super().__init__()
        self.project_name      = project_name
        self.points_with_files = points_with_files
        self.conf              = conf
        self.frame_interval    = frame_interval
        self.model_name        = model_name

    def run(self):
        base_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_name)
        if not os.path.exists(base_model):
            self.log.emit(f"{self.model_name} not found locally — Ultralytics will download it.")
            base_model = self.model_name

        frames_dir = project_frames_dir(self.project_name)
        params = {
            'frames_dir':     frames_dir,
            'model_path':     base_model,
            'conf':           self.conf,
            'frame_interval': self.frame_interval,
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
        result = []
        with open(log_file, 'w', encoding='utf-8') as lf:
            for line in proc.stdout:
                line = line.rstrip()
                lf.write(line + '\n')
                lf.flush()
                if not line:
                    continue
                if line.startswith('VEHICLES:'):
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
        self._js_blocked       = False   # set True while splitter is being dragged
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
        if not self._js_blocked:
            self.update_marker(lat, lon)
        if self.auto_center_enabled and not self._js_blocked:
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
        if self._js_blocked:
            return
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
                boxes.append(cx)
    if len(boxes) < 2:
        continue
    for b1, b2 in itertools.combinations(boxes, 2):
        sep = abs(b1 - b2)
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
points         = p['points']   # [[lat, lon, fname], ...] sorted by filename

VEHICLE_CLASSES = {2, 5, 7}   # car, bus, truck (COCO)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

cum = 0.0
distances = [0.0]
for i in range(1, len(points)):
    cum += haversine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
    distances.append(cum)

total_km = distances[-1]
print(f"Route: {total_km:.1f}km over {len(points)} frames", flush=True)

from ultralytics import YOLO
model = YOLO(model_path)

MAX_GAP     = 3
DIST_THRESH = 0.15
GROW_THRESH = 1.2
COCO_NAMES  = {2: 'car', 5: 'bus', 7: 'truck'}

active_tracks = []
vehicles_out  = []
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

def record_closed(closed, out):
    for t in closed:
        if t['area_last'] >= t['area_first'] * GROW_THRESH:
            out.append({
                'lat':   t['last_lat'],
                'lon':   t['last_lon'],
                'cls':   COCO_NAMES.get(t['cls'], 'vehicle'),
                'frame': t['last_frame_name'],
                'box':   t['last_box'],
            })

for idx, (lat, lon, fname) in enumerate(points):
    pct = int((idx + 1) / total * 100)
    if (idx + 1) % 25 == 0 or idx == total - 1:
        print(f"PROGRESS:{pct}:{idx+1}/{total} frames | {len(active_tracks)} active | {len(vehicles_out)} counted", flush=True)

    fpath = os.path.join(frames_dir, fname)
    if not os.path.exists(fpath):
        continue

    results = model(fpath, conf=conf, verbose=False, stream=True, half=True, device=0)
    detections = []
    img_w = img_h = None

    for r in results:
        if img_w is None and r.orig_shape:
            img_h, img_w = r.orig_shape[:2]
        if r.boxes:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                if cls not in VEHICLE_CLASSES:
                    continue
                if img_w and not (0.2 * img_w < cx < 0.8 * img_w):
                    continue
                if img_h and cy > 0.75 * img_h:
                    continue
                detections.append((cx, cy, area, cls, x1, y1, x2, y2))

    active_tracks, stale = close_stale(active_tracks, idx)
    record_closed(stale, vehicles_out)

    matched = set()
    for cx, cy, area, cls, x1, y1, x2, y2 in detections:
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
            best['cx'] = cx; best['cy'] = cy
            best['area_last']       = area
            best['last_frame']      = idx
            best['last_frame_name'] = fname
            best['last_lat']        = lat
            best['last_lon']        = lon
            best['last_box']        = [x1, y1, x2, y2]
            matched.add(id(best))
        else:
            active_tracks.append({'cx': cx, 'cy': cy,
                                   'area_first': area, 'area_last': area,
                                   'first_frame': idx, 'last_frame': idx,
                                   'first_lat': lat, 'first_lon': lon,
                                   'first_frame_name': fname,
                                   'last_frame_name': fname,
                                   'last_lat': lat, 'last_lon': lon,
                                   'last_box': [x1, y1, x2, y2],
                                   'cls': cls})

_, remaining = close_stale(active_tracks, total, force=True)
record_closed(remaining, vehicles_out)

print(f"VEHICLES:" + json.dumps(vehicles_out), flush=True)
print(f"Done. {len(vehicles_out)} oncoming vehicle(s) detected.", flush=True)
"""


def build_folium_map(project_name, project_data, points_with_files,
                     show_culverts=True, show_vehicles=True, show_poi=True):
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

    # ── Inject saved culvert markers at map load ──────────────────────────────
    map_var = m.get_name()
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
                f"window._culvertLayerGroup = L.layerGroup(){'.addTo(' + map_var + ')' if show_culverts else ''};",
            ]
            for clat, clon, src_frame in culvert_entries:
                safe_frame = src_frame.replace("'", "\\'")
                popup = f"'culvert<br>{clat:.5f}, {clon:.5f}'"
                sz = cv_radius * 2
                if cv_shape == 'Circle':
                    stmt = (f"window._culvertMarkers['{safe_frame}'] = "
                            f"L.circleMarker([{clat},{clon}],"
                            f"{{radius:{cv_radius},color:'white',weight:2,"
                            f"fillColor:'{cv_color}',fillOpacity:0.9}})"
                            f".bindPopup({popup}).addTo(window._culvertLayerGroup);")
                elif cv_shape == 'Square':
                    shape_svg = f'<rect width="{sz}" height="{sz}" fill="{cv_color}" stroke="white" stroke-width="2"/>'
                    svg = f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg">{shape_svg}</svg>'
                    stmt = (f"window._culvertMarkers['{safe_frame}'] = "
                            f"L.marker([{clat},{clon}],{{icon:L.divIcon({{html:'{svg}',"
                            f"iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                            f").bindPopup({popup}).addTo(window._culvertLayerGroup);")
                else:  # Diamond
                    h = sz // 2
                    shape_svg = f'<polygon points="{h},0 {sz},{h} {h},{sz} 0,{h}" fill="{cv_color}" stroke="white" stroke-width="2"/>'
                    svg = f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg">{shape_svg}</svg>'
                    stmt = (f"window._culvertMarkers['{safe_frame}'] = "
                            f"L.marker([{clat},{clon}],{{icon:L.divIcon({{html:'{svg}',"
                            f"iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                            f").bindPopup({popup}).addTo(window._culvertLayerGroup);")
                markers_js.append(stmt)
            culvert_js = (
                "<script>\nsetTimeout(function() {\n"
                + "\n".join(markers_js)
                + "\n}, 700);\n</script>"
            )
            m.get_root().html.add_child(folium.Element(culvert_js))

    # ── Inject saved vehicle markers at map load ──────────────────────────────
    vehicles_csv = project_vehicles_csv(project_name)
    if os.path.exists(vehicles_csv):
        CLASS_COLOR = {'car': 'blue', 'truck': 'red', 'bus': 'orange', 'vehicle': 'purple'}
        v_entries = []
        try:
            with open(vehicles_csv, newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        v_entries.append((float(row['lat']), float(row['lon']),
                                          row['cls'], row['frame']))
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass
        if v_entries:
            lines = [
                f"window._trafficLayer = L.layerGroup(){'.addTo(' + map_var + ')' if show_vehicles else ''};",
            ]
            for vlat, vlon, vcls, vframe in v_entries:
                color = CLASS_COLOR.get(vcls, 'purple')
                safe_frame = vframe.replace("'", "\\'")
                popup = f"'{vcls}<br>{vlat:.5f}, {vlon:.5f}<br>{safe_frame}'"
                lines.append(
                    f"L.circleMarker([{vlat},{vlon}],"
                    f"{{radius:7,color:'white',weight:1.5,fillColor:'{color}',fillOpacity:0.85}})"
                    f".bindPopup({popup}).addTo(window._trafficLayer);"
                )
            vehicle_js = (
                "<script>\nsetTimeout(function() {\n"
                + "\n".join(lines)
                + "\n}, 800);\n</script>"
            )
            m.get_root().html.add_child(folium.Element(vehicle_js))

    # ── Inject saved POI markers at map load ──────────────────────────────────
    poi_csv = project_poi_csv(project_name)
    if os.path.exists(poi_csv):
        poi_entries = []
        try:
            with open(poi_csv, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    try:
                        poi_entries.append((
                            float(row['lat']), float(row['lon']),
                            row.get('description', ''),
                            row.get('shape', 'Circle'),
                            row.get('color', 'yellow'),
                            row.get('size', 'Medium'),
                        ))
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass
        if poi_entries:
            _POI_RADIUS = {'Small': 6, 'Medium': 10, 'Large': 14}
            lines = [
                f"window._poiLayer = L.layerGroup(){'.addTo(' + map_var + ')' if show_poi else ''};",
                "window._poiMarkers = [];",
            ]
            for lat, lon, desc, shape, color, size in poi_entries:
                radius = _POI_RADIUS.get(size, 10)
                sz = radius * 2
                safe_desc = desc.replace("'", "\\'")
                popup = f"'<b>{safe_desc}</b><br>{lat:.5f}, {lon:.5f}'"
                key = f"{lat:.6f}_{lon:.6f}"
                if shape == 'Circle':
                    mk = (f"L.circleMarker([{lat},{lon}],"
                          f"{{radius:{radius},color:'white',weight:1.5,"
                          f"fillColor:'{color}',fillOpacity:0.9}})"
                          f".bindPopup({popup})")
                else:
                    if shape == 'Square':
                        svg_shape = f'<rect width=\\"{sz}\\" height=\\"{sz}\\" fill=\\"{color}\\" stroke=\\"white\\" stroke-width=\\"1.5\\"/>'
                    else:
                        h = sz // 2
                        svg_shape = f'<polygon points=\\"{h},0 {sz},{h} {h},{sz} 0,{h}\\" fill=\\"{color}\\" stroke=\\"white\\" stroke-width=\\"1.5\\"/>'
                    svg = f'<svg width=\\"{sz}\\" height=\\"{sz}\\" xmlns=\\"http://www.w3.org/2000/svg\\">{svg_shape}</svg>'
                    mk = (f"L.marker([{lat},{lon}],{{icon:L.divIcon({{"
                          f"html:'{svg}',iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                          f").bindPopup({popup})")
                lines.append(f"(function(){{ var m={mk}.addTo(window._poiLayer); window._poiMarkers.push({{marker:m,lat:{lat},lon:{lon},key:'{key}'}});}})();")
            poi_js = (
                "<script>\nsetTimeout(function() {\n"
                + "\n".join(lines)
                + "\n}, 900);\n</script>"
            )
            m.get_root().html.add_child(folium.Element(poi_js))

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
                  self.sm_radius_spin, self.sl_size_spin]:
            w.valueChanged.connect(self._save_display_settings)
        self.sl_check.stateChanged.connect(self._save_display_settings)
        self.sl_bg_check.stateChanged.connect(self._save_display_settings)
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
        self.sl_check.setChecked(project_data.get('sl_show', True))
        self.sl_size_spin.setValue(project_data.get('sl_size', 11))
        _sc(self.sl_text_color_btn, 'sl_text_color', '#ffffff')
        self.sl_bg_check.setChecked(project_data.get('sl_bg_show', True))
        _sc(self.sl_bg_color_btn, 'sl_bg_color', '#333333')

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
            'sl_show':       self.sl_check.isChecked(),
            'sl_size':       self.sl_size_spin.value(),
            'sl_text_color': self.sl_text_color_btn.property('hex_color'),
            'sl_bg_show':    self.sl_bg_check.isChecked(),
            'sl_bg_color':   self.sl_bg_color_btn.property('hex_color'),
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
    vehicles_counted = pyqtSignal(str)   # emits project_name when vehicle count finishes
    poi_changed      = pyqtSignal(str)   # emits project_name when POI list changes

    def __init__(self):
        super().__init__()
        self.communicator    = None
        self.channel         = None
        self._tmp_html       = None
        self._project_name   = None
        self._points         = []
        self._culvert_worker = None
        self._vehicle_worker = None
        self._map_loaded      = False  # True once QWebEngineView.loadFinished fires
        self._culvert_loading = False  # True while _load_culvert_settings is running
        self._last_vehicles  = []
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

        # ── Splitter: map | frame viewer (each can go nearly full width) ──────
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(6)
        self.splitter.setStyleSheet(
            "QSplitter::handle { background: #555; border-radius: 3px; }"
            "QSplitter::handle:hover { background: #4a9eff; }"
        )
        layout.addWidget(self.splitter, stretch=1)

        # Splitter drag guard — initialised here, handle installed after panes added
        self._splitter_was_next = False
        self._splitter_was_prev = False
        self._splitter_resume_timer = QTimer()
        self._splitter_resume_timer.setSingleShot(True)
        self._splitter_resume_timer.timeout.connect(self._resume_auto_after_drag)

        # Map pane — minimum so it can collapse to near-zero
        self.map_container = QWidget()
        self.map_container.setMinimumWidth(60)
        self.map_layout = QVBoxLayout(self.map_container)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = None
        self.splitter.addWidget(self.map_container)

        # Frame pane — image only, no controls inside
        image_widget = QWidget()
        image_widget.setMinimumWidth(60)
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = BoxFrameLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label, stretch=1)

        self.splitter.addWidget(image_widget)
        self.splitter.setSizes([500, 900])

        # Install event filter on the splitter handle so we can block JS the
        # instant the user presses the mouse — BEFORE any resize occurs.
        self.splitter.handle(1).installEventFilter(self)

        # ── Bottom control strip (full width, outside splitter) ───────────────
        bottom_strip = QWidget()
        bottom_strip.setObjectName("bottom_strip")
        bottom_strip.setStyleSheet("QWidget#bottom_strip { background-color: #2b2b2b; }")
        bottom_layout = QVBoxLayout(bottom_strip)
        bottom_layout.setContentsMargins(4, 4, 4, 4)
        bottom_layout.setSpacing(3)
        layout.addWidget(bottom_strip)

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

        # ── Row 1: Navigation ─────────────────────────────────────────────────
        nav_container = QWidget()
        nav_container.setObjectName("nav_container")
        nav_container.setFixedHeight(46)
        nav_container.setStyleSheet("QWidget#nav_container { background-color: #1e3a5f; border-radius: 4px; }")
        btn_row = QHBoxLayout(nav_container)
        btn_row.setContentsMargins(6, 4, 6, 4)
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

        for w in [self.auto_left, self.left_btn, self.right_btn,
                  self.auto_right, self.center_toggle]:
            w.setStyleSheet(_btn_style)

        btn_row.addStretch()
        for w in [self.auto_left, self.left_btn, self.spin_box,
                  self.right_btn, self.auto_right, self.center_toggle]:
            btn_row.addWidget(w)
        btn_row.addStretch()
        bottom_layout.addWidget(nav_container)

        # ── Row 2: Points of Interest ────────────────────────────────────────
        poi_container = QWidget()
        poi_container.setObjectName("poi_container")
        poi_container.setStyleSheet("QWidget#poi_container { background-color: #353535; border-radius: 4px; }")
        poi_row = QHBoxLayout(poi_container)
        poi_row.setContentsMargins(6, 4, 6, 4)
        poi_row.setSpacing(6)

        self.show_poi_chk = QCheckBox("Show POI")
        self.show_poi_chk.setChecked(True)
        self.show_poi_chk.toggled.connect(self._toggle_poi_layer)
        poi_row.addWidget(self.show_poi_chk)

        self.poi_desc_edit = QLineEdit()
        self.poi_desc_edit.setPlaceholderText("POI description")
        self.poi_desc_edit.setMinimumWidth(160)
        self.poi_desc_edit.returnPressed.connect(self._add_poi)
        poi_row.addWidget(self.poi_desc_edit)

        self.poi_add_btn = QPushButton("Add")
        self.poi_add_btn.setToolTip("Add a POI marker at the current frame position")
        self.poi_add_btn.clicked.connect(self._add_poi)
        self.poi_add_btn.setStyleSheet(_btn_style)
        poi_row.addWidget(self.poi_add_btn)

        poi_row.addWidget(QLabel("Marker:"))
        self.poi_shape_combo = QComboBox()
        self.poi_shape_combo.addItems(["Circle", "Square", "Diamond"])
        poi_row.addWidget(self.poi_shape_combo)

        self.poi_color_combo = QComboBox()
        self.poi_color_combo.addItems(["yellow", "cyan", "orange", "red", "blue", "green", "purple", "white"])
        poi_row.addWidget(self.poi_color_combo)

        self.poi_size_combo = QComboBox()
        self.poi_size_combo.addItems(["Small", "Medium", "Large"])
        self.poi_size_combo.setCurrentText("Medium")
        poi_row.addWidget(self.poi_size_combo)

        self.poi_del_btn = QPushButton("Del")
        self.poi_del_btn.setToolTip("Delete the POI nearest to the current frame position")
        self.poi_del_btn.clicked.connect(self._del_poi)
        self.poi_del_btn.setStyleSheet(_btn_style)
        poi_row.addWidget(self.poi_del_btn)

        poi_row.addStretch()
        bottom_layout.addWidget(poi_container)

        # ── Row 3: Survey points ──────────────────────────────────────────────
        survey_container = QWidget()
        survey_container.setObjectName("survey_container")
        survey_container.setStyleSheet("QWidget#survey_container { background-color: #353535; border-radius: 4px; }")
        survey_row = QHBoxLayout(survey_container)
        survey_row.setContentsMargins(6, 4, 6, 4)
        survey_row.setSpacing(6)

        self.show_survey_line_chk = QCheckBox("Show Survey Line")
        self.show_survey_line_chk.setChecked(True)
        self.show_survey_line_chk.toggled.connect(self._toggle_survey_line)
        survey_row.addWidget(self.show_survey_line_chk)

        self.show_survey_pts_chk = QCheckBox("Show Survey Points")
        self.show_survey_pts_chk.setChecked(True)
        self.show_survey_pts_chk.toggled.connect(self._toggle_survey_points)
        survey_row.addWidget(self.show_survey_pts_chk)

        self.show_survey_labels_chk = QCheckBox("Show Survey Labels")
        self.show_survey_labels_chk.setChecked(True)
        self.show_survey_labels_chk.toggled.connect(self._toggle_survey_labels)
        survey_row.addWidget(self.show_survey_labels_chk)

        survey_row.addStretch()

        survey_row.addWidget(QLabel("Go To Station:"))
        self.survey_goto_edit = QLineEdit()
        self.survey_goto_edit.setPlaceholderText("e.g. 100")
        self.survey_goto_edit.setFixedWidth(110)
        self.survey_goto_edit.returnPressed.connect(self._goto_survey_station)
        survey_row.addWidget(self.survey_goto_edit)

        self.survey_goto_btn = QPushButton("Go To")
        self.survey_goto_btn.setToolTip("Centre map on the entered survey station")
        self.survey_goto_btn.clicked.connect(self._goto_survey_station)
        self.survey_goto_btn.setStyleSheet(_btn_style)
        survey_row.addWidget(self.survey_goto_btn)

        bottom_layout.addWidget(survey_container)

        # ── Row 4: Culvert detection ──────────────────────────────────────────
        culvert_container = QWidget()
        culvert_container.setObjectName("culvert_container")
        culvert_container.setStyleSheet("QWidget#culvert_container { background-color: #353535; border-radius: 4px; }")
        culvert_row = QHBoxLayout(culvert_container)
        culvert_row.setContentsMargins(6, 4, 6, 4)
        culvert_row.setSpacing(6)

        self.show_culverts_chk = QCheckBox("Show Culverts")
        self.show_culverts_chk.setChecked(True)
        self.show_culverts_chk.toggled.connect(self._toggle_culverts_layer)
        culvert_row.addWidget(self.show_culverts_chk)

        self.culvert_model_combo = QComboBox()
        self.culvert_model_combo.setMinimumWidth(130)
        self.culvert_model_combo.setToolTip("Model to use for culvert detection")
        culvert_row.addWidget(self.culvert_model_combo)

        self.culvert_conf_spin = QDoubleSpinBox()
        self.culvert_conf_spin.setRange(0.05, 0.95)
        self.culvert_conf_spin.setSingleStep(0.05)
        self.culvert_conf_spin.setValue(0.25)
        self.culvert_conf_spin.setDecimals(2)
        self.culvert_conf_spin.setFixedWidth(98)
        culvert_row.addWidget(self.culvert_conf_spin)

        self.find_culverts_btn = QPushButton("Find Culverts")
        self.find_culverts_btn.setToolTip("Scan all frames with YOLO and flag pairs of detections at the given pixel separation")
        self.find_culverts_btn.clicked.connect(self._find_culverts)
        culvert_row.addWidget(self.find_culverts_btn)

        self.culvert_sep_spin = QSpinBox()
        self.culvert_sep_spin.setRange(10, 2000)
        self.culvert_sep_spin.setValue(200)
        self.culvert_sep_spin.setSingleStep(10)
        self.culvert_sep_spin.setPrefix("sep ")
        self.culvert_sep_spin.setSuffix(" px")
        self.culvert_sep_spin.setFixedWidth(154)
        culvert_row.addWidget(self.culvert_sep_spin)

        culvert_row.addWidget(QLabel("Marker:"))
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
        self.add_culvert_btn.setToolTip("Mark current frame as a culvert location")
        self.add_culvert_btn.clicked.connect(self._add_culvert)
        culvert_row.addWidget(self.add_culvert_btn)

        self.remove_culvert_btn = QPushButton("Remove")
        self.remove_culvert_btn.setToolTip("Remove culvert entry for current frame")
        self.remove_culvert_btn.clicked.connect(self._remove_culvert)
        culvert_row.addWidget(self.remove_culvert_btn)

        self.remove_all_culverts_btn = QPushButton("Remove All")
        self.remove_all_culverts_btn.setToolTip("Delete all culvert records for this project")
        self.remove_all_culverts_btn.clicked.connect(self._remove_all_culverts)
        culvert_row.addWidget(self.remove_all_culverts_btn)

        for w in [self.find_culverts_btn, self.prev_culvert_btn, self.next_culvert_btn,
                  self.add_culvert_btn, self.remove_culvert_btn, self.remove_all_culverts_btn]:
            w.setStyleSheet(_btn_style)

        culvert_row.addStretch()

        for combo in [self.culvert_color_combo, self.culvert_shape_combo, self.culvert_size_combo]:
            combo.currentIndexChanged.connect(self._restyle_culvert_markers)

        bottom_layout.addWidget(culvert_container)

        # ── Row 5: Vehicle count / traffic density ────────────────────────────
        traffic_container = QWidget()
        traffic_container.setObjectName("traffic_container")
        traffic_container.setStyleSheet("QWidget#traffic_container { background-color: #353535; border-radius: 4px; }")
        traffic_row = QHBoxLayout(traffic_container)
        traffic_row.setContentsMargins(6, 4, 6, 4)
        traffic_row.setSpacing(6)

        self.show_traffic_chk = QCheckBox("Show Vehicles")
        self.show_traffic_chk.setChecked(True)
        self.show_traffic_chk.toggled.connect(self._toggle_traffic_layer)
        traffic_row.addWidget(self.show_traffic_chk)

        self.count_vehicles_btn = QPushButton("Count Vehicles")
        self.count_vehicles_btn.setToolTip("Run YOLO on all frames to count oncoming vehicles")
        self.count_vehicles_btn.clicked.connect(self._count_vehicles)
        traffic_row.addWidget(self.count_vehicles_btn)

        traffic_row.addWidget(QLabel("Interval:"))
        self.vehicle_interval_spin = QDoubleSpinBox()
        self.vehicle_interval_spin.setRange(0.5, 60.0)
        self.vehicle_interval_spin.setSingleStep(0.5)
        self.vehicle_interval_spin.setDecimals(1)
        self.vehicle_interval_spin.setValue(1.0)
        self.vehicle_interval_spin.setSuffix(" sec")
        self.vehicle_interval_spin.setFixedWidth(126)
        self.vehicle_interval_spin.setToolTip("Frame interval used when frames were extracted")
        traffic_row.addWidget(self.vehicle_interval_spin)

        self.vehicle_conf_spin = QDoubleSpinBox()
        self.vehicle_conf_spin.setRange(0.05, 0.95)
        self.vehicle_conf_spin.setSingleStep(0.05)
        self.vehicle_conf_spin.setDecimals(2)
        self.vehicle_conf_spin.setValue(0.15)
        self.vehicle_conf_spin.setPrefix("conf ")
        self.vehicle_conf_spin.setFixedWidth(140)
        traffic_row.addWidget(self.vehicle_conf_spin)

        traffic_row.addWidget(QLabel("Model:"))
        self.vehicle_model_combo = QComboBox()
        _vmodels = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8x.pt',
                    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11x.pt']
        self.vehicle_model_combo.addItems(_vmodels)
        self.vehicle_model_combo.setCurrentText('yolov8n.pt')
        self.vehicle_model_combo.setToolTip(
            "Pretrained YOLO weights for vehicle detection.\n"
            "n=nano (fastest), x=extra-large (most accurate).\n"
            "Unknown models are auto-downloaded by Ultralytics on first use."
        )
        traffic_row.addWidget(self.vehicle_model_combo)

        self.prev_vehicle_btn = QPushButton("← Vehicle")
        self.prev_vehicle_btn.setToolTip("Jump to previous vehicle detection frame")
        self.prev_vehicle_btn.clicked.connect(self._prev_vehicle_frame)
        traffic_row.addWidget(self.prev_vehicle_btn)

        self.next_vehicle_btn = QPushButton("Vehicle →")
        self.next_vehicle_btn.setToolTip("Jump to next vehicle detection frame")
        self.next_vehicle_btn.clicked.connect(self._next_vehicle_frame)
        traffic_row.addWidget(self.next_vehicle_btn)

        self.clear_traffic_btn = QPushButton("Clear")
        self.clear_traffic_btn.setToolTip("Clear vehicle markers from map")
        self.clear_traffic_btn.clicked.connect(self._clear_vehicle_markers)
        traffic_row.addWidget(self.clear_traffic_btn)

        self.save_report_btn = QPushButton("Save Report")
        self.save_report_btn.setToolTip("Generate PDF report with map and vehicle statistics")
        self.save_report_btn.clicked.connect(self._save_vehicle_report)
        traffic_row.addWidget(self.save_report_btn)

        for w in [self.count_vehicles_btn, self.prev_vehicle_btn, self.next_vehicle_btn,
                  self.clear_traffic_btn, self.save_report_btn]:
            w.setStyleSheet(_btn_style)

        traffic_row.addStretch()
        bottom_layout.addWidget(traffic_container)

        # ── Restore and persist checkbox states ───────────────────────────────
        _vs = load_view_settings()
        _chk_defaults = {
            'show_poi':            (self.show_poi_chk,            True),
            'show_survey_line':    (self.show_survey_line_chk,    True),
            'show_survey_pts':     (self.show_survey_pts_chk,     True),
            'show_survey_labels':  (self.show_survey_labels_chk,  True),
            'show_culverts':       (self.show_culverts_chk,        True),
            'show_vehicles':       (self.show_traffic_chk,         True),
            'auto_center':         (self.center_toggle,            False),
        }
        for key, (widget, default) in _chk_defaults.items():
            widget.blockSignals(True)
            widget.setChecked(_vs.get(key, default))
            widget.blockSignals(False)
            widget.toggled.connect(self._save_view_settings)

    def _save_view_settings(self):
        save_view_settings({
            'show_poi':           self.show_poi_chk.isChecked(),
            'show_survey_line':   self.show_survey_line_chk.isChecked(),
            'show_survey_pts':    self.show_survey_pts_chk.isChecked(),
            'show_survey_labels': self.show_survey_labels_chk.isChecked(),
            'show_culverts':      self.show_culverts_chk.isChecked(),
            'show_vehicles':      self.show_traffic_chk.isChecked(),
            'auto_center':        self.center_toggle.isChecked(),
        })

    def eventFilter(self, obj, event):
        """Intercept mouse press/release on the splitter handle."""
        if obj is self.splitter.handle(1):
            if event.type() == QEvent.MouseButtonPress:
                # If the map hasn't finished loading, eat the event entirely —
                # resizing QWebEngineView during Chromium's initial render crashes it.
                if not self._map_loaded:
                    return True
                # Block JS before any resize happens
                if self.communicator:
                    self.communicator._js_blocked = True
                    if self.communicator.next_timer.isActive():
                        self._splitter_was_next = True
                        self.communicator.next_timer.stop()
                    if self.communicator.prev_timer.isActive():
                        self._splitter_was_prev = True
                        self.communicator.prev_timer.stop()
            elif event.type() == QEvent.MouseButtonRelease:
                # Start debounce — resume after resize has fully settled
                self._splitter_resume_timer.start(350)
        return super().eventFilter(obj, event)

    def _resume_auto_after_drag(self):
        """Re-enable JS and restart whichever auto timer was running before the drag."""
        if self.communicator:
            self.communicator._js_blocked = False
            interval = self.spin_box.value()
            if self._splitter_was_next:
                self.communicator.next_timer.start(interval)
            if self._splitter_was_prev:
                self.communicator.prev_timer.start(interval)
        self._splitter_was_next = False
        self._splitter_was_prev = False

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
        self._map_loaded = False
        self.web_view = QWebEngineView()
        self.web_view.loadFinished.connect(self._on_map_load_finished)
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

        m = build_folium_map(project_name, project_data, points_with_files,
                             show_culverts=self.show_culverts_chk.isChecked(),
                             show_vehicles=self.show_traffic_chk.isChecked(),
                             show_poi=self.show_poi_chk.isChecked())
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
            sv_color      = project_data.get('sv_color',      '#00ccff')
            sv_width      = project_data.get('sv_width',      2)
            sp_color      = project_data.get('sp_color',      '#00ccff')
            sp_radius     = project_data.get('sp_radius',     4)
            sm_color      = project_data.get('sm_color',      '#ff3300')
            sm_radius     = project_data.get('sm_radius',     6)
            sl_show       = 'true' if project_data.get('sl_show', True) else 'false'
            sl_size       = project_data.get('sl_size',       11)
            sl_text_color = project_data.get('sl_text_color', '#ffffff')
            sl_bg_show    = project_data.get('sl_bg_show',    True)
            sl_bg_color   = project_data.get('sl_bg_color',   '#333333')

            bg_css = f'background: {sl_bg_color};' if sl_bg_show else 'background: transparent;'
            m.get_root().html.add_child(folium.Element(
                f'<style>.preplot-label {{ font-size: {sl_size}px; font-weight: bold; '
                f'{bg_css} color: {sl_text_color}; border: none; '
                f'padding: 1px 4px; border-radius: 3px; }}</style>'
            ))

            pts_json = json.dumps(
                [[round(lat, 6), round(lon, 6), stn] for lat, lon, stn in preplot_points]
            )
            preplot_js = (
                "<script>\nsetTimeout(function() {\n"
                "  try {\n"
                f"    var pp = {pts_json};\n"
                "    var latlngs = pp.map(function(p) { return [p[0], p[1]]; });\n"
                f"    window._preplotLine = L.polyline(latlngs, {{color: '{sv_color}', weight: {sv_width}, opacity: 0.8}}).addTo({map_var});\n"
                f"    window._preplotLayer = L.layerGroup().addTo({map_var});\n"
                "    window._preplotMarkers = [];\n"
                "    window._preplotStations = {};\n"
                "    var renderer = L.canvas({padding: 0.5});\n"
                "    for (var i = 0; i < pp.length; i++) {\n"
                "      var isStation = (i % 10 === 0);\n"
                f"      var mk = L.circleMarker([pp[i][0], pp[i][1]], {{renderer: renderer,\n"
                f"        color: isStation ? '{sm_color}' : '{sp_color}',\n"
                f"        fillColor: isStation ? '{sm_color}' : '{sp_color}',\n"
                f"        fillOpacity: 0.85, radius: isStation ? {sm_radius} : {sp_radius}, weight: 1\n"
                "      }).bindPopup('Station: ' + pp[i][2]);\n"
                "      window._preplotLayer.addLayer(mk);\n"
                "      window._preplotMarkers.push({marker: mk, isStation: isStation, name: pp[i][2]});\n"
                "      if (isStation) {\n"
                "        window._preplotStations[pp[i][2]] = [pp[i][0], pp[i][1]];\n"
                f"        if ({sl_show}) {{\n"
                "          mk.bindTooltip(pp[i][2], {\n"
                "            permanent: true, direction: 'right',\n"
                "            className: 'preplot-label'\n"
                "          });\n"
                "        }\n"
                "      }\n"
                "    }\n"
                "  } catch(e) { console.error('[preplot] ' + e); }\n"
                "}, 500);\n"
                "</script>\n"
            )
            m.get_root().html.add_child(folium.Element(preplot_js))

        self._tmp_html = _save_map_to_tempfile(m, self._tmp_html)
        self.web_view.setUrl(QUrl.fromLocalFile(self._tmp_html))
        self.coords_label.setText("Click on the map to view frames")
        self._refresh_culvert_models()
        self._load_culvert_settings()

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

    # ── POI helpers ───────────────────────────────────────────────────────────

    def goto_poi(self, lat, lon):
        """Pan the map and jump to the nearest frame for a given POI coordinate."""
        if not self.web_view or not self.communicator:
            return
        # Pan map
        mv = self.communicator.map_var
        self.web_view.page().runJavaScript(f"{mv}.setView([{lat},{lon}], {mv}.getZoom());")
        # Nearest frame
        best_idx, best_d = 0, float('inf')
        for i, (plat, plon, _) in enumerate(self.communicator.points_with_files):
            d = (plat - lat) ** 2 + (plon - lon) ** 2
            if d < best_d:
                best_d, best_idx = d, i
        self.communicator.current_index = best_idx
        self.communicator._show_frame(best_idx)

    def _toggle_poi_layer(self, checked):
        if not self.web_view:
            return
        mv = self.communicator.map_var if self.communicator else 'map'
        if checked:
            js = f"if (window._poiLayer) {{ {mv}.addLayer(window._poiLayer); }}"
        else:
            js = "if (window._poiLayer) { window._poiLayer.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _add_poi(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        lat, lon, _ = self.communicator.points_with_files[self.communicator.current_index]
        desc   = self.poi_desc_edit.text().strip()
        shape  = self.poi_shape_combo.currentText()
        color  = self.poi_color_combo.currentText()
        size   = self.poi_size_combo.currentText()
        radius = {'Small': 6, 'Medium': 10, 'Large': 14}.get(size, 10)
        mv     = self.communicator.map_var

        # Persist to CSV
        poi_csv = project_poi_csv(self._project_name)
        os.makedirs(os.path.dirname(poi_csv), exist_ok=True)
        write_header = not os.path.exists(poi_csv)
        with open(poi_csv, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['lat', 'lon', 'description', 'shape', 'color', 'size'])
            if write_header:
                w.writeheader()
            w.writerow({'lat': lat, 'lon': lon, 'description': desc,
                        'shape': shape, 'color': color, 'size': size})

        safe_desc = desc.replace("'", "\\'")
        popup = f"'<b>{safe_desc}</b><br>{lat:.5f}, {lon:.5f}'"
        key   = f"{lat:.6f}_{lon:.6f}".replace('.', '_').replace('-', 'n')
        js    = self._make_marker_js(lat, lon, popup, color, radius, shape,
                                     '_poiLayer', '_poiMarkers_dict', key, mv)
        # Also push to _poiMarkers array for del lookup
        js += (f"\nif (!window._poiMarkers) window._poiMarkers = [];"
               f"\nwindow._poiMarkers.push({{marker:window._poiMarkers_dict['{key}'],"
               f"lat:{lat},lon:{lon},key:'{key}'}});")
        self.web_view.page().runJavaScript(js)
        self.poi_desc_edit.clear()
        self.poi_changed.emit(self._project_name)

    def _del_poi(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        lat, lon, _ = self.communicator.points_with_files[self.communicator.current_index]
        poi_csv = project_poi_csv(self._project_name)
        if not os.path.exists(poi_csv):
            return

        # Find nearest POI in CSV to current lat/lon
        try:
            with open(poi_csv, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return
        if not rows:
            return

        def dist(r):
            return (float(r['lat']) - lat) ** 2 + (float(r['lon']) - lon) ** 2

        nearest = min(rows, key=dist)
        plat, plon = float(nearest['lat']), float(nearest['lon'])
        rows.remove(nearest)

        fieldnames = ['lat', 'lon', 'description', 'shape', 'color', 'size']
        with open(poi_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        key = f"{plat:.6f}_{plon:.6f}".replace('.', '_').replace('-', 'n')
        mv  = self.communicator.map_var
        js  = (f"if (window._poiMarkers) {{"
               f"  window._poiMarkers = window._poiMarkers.filter(function(item) {{"
               f"    if (Math.abs(item.lat-{plat})<0.000001 && Math.abs(item.lon-{plon})<0.000001) {{"
               f"      if (window._poiLayer) window._poiLayer.removeLayer(item.marker);"
               f"      return false;"
               f"    }} return true;"
               f"  }});"
               f"}}")
        self.web_view.page().runJavaScript(js)
        if self._project_name:
            self.poi_changed.emit(self._project_name)

    # ── Survey layer helpers ──────────────────────────────────────────────────

    def _on_map_load_finished(self, ok):
        self._map_loaded = True
        if self._project_name:
            self.poi_changed.emit(self._project_name)
        # Preplot (survey) injects at 500 ms → apply visibility at 650 ms.
        # Culverts/vehicles/POI visibility is now baked into build_folium_map,
        # so no timers needed for those layers.
        QTimer.singleShot(650, self._apply_survey_visibility)

    def _apply_survey_visibility(self):
        if not self.show_survey_line_chk.isChecked():
            self._toggle_survey_line(False)
        if not self.show_survey_pts_chk.isChecked():
            self._toggle_survey_points(False)
        if not self.show_survey_labels_chk.isChecked():
            self._toggle_survey_labels(False)

    def _toggle_survey_line(self, checked):
        if not self.web_view:
            return
        map_var = self.communicator.map_var if self.communicator else 'map'
        if checked:
            js = f"if (window._preplotLine) {{ {map_var}.addLayer(window._preplotLine); }}"
        else:
            js = "if (window._preplotLine) { window._preplotLine.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _toggle_survey_points(self, checked):
        if not self.web_view:
            return
        if checked:
            js = (f"if (window._preplotLayer) {{ {self.communicator.map_var if self.communicator else 'map'}"
                  f".addLayer(window._preplotLayer); }}")
        else:
            js = "if (window._preplotLayer) { window._preplotLayer.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _toggle_survey_labels(self, checked):
        if not self.web_view:
            return
        if checked:
            js = ("if (window._preplotMarkers) { window._preplotMarkers.forEach(function(item) {"
                  "  if (item.isStation && item.marker.getTooltip()) { item.marker.openTooltip(); }"
                  "}); }")
        else:
            js = ("if (window._preplotMarkers) { window._preplotMarkers.forEach(function(item) {"
                  "  if (item.isStation && item.marker.getTooltip()) { item.marker.closeTooltip(); }"
                  "}); }")
        self.web_view.page().runJavaScript(js)

    def _goto_survey_station(self):
        if not self.web_view:
            return
        name = self.survey_goto_edit.text().strip()
        if not name:
            return
        safe = name.replace("'", "\\'")
        map_var = self.communicator.map_var if self.communicator else 'map'
        js = f"""
        (function() {{
            if (!window._preplotStations) return;
            var stn = '{safe}';
            var found = window._preplotStations[stn];
            if (!found) {{
                var keys = Object.keys(window._preplotStations);
                for (var k = 0; k < keys.length; k++) {{
                    if (keys[k].toLowerCase() === stn.toLowerCase()) {{
                        found = window._preplotStations[keys[k]];
                        break;
                    }}
                }}
            }}
            if (found) {{ {map_var}.setView(found, {map_var}.getZoom()); }}
        }})();
        """
        self.web_view.page().runJavaScript(js)

    # ── Culvert helpers ───────────────────────────────────────────────────────

    def _refresh_culvert_models(self):
        current = self.culvert_model_combo.currentText()
        self.culvert_model_combo.blockSignals(True)
        self.culvert_model_combo.clear()
        for m in list_models():
            self.culvert_model_combo.addItem(m)
        if current:
            idx = self.culvert_model_combo.findText(current)
            if idx >= 0:
                self.culvert_model_combo.setCurrentIndex(idx)
        self.culvert_model_combo.blockSignals(False)

    def _load_culvert_settings(self):
        self._culvert_loading = True
        try:
            s = load_culvert_settings()
            if 'shape' in s:
                self.culvert_shape_combo.setCurrentText(s['shape'])
            if 'color' in s:
                self.culvert_color_combo.setCurrentText(s['color'])
            if 'size' in s:
                self.culvert_size_combo.setCurrentText(s['size'])
            if 'pixel_sep' in s:
                self.culvert_sep_spin.setValue(s['pixel_sep'])
        finally:
            self._culvert_loading = False

    def _save_culvert_settings(self):
        save_culvert_settings({
            'shape':     self.culvert_shape_combo.currentText(),
            'color':     self.culvert_color_combo.currentText(),
            'size':      self.culvert_size_combo.currentText(),
            'pixel_sep': self.culvert_sep_spin.value(),
        })

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
            else:  # Diamond
                h = sz // 2
                shape_svg = f'<polygon points=\\"{h},0 {sz},{h} {h},{sz} 0,{h}\\" fill=\\"{color}\\" stroke=\\"white\\" stroke-width=\\"1.5\\"/>'
            svg = f'<svg width=\\"{sz}\\" height=\\"{sz}\\" xmlns=\\"http://www.w3.org/2000/svg\\">{shape_svg}</svg>'
            marker = (f"L.marker([{lat},{lon}],{{icon:L.divIcon({{"
                      f"html:'{svg}',iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                      f").bindPopup({popup})")
        return f"{init} window.{markers_var}['{key}'] = {marker}.addTo(window.{layer_var});"

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

    def _restyle_culvert_markers(self):
        """Rebuild culvert markers on the map using the current style combos."""
        if self._culvert_loading or not self.web_view or not self._project_name:
            return
        culverts_csv = project_culverts_csv(self._project_name)
        if not os.path.exists(culverts_csv):
            return
        self._save_culvert_settings()
        color  = self.culvert_color_combo.currentText()
        shape  = self.culvert_shape_combo.currentText()
        radius = {'Small': 6, 'Medium': 10, 'Large': 14}.get(self.culvert_size_combo.currentText(), 10)
        mv     = self.communicator.map_var if self.communicator else 'map'

        entries = []
        try:
            with open(culverts_csv, newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        entries.append((float(row['latitude']), float(row['longitude']),
                                        row['source_frame']))
                    except (KeyError, ValueError):
                        pass
        except Exception:
            return
        if not entries:
            return

        stmts = [
            "if (window._culvertLayerGroup) { window._culvertLayerGroup.clearLayers(); }",
            "window._culvertMarkers = {};",
            f"if (!window._culvertLayerGroup) {{ window._culvertLayerGroup = L.layerGroup().addTo({mv}); }}",
        ]
        for lat, lon, fname in entries:
            safe  = fname.replace("'", "\\'")
            popup = f"'culvert<br>{lat:.5f}, {lon:.5f}'"
            stmts.append(self._make_marker_js(lat, lon, popup, color, radius, shape,
                                              '_culvertLayerGroup', '_culvertMarkers',
                                              safe, mv))
        self.web_view.page().runJavaScript("\n".join(stmts))

    def _toggle_culverts_layer(self, visible):
        if not self.web_view:
            return
        mv = self.communicator.map_var if self.communicator else 'map'
        if visible:
            js = f"if (window._culvertLayerGroup) {{ window._culvertLayerGroup.addTo({mv}); }}"
        else:
            js = "if (window._culvertLayerGroup) { window._culvertLayerGroup.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _find_culverts(self):
        if not self._project_name or not self.communicator:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        model_name = self.culvert_model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "No Model", "No models available. Create and train a model first.")
            return
        if not os.path.exists(model_weights_path(model_name)):
            local = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
            if not os.path.exists(local):
                QMessageBox.warning(self, "No Weights",
                                    f"Model '{model_name}' has no trained weights yet.")
                return
        self._save_culvert_settings()
        self.find_culverts_btn.setEnabled(False)
        self._culvert_worker = CulvertFindWorker(
            self._project_name,
            self.communicator.points_with_files,
            self.culvert_conf_spin.value(),
            self.culvert_sep_spin.value(),
            model_name=model_name,
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

    # ── Vehicle / traffic helpers ─────────────────────────────────────────────

    _VEHICLE_CLASS_COLOR = {'car': 'blue', 'truck': 'red', 'bus': 'orange', 'vehicle': 'purple'}

    def _vehicle_frame_indices(self):
        """Return sorted list of communicator indices that have a vehicle detection."""
        if not self.communicator or not self._project_name:
            return []
        csv_path = project_vehicles_csv(self._project_name)
        if not os.path.exists(csv_path):
            return []
        try:
            with open(csv_path, newline='') as f:
                frames = {row['frame'] for row in csv.DictReader(f) if row.get('frame')}
        except Exception:
            return []
        fname_to_idx = {p[2]: i for i, p in enumerate(self.communicator.points_with_files)}
        return sorted(fname_to_idx[f] for f in frames if f in fname_to_idx)

    def _prev_vehicle_frame(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        indices = self._vehicle_frame_indices()
        if not indices:
            self.coords_label.setText("No vehicle detection frames found.")
            return
        cur = self.communicator.current_index
        before = [i for i in indices if i < cur]
        target = before[-1] if before else indices[-1]
        self.communicator.current_index = target
        self.communicator._show_frame(target)

    def _next_vehicle_frame(self):
        if not self.communicator or self.communicator.current_index is None:
            return
        indices = self._vehicle_frame_indices()
        if not indices:
            self.coords_label.setText("No vehicle detection frames found.")
            return
        cur = self.communicator.current_index
        after = [i for i in indices if i > cur]
        target = after[0] if after else indices[0]
        self.communicator.current_index = target
        self.communicator._show_frame(target)

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
            model_name=self.vehicle_model_combo.currentText(),
        )
        self._vehicle_worker.log.connect(self.coords_label.setText)
        self._vehicle_worker.done.connect(self._on_vehicles_counted)
        self._vehicle_worker.finished.connect(lambda _: self.count_vehicles_btn.setEnabled(True))
        self._vehicle_worker.start()

    def _on_vehicles_counted(self, result):
        vehicles = result if isinstance(result, list) else []
        self._last_vehicles = vehicles
        if not vehicles:
            self.coords_label.setText("No vehicles detected.")
            return

        # Persist to CSV
        csv_path = project_vehicles_csv(self._project_name)
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['frame', 'lat', 'lon', 'cls',
                                              'box_x1', 'box_y1', 'box_x2', 'box_y2'])
            w.writeheader()
            for v in vehicles:
                box = v.get('box') or [0, 0, 0, 0]
                w.writerow({'frame': v.get('frame', ''), 'lat': v.get('lat', 0),
                            'lon': v.get('lon', 0), 'cls': v.get('cls', 'vehicle'),
                            'box_x1': box[0], 'box_y1': box[1],
                            'box_x2': box[2], 'box_y2': box[3]})

        mv = self.communicator.map_var
        lines = [
            f"window._trafficLayer = window._trafficLayer || L.layerGroup().addTo({mv});",
            "window._trafficLayer.clearLayers();",
        ]
        counts = {}
        for v in vehicles:
            cls   = v.get('cls', 'vehicle')
            lat   = v['lat']
            lon   = v['lon']
            frame = v.get('frame', '')
            color = self._VEHICLE_CLASS_COLOR.get(cls, 'purple')
            counts[cls] = counts.get(cls, 0) + 1
            safe_frame = frame.replace("'", "\\'")
            popup = f"'{cls}<br>{lat:.5f}, {lon:.5f}<br>{safe_frame}'"
            lines.append(
                f"L.circleMarker([{lat},{lon}],"
                f"{{radius:7,color:'white',weight:1.5,fillColor:'{color}',fillOpacity:0.85}})"
                f".bindPopup({popup}).addTo(window._trafficLayer);"
            )
        js = "setTimeout(function(){\n" + "\n".join(lines) + "\n}, 100);"
        self.web_view.page().runJavaScript(js)
        summary = ", ".join(f"{n} {c}" for c, n in sorted(counts.items()))
        self.coords_label.setText(f"{len(vehicles)} vehicles detected: {summary}.")
        if self._project_name:
            self.vehicles_counted.emit(self._project_name)

    def _toggle_traffic_layer(self, visible):
        if not self.web_view:
            return
        mv = self.communicator.map_var if self.communicator else 'map'
        if visible:
            js = f"if (window._trafficLayer) {{ window._trafficLayer.addTo({mv}); }}"
        else:
            js = "if (window._trafficLayer) { window._trafficLayer.remove(); }"
        self.web_view.page().runJavaScript(js)

    def _clear_vehicle_markers(self):
        if not self.web_view:
            return
        self.web_view.page().runJavaScript(
            "if (window._trafficLayer) { window._trafficLayer.clearLayers(); }"
        )
        self.coords_label.setText("Vehicle markers cleared.")

    def _save_vehicle_report(self):
        # Read from CSV (reflects any manual edits) then fall back to in-memory list
        vehicles = []
        if self._project_name:
            csv_path = project_vehicles_csv(self._project_name)
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, newline='') as f:
                        for row in csv.DictReader(f):
                            vehicles.append({
                                'frame': row['frame'],
                                'lat':   float(row['lat']),
                                'lon':   float(row['lon']),
                                'cls':   row['cls'],
                                'box':   [float(row['box_x1']), float(row['box_y1']),
                                          float(row['box_x2']), float(row['box_y2'])],
                            })
                except Exception:
                    pass
        if not vehicles:
            vehicles = self._last_vehicles
        if not vehicles:
            QMessageBox.warning(self, "No Data", "Run Count Vehicles first.")
            return

        default_name = f"{self._project_name or 'survey'}_vehicle_report.pdf"
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Save Vehicle Report", default_path, "PDF Files (*.pdf)")
        if not path:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.backends.backend_pdf import PdfPages

            CLASS_COLOR = {'car': 'steelblue', 'truck': 'crimson', 'bus': 'darkorange',
                           'vehicle': 'purple'}

            points = self.communicator.points_with_files
            counts = {}
            for v in vehicles:
                cls = v.get('cls', 'vehicle')
                counts[cls] = counts.get(cls, 0) + 1

            def _hav(lat1, lon1, lat2, lon2):
                R = 6371.0
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = (math.sin(dlat/2)**2 +
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
                return R * 2 * math.asin(math.sqrt(a))

            total_km = sum(_hav(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
                           for i in range(1, len(points)))

            with PdfPages(path) as pdf:

                # ── Page 1: Title + summary ───────────────────────────────────
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis('off')
                ax.text(0.5, 0.96, 'Vehicle Detection Report', ha='center', va='top',
                        fontsize=22, fontweight='bold', transform=ax.transAxes)
                ax.text(0.5, 0.91, f'Project: {self._project_name}', ha='center', va='top',
                        fontsize=14, color='#333333', transform=ax.transAxes)
                ax.text(0.5, 0.87, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                        ha='center', va='top', fontsize=11, color='gray', transform=ax.transAxes)
                ax.plot([0.05, 0.95], [0.84, 0.84], color='#cccccc', linewidth=1,
                        transform=ax.transAxes)
                y = 0.80
                ax.text(0.05, y, 'Survey Summary', fontsize=14, fontweight='bold',
                        transform=ax.transAxes)
                y -= 0.04
                for label, value in [
                    ('Total frames processed', f'{len(points):,}'),
                    ('Route distance',          f'{total_km:.2f} km'),
                    ('Frame interval',          f'{self.vehicle_interval_spin.value():.1f} sec'),
                    ('YOLO model',              self.vehicle_model_combo.currentText()),
                    ('Total vehicles detected', f'{len(vehicles)}'),
                ]:
                    ax.text(0.08, y, label + ':', fontsize=11, transform=ax.transAxes)
                    ax.text(0.55, y, value, fontsize=11, fontweight='bold', transform=ax.transAxes)
                    y -= 0.038
                ax.plot([0.05, 0.95], [y - 0.01, y - 0.01], color='#cccccc', linewidth=1,
                        transform=ax.transAxes)
                y -= 0.04
                ax.text(0.05, y, 'Detections by Class', fontsize=14, fontweight='bold',
                        transform=ax.transAxes)
                y -= 0.04
                for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
                    color = CLASS_COLOR.get(cls, 'purple')
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (0.08, y - 0.012), 0.03, 0.028,
                        boxstyle='round,pad=0.002', facecolor=color,
                        transform=ax.transAxes, zorder=3))
                    ax.text(0.13, y, f'{cls.capitalize()}', fontsize=11, transform=ax.transAxes)
                    ax.text(0.55, y, str(n), fontsize=11, fontweight='bold', transform=ax.transAxes)
                    pct = n / len(vehicles) * 100
                    ax.text(0.65, y, f'({pct:.0f}%)', fontsize=10, color='gray', transform=ax.transAxes)
                    y -= 0.038
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # ── Page 2: Route map with vehicle markers ────────────────────
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                lats = [p[0] for p in points]
                lons = [p[1] for p in points]
                pad_lat = (max(lats) - min(lats)) * 0.15 + 0.002
                pad_lon = (max(lons) - min(lons)) * 0.15 + 0.002
                ax.set_xlim(min(lons) - pad_lon, max(lons) + pad_lon)
                ax.set_ylim(min(lats) - pad_lat, max(lats) + pad_lat)
                try:
                    import contextily as ctx
                    ctx.add_basemap(ax, crs='EPSG:4326',
                                    source=ctx.providers.Esri.WorldImagery,
                                    attribution=False, zorder=0)
                except Exception:
                    try:
                        import contextily as ctx
                        ctx.add_basemap(ax, crs='EPSG:4326',
                                        source=ctx.providers.OpenStreetMap.Mapnik,
                                        attribution=False, zorder=0)
                    except Exception:
                        ax.set_facecolor('#e8f0f8')
                        ax.grid(True, alpha=0.4, linestyle='--', color='white')
                ax.plot(lons, lats, '-', color='white', linewidth=1.5, alpha=0.6, label='Route', zorder=2)
                ax.scatter([lons[0]], [lats[0]], c='lime', s=100, zorder=6,
                           marker='o', label='Start', edgecolors='black', linewidths=0.8)
                ax.scatter([lons[-1]], [lats[-1]], c='red', s=100, zorder=6,
                           marker='s', label='End', edgecolors='black', linewidths=0.8)
                for cls in sorted(counts):
                    vpts = [(v['lon'], v['lat']) for v in vehicles if v.get('cls') == cls]
                    if vpts:
                        xs, ys = zip(*vpts)
                        color = CLASS_COLOR.get(cls, 'purple')
                        ax.scatter(xs, ys, c=color, s=60, zorder=4, alpha=0.9,
                                   edgecolors='white', linewidths=0.8,
                                   label=f'{cls.capitalize()} ({counts[cls]})')
                ax.set_xlabel('Longitude', fontsize=10)
                ax.set_ylabel('Latitude', fontsize=10)
                ax.set_title(f'Vehicle Detections — {self._project_name}',
                             fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=9, framealpha=0.85)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # ── Page 3: Bar chart + detection table ───────────────────────
                fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(11.69, 8.27),
                                                      gridspec_kw={'width_ratios': [1, 1.5]})
                classes = sorted(counts.keys(), key=lambda c: -counts[c])
                vals    = [counts[c] for c in classes]
                colors  = [CLASS_COLOR.get(c, 'purple') for c in classes]
                bars = ax_bar.bar(classes, vals, color=colors, edgecolor='white', width=0.6)
                ax_bar.set_title('Vehicles by Class', fontsize=13, fontweight='bold')
                ax_bar.set_ylabel('Count')
                ax_bar.set_ylim(0, max(vals) * 1.2)
                for bar, val in zip(bars, vals):
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
                ax_tbl.axis('off')
                display = vehicles[:50]
                col_labels = ['#', 'Class', 'Latitude', 'Longitude']
                rows = [[str(i+1), v.get('cls', ''), f"{v['lat']:.5f}", f"{v['lon']:.5f}"]
                        for i, v in enumerate(display)]
                if len(vehicles) > 50:
                    rows.append(['...', f'+{len(vehicles)-50} more', '', ''])
                tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                                   loc='center', cellLoc='center')
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(7)
                tbl.scale(1, 1.15)
                for j in range(len(col_labels)):
                    tbl[0, j].set_facecolor('#333333')
                    tbl[0, j].set_text_props(color='white', fontweight='bold')
                for i in range(1, len(rows) + 1):
                    for j in range(len(col_labels)):
                        if i % 2 == 0:
                            tbl[i, j].set_facecolor('#f5f5f5')
                ax_tbl.set_title('Detection List', fontsize=13, fontweight='bold', pad=16)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # ── Pages 4+: vehicle frame images (6 per page) ───────────────
                image_dir = self.communicator.image_dir
                COLS, ROWS = 3, 2
                per_page   = COLS * ROWS
                for page_start in range(0, len(vehicles), per_page):
                    batch = vehicles[page_start:page_start + per_page]
                    fig, axes = plt.subplots(ROWS, COLS, figsize=(11.69, 8.27))
                    axes = [a for row in axes for a in row]
                    for a in axes:
                        a.axis('off')
                    for a, v in zip(axes, batch):
                        fpath = os.path.join(image_dir, v.get('frame', ''))
                        if os.path.exists(fpath):
                            try:
                                img = plt.imread(fpath)
                                a.imshow(img)
                                box = v.get('box')
                                if box:
                                    x1, y1, x2, y2 = box
                                    cls   = v.get('cls', 'vehicle')
                                    color = CLASS_COLOR.get(cls, 'purple')
                                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                          linewidth=2, edgecolor=color, facecolor='none')
                                    a.add_patch(rect)
                            except Exception:
                                a.text(0.5, 0.5, 'Image error', ha='center', va='center',
                                       transform=a.transAxes)
                        else:
                            a.text(0.5, 0.5, 'Not found', ha='center', va='center',
                                   transform=a.transAxes, color='gray')
                        cls   = v.get('cls', 'vehicle')
                        color = CLASS_COLOR.get(cls, 'purple')
                        a.set_title(f"{cls.capitalize()}  {v['lat']:.5f}, {v['lon']:.5f}",
                                    fontsize=8, color=color, fontweight='bold', pad=3)
                    page_num = page_start // per_page + 4
                    n_pages  = (len(vehicles) + per_page - 1) // per_page + 3
                    fig.suptitle(f"Detection Frames  (page {page_num} of {n_pages})",
                                 fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            self.coords_label.setText(f"Report saved: {path}")

        except ImportError:
            QMessageBox.critical(self, "Missing Library",
                                 "matplotlib is required.\nInstall with: pip install matplotlib")
        except Exception:
            import traceback
            QMessageBox.critical(self, "Report Error", traceback.format_exc())


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
    detection_finished = pyqtSignal(str)   # emits project_name when detect run completes

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
        eval_group  = QGroupBox("Evaluate  (uses val frames from selected training sets)")
        eval_layout = QFormLayout(eval_group)

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
        self.detect_project_combo.clear()
        for p in projects:
            self.detect_project_combo.addItem(p)
        if self._project_name and self._project_name in projects:
            self.detect_project_combo.setCurrentText(self._project_name)

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
        """Run YOLO detect on val frames from training sets, compare to labels."""
        if not self._selected_model:
            QMessageBox.warning(self, "No Model", "Select a model first.")
            return
        mwp = model_weights_path(self._selected_model)
        if not os.path.exists(mwp):
            QMessageBox.warning(self, "No Weights", "Model has not been trained yet.")
            return

        config        = load_model_config(self._selected_model)
        class_ids     = config.get('classes', [])
        train_set_ids = config.get('training_sets', [])
        classes       = load_classes()
        class_names   = [next((c['name'] for c in classes if c['id'] == cid), str(cid))
                         for cid in class_ids]
        id_to_idx     = {cid: i for i, cid in enumerate(class_ids)}
        conf          = self.conf_spin.value()

        # Collect val frames from all training sets that have a saved split
        # Each entry: (fname, frames_dir, labels_dir_for_set)
        val_entries = []
        for set_id in train_set_ids:
            meta  = load_training_set_meta(set_id)
            split = load_tset_split(set_id)
            val_frames = split.get('val', [])
            if not val_frames:
                # No saved split — fall back to all TP frames as a rough eval set
                val_frames = meta.get('tp_frames', [])
            proj       = meta.get('project', '')
            frames_dir = project_frames_dir(proj)
            lbl_dir    = training_set_labels_dir(set_id)
            for fname in val_frames:
                val_entries.append((fname, frames_dir, lbl_dir))

        if not val_entries:
            self.eval_result_lbl.setText("No val frames found in selected training sets.")
            return

        self.log_edit.append(f"Evaluating on {len(val_entries)} val frames "
                             f"from {len(train_set_ids)} training set(s)...")
        self.eval_result_lbl.setText("Running...")
        QApplication.processEvents()

        # Run detection — group by frames_dir to minimise subprocess calls
        from collections import defaultdict as _dd
        by_dir = _dd(list)
        for fname, fdir, ldir in val_entries:
            by_dir[fdir].append((fname, ldir))

        detections_by_frame = defaultdict(list)
        script_file = os.path.join(tempfile.gettempdir(), 'dcv_v1_detect_script.py')
        with open(script_file, 'w', encoding='utf-8') as sf:
            sf.write(_DETECT_SCRIPT)

        for frames_dir, entries in by_dir.items():
            fnames    = [e[0] for e in entries]
            coord_map = {f: [0.0, 0.0] for f in fnames}
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

            proc = subprocess.Popen(
                [sys.executable, script_file, params_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='replace',
            )
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

        # Compare detections to ground truth labels (from training set labels dir)
        # Build lookup: fname → labels_dir
        fname_to_ldir = {e[0]: e[1] for _, entries in by_dir.items() for e in entries}
        fname_to_fdir = {e[0]: fdir  for fdir, entries in by_dir.items() for e in entries}

        def iou(b1, b2):
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        tp = fp = fn = 0
        IOU_THRESH = 0.5
        all_val_fnames = [e[0] for _, entries in by_dir.items() for e in entries]

        for fname in all_val_fnames:
            ldir = fname_to_ldir[fname]
            fdir = fname_to_fdir[fname]
            # Load ground truth — parse label file directly from training set labels dir
            gt_boxes = []
            stem     = os.path.splitext(fname)[0]
            lbl_path = os.path.join(ldir, stem + '.txt')
            if os.path.exists(lbl_path):
                with open(lbl_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cid = int(parts[0])
                            if cid in id_to_idx:
                                cx, cy, w, h = map(float, parts[1:])
                                gt_boxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))

            preds      = detections_by_frame.get(fname, [])
            pred_boxes = [(d['box_x'], d['box_y'],
                           d['box_x']+d['box_w'], d['box_y']+d['box_h'])
                          for d in preds]

            # Normalise pred boxes to 0-1 space using jpeg_size
            size = jpeg_size(os.path.join(fdir, fname))
            if size:
                iw, ih = size
                pred_boxes_norm = [(x1/iw, y1/ih, x2/iw, y2/ih)
                                   for x1, y1, x2, y2 in pred_boxes]
            else:
                pred_boxes_norm = pred_boxes

            matched_gt   = set()
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
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.start()

    def _on_detect_finished(self, ok):
        proj = self.detect_project_combo.currentText()
        self.log_edit.append("Detection done." if ok else "Detection failed.")
        if ok and proj:
            self.detection_finished.emit(proj)


# ─── Vehicle Review Panel ─────────────────────────────────────────────────────

_VEHICLE_CLASS_COLOR_HEX = {
    'car':     '#4682b4',
    'truck':   '#dc143c',
    'bus':     '#ff8c00',
    'vehicle': '#800080',
}


class VehicleThumb(QLabel):
    """Clickable thumbnail showing a vehicle detection frame with bounding box overlay."""
    toggled = pyqtSignal(str, bool)   # frame filename, is_selected

    THUMB = 420

    def __init__(self, frames_dir, row):
        super().__init__()
        self.frame_name = row.get('frame', '')
        self._selected  = False
        sz = self.THUMB

        img_path = os.path.join(frames_dir, self.frame_name)
        px = QPixmap(img_path)
        if not px.isNull():
            try:
                x1 = float(row.get('box_x1', 0))
                y1 = float(row.get('box_y1', 0))
                x2 = float(row.get('box_x2', 0))
                y2 = float(row.get('box_y2', 0))
                px_draw = px.scaled(sz, sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                if x2 > x1 and y2 > y1:
                    sx = px_draw.width()  / px.width()
                    sy = px_draw.height() / px.height()
                    painter = QPainter(px_draw)
                    cls   = row.get('cls', 'vehicle')
                    color = QColor(_VEHICLE_CLASS_COLOR_HEX.get(cls, '#800080'))
                    pen   = QPen(color, max(2, sz // 75))
                    painter.setPen(pen)
                    painter.drawRect(int(x1*sx), int(y1*sy),
                                     int((x2-x1)*sx), int((y2-y1)*sy))
                    painter.end()
                self.setPixmap(px_draw)
            except Exception:
                self.setPixmap(px.scaled(sz, sz, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.setText("(missing)")
            self.setAlignment(Qt.AlignCenter)

        self.setFixedSize(sz + 10, sz + 36)
        self.setAlignment(Qt.AlignCenter)
        cls = row.get('cls', 'vehicle')
        lat = row.get('lat', '')
        lon = row.get('lon', '')
        self.setToolTip(f"{self.frame_name}\n{cls}  {lat}, {lon}")

        color_hex = _VEHICLE_CLASS_COLOR_HEX.get(cls, '#800080')
        lbl_cls = QLabel(cls.capitalize(), self)
        lbl_cls.setAlignment(Qt.AlignCenter)
        lbl_cls.setStyleSheet(f"font-size: 9px; color: {color_hex}; font-weight: bold;")
        lbl_cls.setGeometry(0, sz + 4, sz + 10, 14)

        short = self.frame_name[-28:] if len(self.frame_name) > 28 else self.frame_name
        lbl_frame = QLabel(short, self)
        lbl_frame.setAlignment(Qt.AlignCenter)
        lbl_frame.setStyleSheet("font-size: 8px; color: #aaa;")
        lbl_frame.setGeometry(0, sz + 18, sz + 10, 14)

        self._apply_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected = not self._selected
            self._apply_style()
            self.toggled.emit(self.frame_name, self._selected)

    def _apply_style(self):
        if self._selected:
            self.setStyleSheet("border: 3px solid #ff4444; background: rgba(255,50,50,40);")
        else:
            self.setStyleSheet("border: 2px solid #555; background: #2b2b2b;")


class VehicleReviewPanel(QWidget):
    """Review vehicle detections — click to select false positives, then remove them."""

    def __init__(self):
        super().__init__()
        self._project_name = None
        self._selected     = set()   # frame filenames marked for removal
        self._thumbs       = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Vehicle Detection Review"))
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        top.addWidget(refresh_btn)
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
        self._thumbs.clear()
        self._selected.clear()
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._project_name:
            return

        csv_path   = project_vehicles_csv(self._project_name)
        frames_dir = project_frames_dir(self._project_name)

        if not os.path.exists(csv_path):
            self.count_lbl.setText("No vehicles.csv — run Count Vehicles first.")
            return

        try:
            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            self.count_lbl.setText(f"Error reading CSV: {e}")
            return

        COLS = 4
        for i, row in enumerate(rows):
            thumb = VehicleThumb(frames_dir, row)
            thumb.toggled.connect(self._on_thumb_toggled)
            self.grid.addWidget(thumb, i // COLS, i % COLS)
            self._thumbs.append(thumb)

        self.count_lbl.setText(f"{len(rows)} detection(s)")

    def _on_thumb_toggled(self, frame_name, selected):
        if selected:
            self._selected.add(frame_name)
        else:
            self._selected.discard(frame_name)
        n     = len(self._selected)
        total = len(self._thumbs)
        self.count_lbl.setText(
            f"{total} detection(s)  |  {n} selected for removal" if n else f"{total} detection(s)"
        )

    def _remove_selected(self):
        if not self._selected:
            return
        reply = QMessageBox.question(
            self, "Remove Selected",
            f"Remove {len(self._selected)} detection(s) from vehicles.csv?\n"
            "(Affects the PDF report — does not delete frame images.)",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        csv_path = project_vehicles_csv(self._project_name)
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                reader     = csv.DictReader(f)
                fieldnames = reader.fieldnames
                kept       = [r for r in reader if r.get('frame') not in self._selected]
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept)
        self._refresh()


# ─── Review Panel ─────────────────────────────────────────────────────────────

class DetectionThumb(QLabel):
    """Clickable thumbnail showing a detection crop."""
    toggled = pyqtSignal(int, bool)   # row_index, is_selected

    def __init__(self, row_index, class_name, conf, result, thumb_size=150):
        super().__init__()
        self._row_index  = row_index
        self._selected   = False
        self._result     = result
        self._thumb_size = thumb_size

        self.setText("...")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(thumb_size + 10, thumb_size + 36)

        self._lbl = QLabel(f"{class_name}\n{conf:.2f}", self)
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet("font-size: 9px; color: #ddd;")
        self._lbl.setGeometry(0, thumb_size + 4, thumb_size + 10, 28)

        self._apply_style()

    def set_crop(self, qimage):
        """Called from main thread when background loader delivers the crop."""
        if qimage and not qimage.isNull():
            px = QPixmap.fromImage(qimage).scaled(
                self._thumb_size, self._thumb_size,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(px)
            self.setAlignment(Qt.AlignCenter)
            self.setText("")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected = not self._selected
            self._apply_style()
            self.toggled.emit(self._row_index, self._selected)

    def _apply_style(self):
        if self._selected:
            border = "border: 3px solid #4499ff;"   # blue selection always wins
        elif self._result == 'tp':
            border = "border: 3px solid #44cc44;"
        elif self._result == 'fp':
            border = "border: 3px solid #cc4444;"
        else:
            border = "border: 2px solid #555;"
        self.setStyleSheet(f"{border} background: #2b2b2b;")

    def set_result(self, result):
        self._result = result
        self._apply_style()


class ThumbLoaderWorker(QThread):
    """Loads detection crops in background, emits (index, QImage) one at a time."""
    crop_ready = pyqtSignal(int, object)   # index, QImage (or None)
    done       = pyqtSignal()

    def __init__(self, detections, frames_dir):
        super().__init__()
        self._detections = detections
        self._frames_dir = frames_dir
        self._cancelled  = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        frame_cache = {}   # filename → cv2 BGR image (one full frame loaded per unique file)
        for i, det in enumerate(self._detections):
            if self._cancelled:
                break
            fname = det.get('frame', '')
            try:
                bx = int(det['box_x']); by = int(det['box_y'])
                bw = int(det['box_w']); bh = int(det['box_h'])
                if bw <= 0 or bh <= 0:
                    self.crop_ready.emit(i, None)
                    continue
                if fname not in frame_cache:
                    img = cv2.imread(os.path.join(self._frames_dir, fname))
                    frame_cache[fname] = img   # may be None if file missing
                img = frame_cache[fname]
                if img is None:
                    self.crop_ready.emit(i, None)
                    continue
                ih, iw = img.shape[:2]
                # Clamp box to image bounds
                x1 = max(0, bx);          y1 = max(0, by)
                x2 = min(iw, bx + bw);    y2 = min(ih, by + bh)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    self.crop_ready.emit(i, None)
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ch, cw = crop_rgb.shape[:2]
                qimg = QImage(crop_rgb.data.tobytes(), cw, ch, cw * 3, QImage.Format_RGB888)
                self.crop_ready.emit(i, qimg.copy())   # copy so data outlives local array
            except Exception:
                self.crop_ready.emit(i, None)
        self.done.emit()


class ReviewPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._project_name  = None
        self._detections    = []
        self._selected      = set()
        self._thumbs        = []
        self._loader        = None   # ThumbLoaderWorker
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
        # Cancel any in-progress loader
        if self._loader is not None:
            self._loader.cancel()
            self._loader = None

        # Clear old widgets
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thumbs = []

        THUMB_SIZE = 150
        COLS = 6

        # Create all placeholder thumbnails immediately (no I/O, no freeze)
        for i, det in enumerate(self._detections):
            thumb = DetectionThumb(
                i,
                det.get('class_name', ''), float(det.get('conf', 0)),
                det.get('result', ''),
                THUMB_SIZE
            )
            thumb.toggled.connect(self._on_thumb_toggled)
            self.grid.addWidget(thumb, i // COLS, i % COLS)
            self._thumbs.append(thumb)

        self.count_lbl.setText(f"{len(self._detections)} detections")

        if not self._detections:
            return

        # Load crops in background; update thumbnails as each arrives
        frames_dir = project_frames_dir(self._project_name)
        self._loader = ThumbLoaderWorker(self._detections, frames_dir)
        self._loader.crop_ready.connect(self._on_crop_ready)
        self._loader.start()

    def _on_crop_ready(self, index, qimage):
        if index < len(self._thumbs):
            self._thumbs[index].set_crop(qimage)

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

            stem = os.path.splitext(frame)[0]
            line = f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

            # Write to training set labels dir (for training)
            with open(os.path.join(lbl_dir, stem + '.txt'), 'a') as lf:
                lf.write(line + '\n')

            # Also write to project labels dir so Label tab shows these boxes
            proj_lbl = os.path.join(project_labels_dir(self._project_name), stem + '.txt')
            os.makedirs(project_labels_dir(self._project_name), exist_ok=True)
            existing_lines = []
            if os.path.exists(proj_lbl):
                with open(proj_lbl) as lf:
                    existing_lines = [l.strip() for l in lf if l.strip()]
            if line not in existing_lines:
                with open(proj_lbl, 'a') as lf:
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
        sidebar.setFixedWidth(150)
        sidebar.setStyleSheet("background: #2a2a3a;")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(4, 8, 4, 8)
        sb_layout.setSpacing(4)

        self._panel_buttons = []
        panel_names = ["Process", "View", "Label", "Models", "Review", "Vehicles"]
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

        poi_lbl = QLabel("POIs")
        poi_lbl.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px 2px 2px 2px;")
        sb_layout.addWidget(poi_lbl)

        self.poi_list = QListWidget()
        self.poi_list.setStyleSheet(
            "QListWidget { background: #1e1e2e; border: 1px solid #444; color: #ddd; font-size: 11px; }"
            "QListWidget::item { padding: 3px 4px; }"
            "QListWidget::item:selected { background: #5566cc; color: white; }"
            "QListWidget::item:hover { background: #3a3a5a; }"
        )
        self.poi_list.setMinimumHeight(140)
        self.poi_list.itemClicked.connect(self._on_poi_list_clicked)
        sb_layout.addWidget(self.poi_list)

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

        self.view_panel            = ViewPanel()
        self.label_panel           = LabelPanel()
        self.models_panel          = ModelsPanel()
        self.review_panel          = ReviewPanel()
        self.vehicle_review_panel  = VehicleReviewPanel()

        self.models_panel.detection_finished.connect(self.review_panel.load_project)
        self.view_panel.vehicles_counted.connect(self.vehicle_review_panel.load_project)
        self.view_panel.poi_changed.connect(self._refresh_poi_list)

        self.stack.addWidget(self.process_panel)          # 0
        self.stack.addWidget(self.view_panel)              # 1
        self.stack.addWidget(self.label_panel)             # 2
        self.stack.addWidget(self.models_panel)            # 3
        self.stack.addWidget(self.review_panel)            # 4
        self.stack.addWidget(self.vehicle_review_panel)    # 5

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
            elif index == 5:
                self.vehicle_review_panel.load_project(name)

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
        elif idx == 5:
            self.vehicle_review_panel.load_project(name)

    def _refresh_poi_list(self, project_name):
        self.poi_list.clear()
        if not project_name:
            return
        poi_csv = project_poi_csv(project_name)
        if not os.path.exists(poi_csv):
            return
        try:
            with open(poi_csv, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    desc = row.get('description', '').strip() or f"{float(row['lat']):.4f}, {float(row['lon']):.4f}"
                    item = QListWidgetItem(desc)
                    item.setData(Qt.UserRole, (float(row['lat']), float(row['lon'])))
                    item.setToolTip(f"{desc}\n{row['lat']}, {row['lon']}")
                    self.poi_list.addItem(item)
        except Exception:
            pass

    def _on_poi_list_clicked(self, item):
        coords = item.data(Qt.UserRole)
        if coords:
            self.view_panel.goto_poi(coords[0], coords[1])
        # Switch to View tab if not already there
        if self.stack.currentIndex() != 1:
            self._switch_panel(1)

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
