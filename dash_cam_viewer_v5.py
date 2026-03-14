#!/usr/bin/env python3
"""
Dash Cam Viewer v5 - Combined Frame Extractor and Map Viewer
Adds manual feature identification with box selection and map marking.
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
    QInputDialog, QFormLayout, QGroupBox, QCheckBox, QColorDialog, QRubberBand
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
GLOBAL_SETTINGS_PATH = os.path.join(PROJECTS_DIR, "settings.json")
CULVERT_SETTINGS_PATH = os.path.join(PROJECTS_DIR, "culvert_settings.json")
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


def load_culvert_settings():
    if os.path.exists(CULVERT_SETTINGS_PATH):
        with open(CULVERT_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def save_culvert_settings(settings):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    with open(CULVERT_SETTINGS_PATH, "w") as f:
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


def project_culverts_csv(name):
    return os.path.join(PROJECTS_DIR, name, "culverts.csv")


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

            print("DEBUG: entering ThreadPoolExecutor", flush=True)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                print("DEBUG: submitting jobs", flush=True)
                future_map = {executor.submit(self._extract_video, mp4, prefix, log_queue): mp4
                              for mp4, prefix in jobs}
                pending = set(future_map)
                print(f"DEBUG: {len(pending)} futures submitted, entering poll loop", flush=True)

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
                        print(f"DEBUG: future done, collecting result", flush=True)
                        try:
                            count, entries = future.result()
                            total_saved += count
                            all_entries.extend(entries)
                            print(f"DEBUG: collected {count} frames, {len(entries)} entries", flush=True)
                        except Exception as e:
                            self.log.emit(f"  Worker error: {e}")
                            print(f"DEBUG: worker exception: {e}", flush=True)

                print("DEBUG: poll loop done, doing final log drain", flush=True)
                # Final log drain
                while True:
                    try:
                        self.log.emit(log_queue.get_nowait())
                    except _queue.Empty:
                        break

            print("DEBUG: exited ThreadPoolExecutor context (all threads joined)", flush=True)

            # Write txt file from QThread — no thread conflicts
            print("DEBUG: writing txt file", flush=True)
            with open(self.txt_path, "w") as f:
                f.write("Filename\tLatitude\tLongitude\n")
                for filename, lat, lon in all_entries:
                    f.write(f"{filename}\t{lat}\t{lon}\n")
            print("DEBUG: txt file written", flush=True)

            self.log.emit(f"\nDone. Total frames saved: {total_saved}")
            self.finished.emit(True)
            print("DEBUG: finished.emit(True) called", flush=True)

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

        print(f"DEBUG: _extract_video START {os.path.basename(video_path)}", flush=True)
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
        print(f"DEBUG: _extract_video END {os.path.basename(video_path)} — {saved_count} frames", flush=True)
        log(f"  Finished. {saved_count} frames saved.")
        return saved_count, entries




# ─── Selectable Frame Label ───────────────────────────────────────────────────

class SelectableFrameLabel(QLabel):
    """QLabel that supports rubber-band box selection and persistent box overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rb = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = QPoint()
        self._box = QRect()         # in widget coords
        self._norm_box = None       # normalized image coords (x,y,w,h) 0-1
        self._orig_size = None      # (orig_w, orig_h) of last displayed image
        self._selecting = False
        self._sel_enabled = False
        self.show_box = True

    def enable_selection(self):
        self._sel_enabled = True
        self.setCursor(Qt.CrossCursor)

    def set_orig_size(self, orig_w, orig_h):
        """Call after loading a new image. Restores box from saved norm coords if set."""
        self._orig_size = (orig_w, orig_h)
        if self._norm_box:
            self._recompute_box()

    def set_norm_box(self, norm):
        """Restore box from normalized dict {x,y,w,h}. Applied on next set_orig_size call."""
        self._norm_box = norm
        if self._orig_size:
            self._recompute_box()

    def get_norm_box(self):
        """Return current box as normalized {x,y,w,h} dict, or None."""
        if not self._orig_size:
            return None
        r = self.get_crop_rect(*self._orig_size)
        if not r or not r.isValid():
            return None
        ow, oh = self._orig_size
        return {'x': r.x()/ow, 'y': r.y()/oh, 'w': r.width()/ow, 'h': r.height()/oh}

    def _recompute_box(self):
        """Recompute widget-coord _box from _norm_box and _orig_size."""
        if not self._norm_box or not self._orig_size:
            return
        orig_w, orig_h = self._orig_size
        lw, lh = self.width(), self.height()
        if lw <= 0 or lh <= 0:
            return
        scale = min(lw / orig_w, lh / orig_h)
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        ox, oy = (lw - sw) // 2, (lh - sh) // 2
        n = self._norm_box
        self._box = QRect(
            ox + int(n['x'] * sw), oy + int(n['y'] * sh),
            int(n['w'] * sw),      int(n['h'] * sh)
        )
        self.update()

    def get_crop_rect(self, orig_w, orig_h):
        """Return selected box as QRect in original image pixel coords, or None."""
        if not self._box.isValid() or orig_w <= 0 or orig_h <= 0:
            return None
        lw, lh = self.width(), self.height()
        scale = min(lw / orig_w, lh / orig_h)
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        ox, oy = (lw - sw) // 2, (lh - sh) // 2
        b = self._box.intersected(QRect(ox, oy, sw, sh))
        if not b.isValid():
            return None
        inv = orig_w / sw
        return QRect(
            max(0, int((b.x() - ox) * inv)),
            max(0, int((b.y() - oy) * inv)),
            min(int(b.width() * inv), orig_w),
            min(int(b.height() * inv), orig_h),
        )

    def mousePressEvent(self, event):
        if self._sel_enabled and event.button() == Qt.LeftButton:
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
            self._sel_enabled = False
            self.setCursor(Qt.ArrowCursor)
            self._box = QRect(self._origin, event.pos()).normalized()
            self._norm_box = self.get_norm_box()  # keep norm in sync
            self._rb.hide()
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._norm_box and self._orig_size:
            self._recompute_box()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.show_box and self._box.isValid():
            p = QPainter(self)
            p.setPen(QPen(QColor(255, 50, 50), 2, Qt.DashLine))
            p.drawRect(self._box.adjusted(0, 0, -1, -1))


# ─── Map Communicator ─────────────────────────────────────────────────────────

class Communicator(QObject):
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

        self.image_label = SelectableFrameLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label, stretch=1)

        btn_row = QHBoxLayout()
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
        self.center_toggle.setStyleSheet(
            "QPushButton { padding: 0 8px; }"
            "QPushButton:checked { background: #4a9eff; color: white; }"
        )
        for w in [self.auto_left, self.left_btn, self.spin_box, self.right_btn, self.auto_right, self.center_toggle]:
            btn_row.addWidget(w)
        btn_row.addStretch()
        image_layout.addLayout(btn_row)

        # ── Culvert detection row ─────────────────────────────────────────────
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Culvert Detection"))
        self.select_box_btn = QPushButton("Select Box")
        self.select_box_btn.clicked.connect(self._select_box)
        id_row.addWidget(self.select_box_btn)
        self.show_box_chk = QCheckBox("Show Box")
        self.show_box_chk.setChecked(True)
        self.show_box_chk.toggled.connect(self._toggle_show_box)
        id_row.addWidget(self.show_box_chk)
        self.acquire_btn = QPushButton("Acquire")
        self.acquire_btn.clicked.connect(self._acquire)
        id_row.addWidget(self.acquire_btn)
        self.remove_culvert_btn = QPushButton("Remove")
        self.remove_culvert_btn.clicked.connect(self._remove_culvert)
        id_row.addWidget(self.remove_culvert_btn)
        self.id_shape_combo = QComboBox()
        self.id_shape_combo.addItems(["Circle", "Square", "Triangle"])
        id_row.addWidget(self.id_shape_combo)
        self.id_color_combo = QComboBox()
        self.id_color_combo.addItems(["red", "blue", "green", "orange", "purple", "yellow"])
        id_row.addWidget(self.id_color_combo)
        self.id_size_combo = QComboBox()
        self.id_size_combo.addItems(["Small", "Medium", "Large"])
        self.id_size_combo.setCurrentText("Medium")
        id_row.addWidget(self.id_size_combo)
        id_row.addStretch()
        image_layout.addLayout(id_row)

        self.splitter.addWidget(image_widget)
        self._load_culvert_settings()
        self.splitter.setSizes([500, 900])

    # ── Culvert detection helpers ─────────────────────────────────────────────

    def _load_culvert_settings(self):
        s = load_culvert_settings()
        if 'shape' in s:
            self.id_shape_combo.setCurrentText(s['shape'])
        if 'color' in s:
            self.id_color_combo.setCurrentText(s['color'])
        if 'size' in s:
            self.id_size_combo.setCurrentText(s['size'])
        if 'box' in s:
            self.image_label.set_norm_box(s['box'])

    def _save_culvert_settings(self):
        s = {
            'shape': self.id_shape_combo.currentText(),
            'color': self.id_color_combo.currentText(),
            'size':  self.id_size_combo.currentText(),
            'box':   self.image_label.get_norm_box(),
        }
        save_culvert_settings(s)

    def _remove_culvert(self):
        if not self.communicator or self.communicator.current_index is None:
            QMessageBox.warning(self, "No Frame", "Select a frame on the map first.")
            return

        _, _, fname = self.communicator.points_with_files[self.communicator.current_index]
        csv_path = project_culverts_csv(self._project_name)

        if not os.path.exists(csv_path):
            self.coords_label.setText("No culverts recorded for this project.")
            return

        # Read all rows, find matching ones
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            all_rows = list(reader)

        matching = [r for r in all_rows if r.get('source_frame') == fname]
        if not matching:
            self.coords_label.setText(f"No culvert recorded for current frame.")
            return

        # Delete saved images
        culverts_dir = os.path.join(PROJECTS_DIR, self._project_name, "culverts")
        for row in matching:
            img_file = os.path.join(culverts_dir, row.get('saved_image', ''))
            if os.path.exists(img_file):
                os.remove(img_file)

        # Rewrite CSV without matching rows
        remaining = [r for r in all_rows if r.get('source_frame') != fname]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(remaining)

        # Remove marker from map
        safe_frame = fname.replace("'", "\\'")
        js = (f"if (window._culvertMarkers && window._culvertMarkers['{safe_frame}']) {{"
              f"  window._culvertMarkers['{safe_frame}'].remove();"
              f"  delete window._culvertMarkers['{safe_frame}'];"
              f"}}")
        self.web_view.page().runJavaScript(js)
        self.coords_label.setText(f"Culvert removed for frame: {fname}")

    def _select_box(self):
        self.image_label.enable_selection()

    def _toggle_show_box(self, checked):
        self.image_label.show_box = checked
        self.image_label.update()

    def _acquire(self):
        if not self.communicator or self.communicator.current_index is None:
            QMessageBox.warning(self, "No Frame", "Select a frame on the map first.")
            return

        lat, lon, fname = self.communicator.points_with_files[self.communicator.current_index]
        name = "culvert"

        img_path = os.path.join(self.communicator.image_dir, fname)
        orig = QPixmap(img_path)
        if orig.isNull():
            QMessageBox.warning(self, "Error", f"Cannot load image:\n{img_path}")
            return

        crop_rect = self.image_label.get_crop_rect(orig.width(), orig.height())
        if crop_rect and crop_rect.isValid() and crop_rect.width() > 0 and crop_rect.height() > 0:
            cropped = orig.copy(crop_rect)
            bx, by, bw, bh = crop_rect.x(), crop_rect.y(), crop_rect.width(), crop_rect.height()
        else:
            cropped = orig
            bx = by = bw = bh = ""

        # Save cropped image
        culverts_dir = os.path.join(PROJECTS_DIR, self._project_name, "culverts")
        os.makedirs(culverts_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{name}_{ts}.jpg"
        cropped.save(os.path.join(culverts_dir, img_filename), "JPEG", 90)

        # Append to CSV
        csv_path = project_culverts_csv(self._project_name)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["name", "latitude", "longitude", "source_frame", "saved_image",
                             "box_x", "box_y", "box_w", "box_h"])
            w.writerow([name, lat, lon, fname, img_filename, bx, by, bw, bh])

        # Add marker on map
        shape = self.id_shape_combo.currentText()
        color = self.id_color_combo.currentText()
        radius = {"Small": 6, "Medium": 10, "Large": 14}[self.id_size_combo.currentText()]
        mv = self.communicator.map_var
        safe_name = name.replace("'", "\\'")
        popup = f"'{safe_name}<br>{lat:.5f}, {lon:.5f}'"

        safe_frame = fname.replace("'", "\\'")
        key = f"window._culvertMarkers = window._culvertMarkers || {{}}; window._culvertMarkers['{safe_frame}'] = "
        if shape == "Circle":
            js = (key +
                  f"L.circleMarker([{lat},{lon}],"
                  f"{{radius:{radius},color:'white',weight:1.5,"
                  f"fillColor:'{color}',fillOpacity:0.9}})"
                  f".bindPopup({popup}).addTo({mv});")
        else:
            sz = radius * 2
            if shape == "Square":
                svg_shape = f'<rect width="{sz}" height="{sz}" fill="{color}" stroke="white" stroke-width="1.5"/>'
            else:  # Triangle
                pts = f"0,{sz} {sz},{sz} {sz//2},0"
                svg_shape = f'<polygon points="{pts}" fill="{color}" stroke="white" stroke-width="1.5"/>'
            svg = f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg">{svg_shape}</svg>'
            js = (key +
                  f"L.marker([{lat},{lon}],{{icon:L.divIcon({{"
                  f"html:'{svg}',iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                  f").bindPopup({popup}).addTo({mv});")

        self.web_view.page().runJavaScript(js)
        self._save_culvert_settings()
        self.coords_label.setText(
            f"Culvert acquired  Lat: {lat:.5f}  Lon: {lon:.5f}  — saved to culverts/"
        )

    def load_project(self, project_name):
        import traceback
        try:
            self._load_project_inner(project_name)
        except Exception as e:
            msg = traceback.format_exc()
            print(msg)
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

        # ── Inject saved culvert markers ──────────────────────────────────────
        culverts_csv = project_culverts_csv(project_name)
        if os.path.exists(culverts_csv):
            cs = load_culvert_settings()
            cv_shape  = cs.get('shape', 'Circle')
            cv_color  = cs.get('color', 'red')
            cv_radius = {'Small': 6, 'Medium': 10, 'Large': 14}.get(cs.get('size', 'Medium'), 10)
            culvert_entries = []
            try:
                with open(culverts_csv, newline='') as f:
                    for row in csv.DictReader(f):
                        try:
                            culvert_entries.append((
                                float(row['latitude']), float(row['longitude']),
                                row['name'], row['source_frame']
                            ))
                        except (KeyError, ValueError):
                            pass
            except Exception:
                pass
            if culvert_entries:
                markers_js = ["window._culvertMarkers = window._culvertMarkers || {};"]
                for clat, clon, cname, src_frame in culvert_entries:
                    safe = cname.replace("'", "\\'")
                    safe_frame = src_frame.replace("'", "\\'")
                    popup = f"'{safe}<br>{clat:.5f}, {clon:.5f}'"
                    if cv_shape == 'Circle':
                        stmt = (f"window._culvertMarkers['{safe_frame}'] = "
                                f"L.circleMarker([{clat},{clon}],"
                                f"{{radius:{cv_radius},color:'white',weight:1.5,"
                                f"fillColor:'{cv_color}',fillOpacity:0.9}})"
                                f".bindPopup({popup}).addTo({map_var});")
                    else:
                        sz = cv_radius * 2
                        if cv_shape == 'Square':
                            shape_svg = f'<rect width="{sz}" height="{sz}" fill="{cv_color}" stroke="white" stroke-width="1.5"/>'
                        else:
                            pts = f"0,{sz} {sz},{sz} {sz//2},0"
                            shape_svg = f'<polygon points="{pts}" fill="{cv_color}" stroke="white" stroke-width="1.5"/>'
                        svg = f'<svg width="{sz}" height="{sz}" xmlns="http://www.w3.org/2000/svg">{shape_svg}</svg>'
                        stmt = (f"window._culvertMarkers['{safe_frame}'] = "
                                f"L.marker([{clat},{clon}],{{icon:L.divIcon({{"
                                f"html:'{svg}',iconSize:[{sz},{sz}],iconAnchor:[{sz//2},{sz//2}],className:''}})}}"
                                f").bindPopup({popup}).addTo({map_var});")
                    markers_js.append(stmt)

                culvert_js = (
                    "<script>\nsetTimeout(function() {\n"
                    + "\n".join(markers_js)
                    + "\n}, 600);\n</script>"
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
        for btn in [self.process_btn, self.view_btn_side]:
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

        self.stack.addWidget(self.process_panel)  # index 0
        self.stack.addWidget(self.view_panel)      # index 1
        body_layout.addWidget(self.stack, stretch=1)
        root.addWidget(body, stretch=1)

        self.process_btn.clicked.connect(lambda: self._switch_panel(0))
        self.view_btn_side.clicked.connect(self._switch_to_view)
        self._switch_panel(0)

    def _switch_panel(self, index):
        self.stack.setCurrentIndex(index)
        self.process_btn.setChecked(index == 0)
        self.view_btn_side.setChecked(index == 1)

    def _switch_to_view(self):
        if self.current_project:
            self.view_panel.load_project(self.current_project["name"])
        self._switch_panel(1)

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
