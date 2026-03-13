#!/usr/bin/env python3
"""
Dash Cam Viewer - Combined Frame Extractor and Map Viewer
"""

import sys
import os
import json
import shutil
import cv2
import math
import re
import subprocess
import statistics
from datetime import timedelta
from pathlib import Path
from collections import defaultdict
from io import BytesIO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QMessageBox,
    QDesktopWidget, QSplitter, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QComboBox, QLineEdit, QFileDialog, QTextEdit, QStackedWidget,
    QInputDialog, QFormLayout, QGroupBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import pyqtSlot, QObject, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont

import folium

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECTS_DIR = str(Path(__file__).parent / "dash_cam_projects")
GLOBAL_SETTINGS_PATH = os.path.join(PROJECTS_DIR, "settings.json")
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

            with open(self.txt_path, "w") as f:
                f.write("Filename\tLatitude\tLongitude\n")

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

            total_saved = 0
            for mp4 in mp4_files:
                if self._abort:
                    self.log.emit("Aborted by user.")
                    self.finished.emit(False)
                    return
                rel = os.path.relpath(os.path.dirname(mp4), self.video_dir).replace('\\', '_').replace('/', '_')
                base = os.path.splitext(os.path.basename(mp4))[0]
                prefix = f"{rel}_{base}" if rel != '.' else base
                total_saved += self._process_video(mp4, prefix)

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
        except Exception as e:
            self.log.emit(f"  GPS extraction failed: {e}")
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

    def _process_video(self, video_path, prefix):
        self.log.emit(f"\nProcessing: {os.path.basename(video_path)}")
        gps_data = self._extract_gps(video_path)
        self.log.emit(f"  GPS points: {len(gps_data)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log.emit("  Cannot open video.")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not math.isfinite(fps):
            fps = FPS_FALLBACK
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        self.log.emit(f"  FPS: {fps:.1f} | Frames: {total_frames} | Duration: {timedelta(seconds=int(duration_sec))}")

        saved_count = 0
        frame_idx = 0
        last_saved_sec = -1

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
                with open(self.txt_path, "a") as f:
                    f.write(f"{filename}\t{lat}\t{lon}\n")
                self.log.emit(f"  Saved: {filename}  Lat: {lat}  Lon: {lon}")

            frame_idx += 1

        cap.release()
        self.log.emit(f"  Finished. {saved_count} frames saved.")
        return saved_count


# ─── Map Communicator ─────────────────────────────────────────────────────────

class Communicator(QObject):
    def __init__(self, coords_label, image_label, points_with_files, web_view, map_var, spin_box, image_dir):
        super().__init__()
        self.coords_label = coords_label
        self.image_label = image_label
        self.points_with_files = sorted(points_with_files, key=lambda x: x[2])
        self.image_dir = image_dir
        self.current_index = None
        self.web_view = web_view
        self.map_var = map_var
        self.spin_box = spin_box
        self.prev_timer = QTimer()
        self.prev_timer.timeout.connect(self.show_prev)
        self.next_timer = QTimer()
        self.next_timer.timeout.connect(self.show_next)

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
            size = self.image_label.size()
            if size.width() > 0 and size.height() > 0:
                pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)
        else:
            self.image_label.setText(f"Image not found:\n{full_path}")
        self.coords_label.setText(
            f"Latitude: {lat:.6f}  Longitude: {lon:.6f}\nClosest Image: {fname}"
        )
        self.update_marker(lat, lon)

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

    def update_marker(self, lat, lon):
        js = f"""
        if (window.marker) {{ {self.map_var}.removeLayer(window.marker); }}
        window.marker = L.circleMarker([{lat}, {lon}], {{
            color: 'yellow', fillColor: 'yellow', fillOpacity: 0.8, radius: 10
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
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        form_group = QGroupBox("Settings")
        form = QFormLayout(form_group)
        form.setSpacing(8)

        # Video source folder
        src_row = QHBoxLayout()
        self.src_edit = QLineEdit()
        src_browse = QPushButton("Browse")
        src_browse.clicked.connect(self._browse_src)
        src_row.addWidget(self.src_edit)
        src_row.addWidget(src_browse)
        form.addRow("Video Folder:", src_row)

        # ExifTool path
        exif_row = QHBoxLayout()
        self.exif_edit = QLineEdit(self.global_settings.get("exiftool_path", ""))
        exif_browse = QPushButton("Browse")
        exif_browse.clicked.connect(self._browse_exif)
        exif_row.addWidget(self.exif_edit)
        exif_row.addWidget(exif_browse)
        form.addRow("ExifTool Path:", exif_row)

        # Frame interval
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(1)
        self.interval_spin.setSuffix(" sec")
        self.interval_spin.setFixedWidth(100)
        form.addRow("Frame Interval:", self.interval_spin)

        # JPEG quality
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(92)
        self.quality_spin.setFixedWidth(100)
        form.addRow("JPEG Quality:", self.quality_spin)

        layout.addWidget(form_group)

        # Action buttons
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

        # Log output
        log_label = QLabel("Log:")
        layout.addWidget(log_label)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_edit, stretch=1)

    def load_project(self, project_data):
        self.src_edit.setText(project_data.get("video_source", ""))
        self.interval_spin.setValue(project_data.get("frame_interval", 1))
        self.quality_spin.setValue(project_data.get("jpeg_quality", 92))
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

        # Save settings into project file
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

        # Map container — holds a fresh QWebEngineView on each project load
        self.map_container = QWidget()
        self.map_layout = QVBoxLayout(self.map_container)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = None
        self.splitter.addWidget(self.map_container)

        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = QLabel()
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
        for w in [self.auto_left, self.left_btn, self.spin_box, self.right_btn, self.auto_right]:
            btn_row.addWidget(w)
        btn_row.addStretch()
        image_layout.addLayout(btn_row)

        self.splitter.addWidget(image_widget)
        self.splitter.setSizes([500, 900])

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
        # Stop any running timers from the previous project and clear the image
        if self.communicator:
            self.communicator.prev_timer.stop()
            self.communicator.next_timer.stop()
            self.communicator = None
        self.image_label.clear()
        self.coords_label.setText("Click on the map to get coordinates")
        # Destroy old web view and create a fresh one to avoid WebEngine crash
        if self.web_view is not None:
            self.map_layout.removeWidget(self.web_view)
            self.web_view.setParent(None)
            self.web_view.deleteLater()
        self.web_view = QWebEngineView()
        self.map_layout.addWidget(self.web_view)
        self.splitter.setSizes([500, 900])

        txt_path = project_txt_path(project_name)
        frames_dir = project_frames_dir(project_name)

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
                    color='red', weight=5, opacity=0.8,
                    popup=f"Section: {section_name}"
                ).add_to(m)

        folium.LayerControl().add_to(m)
        map_var = m.get_name()

        # Set up web channel
        self.channel = QWebChannel()
        self.web_view.page().setWebChannel(self.channel)

        self.communicator = Communicator(
            self.coords_label, self.image_label,
            points_with_files, self.web_view,
            map_var, self.spin_box, frames_dir
        )
        self.channel.registerObject('communicator', self.communicator)

        # Reconnect buttons to new communicator
        for btn in [self.auto_left, self.left_btn, self.right_btn, self.auto_right]:
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

        out = BytesIO()
        m.save(out, close_file=False)
        self.web_view.setHtml(out.getvalue().decode('utf-8'))
        self.coords_label.setText("Click on the map to get coordinates")

    def _read_points(self, txt_path):
        grouped_points = defaultdict(list)
        points_with_files = []
        with open(txt_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    lat = float(parts[-2])
                    lon = float(parts[-1])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and lat != -180 and lon != -180):
                        continue
                    filename = ' '.join(parts[:-2])
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
        # Restore last used project
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

        # ── Top bar ──
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

        # ── Body ──
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar
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

        # Content stack
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
            # If currently on the View tab, reload the map for the new project
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
