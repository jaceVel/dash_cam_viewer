import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QMessageBox, QDesktopWidget, QSplitter, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import pyqtSlot, QObject, Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont
import folium
import os
import statistics
import math
from collections import defaultdict
from io import BytesIO

# Function to read lat/lon points from frames_latlon.txt and group by section name
def read_grouped_points(txt_path):
    grouped_points = defaultdict(list)
    points_with_files = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        # Skip first line, process from line 2 onwards
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    lat = float(parts[-2])
                    lon = float(parts[-1])
                    filename = ' '.join(parts[:-2])  # Join all parts except the last two as filename
                    if '_frame_' in filename:
                        section_name = filename.split('_frame_')[0]
                        grouped_points[section_name].append((lat, lon))
                        points_with_files.append((lat, lon, filename))
                except ValueError:
                    pass
        if not grouped_points:
            raise ValueError("No valid lat/lon points found in the input file")
        return grouped_points, points_with_files
    except Exception as e:
        raise RuntimeError(f"Error reading {txt_path}: {e}")

class Communicator(QObject):
    def __init__(self, coords_label, image_label, points_with_files, web_view, map_var, spin_box):
        super().__init__()
        self.coords_label = coords_label
        self.image_label = image_label
        self.points_with_files = sorted(points_with_files, key=lambda x: x[2])
        self.image_dir = 'combined_frames'
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
        # Parse the coordinates
        lines = coords.split('\n')
        if len(lines) >= 2:
            try:
                lat_str = lines[0].split(': ', 1)[1]
                lon_str = lines[1].split(': ', 1)[1]
                clicked_lat = float(lat_str)
                clicked_lon = float(lon_str)

                # Find the closest point and its index
                min_dist = float('inf')
                closest_index = None
                for i, (lat, lon, fname) in enumerate(self.points_with_files):
                    dist = math.sqrt((lat - clicked_lat)**2 + (lon - clicked_lon)**2)
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
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setAlignment(Qt.AlignCenter)
        else:
            self.image_label.setText("Image not found")
        display_text = f"Latitude: {lat:.6f} Longitude: {lon:.6f}\nClosest Image: {fname}"
        self.coords_label.setText(display_text)
        self.update_marker(lat, lon)

    @pyqtSlot()
    def show_prev(self):
        if self.current_index is None:
            QMessageBox.information(None, "No Selection", "Please click on the map to select a starting point.")
            return
        if self.current_index <= 0:
            if self.prev_timer.isActive():
                self.prev_timer.stop()
            QMessageBox.information(None, "No Image", "Already at the first frame.")
            return
        self.current_index -= 1
        self._show_frame(self.current_index)

    @pyqtSlot()
    def show_next(self):
        if self.current_index is None:
            QMessageBox.information(None, "No Selection", "Please click on the map to select a starting point.")
            return
        if self.current_index >= len(self.points_with_files) - 1:
            if self.next_timer.isActive():
                self.next_timer.stop()
            QMessageBox.information(None, "No Image", "Already at the last frame.")
            return
        self.current_index += 1
        self._show_frame(self.current_index)

    def update_marker(self, lat, lon):
        js_code = f"""
        if (window.marker) {{
            {self.map_var}.removeLayer(window.marker);
        }}
        window.marker = L.circleMarker([{lat}, {lon}], {{
            color: 'yellow',
            fillColor: 'yellow',
            fillOpacity: 0.8,
            radius: 10
        }}).addTo({self.map_var});
        """
        self.web_view.page().runJavaScript(js_code)

    @pyqtSlot()
    def toggle_auto_prev(self):
        if self.prev_timer.isActive():
            self.prev_timer.stop()
        else:
            if self.next_timer.isActive():
                self.next_timer.stop()
            self.prev_timer.start(self.spin_box.value())

    @pyqtSlot()
    def toggle_auto_next(self):
        if self.next_timer.isActive():
            self.next_timer.stop()
        else:
            if self.prev_timer.isActive():
                self.prev_timer.stop()
            self.next_timer.start(self.spin_box.value())

    @pyqtSlot(int)
    def update_interval(self, value):
        if self.prev_timer.isActive():
            self.prev_timer.setInterval(value)
        if self.next_timer.isActive():
            self.next_timer.setInterval(value)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Satellite Map GUI")
        screen = QDesktopWidget().screenGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        self.setMinimumSize(800, 600)
        self.center()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical splitter
        main_splitter = QSplitter(Qt.Vertical)
        central_widget.setLayout(QVBoxLayout())
        central_widget.layout().addWidget(main_splitter)

        # Top widget for coordinates label
        top_widget = QWidget()
        top_layout = QVBoxLayout()
        top_widget.setLayout(top_layout)
        self.coords_label = QLabel("Click on the map to get coordinates")
        self.coords_label.setFont(QFont("Arial", 24))
        top_layout.addWidget(self.coords_label)
        main_splitter.addWidget(top_widget)

        # Bottom splitter for map and image
        bottom_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(bottom_splitter)

        # Web view for displaying the map
        self.web_view = QWebEngineView()
        bottom_splitter.addWidget(self.web_view)

        # Image container with label and buttons
        image_widget = QWidget()
        image_layout = QVBoxLayout()
        image_widget.setLayout(image_layout)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label, stretch=1)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        auto_left_button = QPushButton("Auto ←")
        buttons_layout.addWidget(auto_left_button)

        left_button = QPushButton("← Prev")
        buttons_layout.addWidget(left_button)

        self.spin_box = QSpinBox()
        self.spin_box.setRange(100, 5000)
        self.spin_box.setValue(1000)
        self.spin_box.setSingleStep(100)
        self.spin_box.setSuffix(" ms")
        buttons_layout.addWidget(self.spin_box)

        right_button = QPushButton("Next →")
        buttons_layout.addWidget(right_button)

        auto_right_button = QPushButton("Auto →")
        buttons_layout.addWidget(auto_right_button)

        buttons_layout.addStretch()
        image_layout.addLayout(buttons_layout)

        bottom_splitter.addWidget(image_widget)

        # Set initial sizes for bottom splitter (1/3 map, 2/3 image)
        bottom_splitter.setSizes([self.width() // 3, 2 * self.width() // 3])

        # Set initial sizes for main splitter (small top, large bottom)
        main_splitter.setSizes([100, self.height() - 100])  # Adjust as needed

        # Set up web channel
        self.channel = QWebChannel()
        self.web_view.page().setWebChannel(self.channel)

        # Load the default file
        default_txt_path = "frames_latlon.txt"
        if not os.path.exists(default_txt_path):
            QMessageBox.warning(self, "Input Error", "Default TXT file not found.")
            return

        try:
            grouped_points, points_with_files = read_grouped_points(default_txt_path)

            # Create a folium map centered on the mean coordinates with an initial zoom level
            all_points = [point for points in grouped_points.values() for point in points]
            if not all_points:
                raise ValueError("No points available for map centering")

            latitudes = [lat for lat, lon in all_points]
            longitudes = [lon for lat, lon in all_points]
            center_lat = statistics.mean(latitudes)
            center_lon = statistics.mean(longitudes)

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)

            # Add satellite basemap (Esri World Imagery)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
                name='Satellite Imagery',
                overlay=False,
                control=True
            ).add_to(m)

            # Add roads overlay (Esri World Transportation)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri',
                name='Roads',
                overlay=True,
                control=True
            ).add_to(m)

            # Add places and boundaries overlay (Esri World Boundaries and Places)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri',
                name='Places & Boundaries',
                overlay=True,
                control=True
            ).add_to(m)

            # Add polylines for each section using only red color
            for section_name, points in grouped_points.items():
                if len(points) > 1:  # Only draw line if more than one point
                    folium.PolyLine(
                        locations=[(lat, lon) for lat, lon in points],
                        color='red',
                        weight=5,
                        opacity=0.8,
                        popup=f"Section: {section_name}"
                    ).add_to(m)

            # Add layer control to toggle layers
            folium.LayerControl().add_to(m)

            map_var = m.get_name()

            # Register communicator after loading points
            self.communicator = Communicator(self.coords_label, self.image_label, points_with_files, self.web_view, map_var, self.spin_box)
            self.channel.registerObject('communicator', self.communicator)

            # Connect buttons
            auto_left_button.clicked.connect(self.communicator.toggle_auto_prev)
            left_button.clicked.connect(self.communicator.show_prev)
            right_button.clicked.connect(self.communicator.show_next)
            auto_right_button.clicked.connect(self.communicator.toggle_auto_next)
            self.spin_box.valueChanged.connect(self.communicator.update_interval)

            # Add qwebchannel script to header
            m.get_root().header.add_child(folium.Element('<script type="text/javascript" src="qrc:///qtwebchannel/qwebchannel.js"></script>'))

            # Add custom JS for click event
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

            # Generate HTML content from the map
            out = BytesIO()
            m.save(out, close_file=False)
            html = out.getvalue().decode('utf-8')

            # Display the HTML in the web view
            self.web_view.setHtml(html)

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())