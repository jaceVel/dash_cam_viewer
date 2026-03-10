import os
import cv2
import math
from datetime import timedelta
from pathlib import Path
import subprocess
import re

# Configuration
ROOT_DIR = r"D:\vcloud\DPIRD_MacquarieArc"
OUTPUT_FOLDER = "combined_frames"
TXT_PATH = "frames_latlon.txt"
FPS_FALLBACK = 30.0
EXIFTOOL_CMD = r"C:\Users\jstep\OneDrive\Desktop\python 2026\garmin video\03_make-frames-cataloge\exiftool(-k).exe"  # Adjust to full path if not in system PATH, e.g., r"C:\Path\to\exiftool.exe"

# Create output folder
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# Write header to TXT file
with open(TXT_PATH, "w") as f:
    f.write("Filename\tLatitude\tLongitude\n")


def extract_gps_timeline(video_path: str) -> list[tuple[float, str, str]]:
    """
    Uses exiftool to extract time-stamped GPS data from the video.
    Returns a list of tuples: (time_in_seconds, latitude, longitude)
    """
    cmd = [EXIFTOOL_CMD, "-m", "-ee", "-n", "-p", "${SampleTime} ${GPSLatitude} ${GPSLongitude}", video_path]
    try:
        result = subprocess.check_output(cmd, text=True, input='\n')
        lines = result.splitlines()
        gps_data = []
        for line in lines:
            line = line.strip()
            if line:
                parts = re.split(r'\s+', line)
                if len(parts) >= 3:
                    # Parse SampleTime assuming it's a float in seconds
                    time_str = parts[0]
                    secs = float(time_str)
                    lat = parts[-2]
                    lon = parts[-1]
                    gps_data.append((secs, lat, lon))
        return gps_data
    except subprocess.CalledProcessError as e:
        print(f"Exiftool error for {video_path}: {e}")
        return []
    except ValueError as ve:
        print(f"Value error parsing SampleTime for {video_path}: {ve}")
        return []
    except Exception as e:
        print(f"Unexpected error extracting GPS from {video_path}: {e}")
        return []


def get_lat_lon_at_sec(gps_data: list[tuple[float, str, str]], target_sec: float) -> tuple[str, str]:
    """
    Interpolates or extrapolates GPS coordinates for the target second.
    Returns (latitude, longitude) or ("", "") if no data available.
    """
    if not gps_data:
        return "", ""
    # Sort by time
    gps_data.sort(key=lambda x: x[0])

    if target_sec <= gps_data[0][0]:
        # Before first: use first
        return gps_data[0][1], gps_data[0][2]

    if target_sec >= gps_data[-1][0]:
        # After last: extrapolate using last two points
        if len(gps_data) < 2:
            return gps_data[-1][1], gps_data[-1][2]
        last2 = gps_data[-2]
        last1 = gps_data[-1]
        dt = last1[0] - last2[0]
        if dt == 0:
            return last1[1], last1[2]
        vlat = (float(last1[1]) - float(last2[1])) / dt
        vlon = (float(last1[2]) - float(last2[2])) / dt
        extrat = target_sec - last1[0]
        ext_lat = float(last1[1]) + vlat * extrat
        ext_lon = float(last1[2]) + vlon * extrat
        return str(ext_lat), str(ext_lon)

    # Between points: linear interpolation
    for i in range(len(gps_data) - 1):
        if gps_data[i][0] <= target_sec <= gps_data[i + 1][0]:
            t1, lat1, lon1 = gps_data[i]
            t2, lat2, lon2 = gps_data[i + 1]
            dt = t2 - t1
            if dt == 0:
                return lat1, lon1
            frac = (target_sec - t1) / dt
            inter_lat = float(lat1) + frac * (float(lat2) - float(lat1))
            inter_lon = float(lon1) + frac * (float(lon2) - float(lon1))
            return str(inter_lat), str(inter_lon)

    # Should not reach here
    return "", ""


def extract_frames_every_second(
        video_path,
        output_dir,
        prefix="",
        txt_path=None,
        fps_fallback=30.0
):
    # Extract GPS data once for the video
    gps_data = extract_gps_timeline(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not math.isfinite(fps):
        print(f"FPS not readable for {video_path} → using fallback {fps_fallback}")
        fps = fps_fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"\nProcessing {video_path}:")
    print(f"  FPS:          {fps:.3f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration:     {duration_sec:.1f} seconds ({timedelta(seconds=int(duration_sec))})")

    saved_count = 0
    frame_idx = 0
    last_saved_sec = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / fps
        current_int_sec = math.floor(current_sec)

        if current_int_sec > last_saved_sec:
            secs_total = current_int_sec
            time_str = str(timedelta(seconds=secs_total)).replace(':', '_')

            filename = f"{prefix}_frame_{frame_idx:06d}_sec_{time_str}.jpg" if prefix else f"frame_{frame_idx:06d}_sec_{time_str}.jpg"
            full_path = os.path.join(output_dir, filename)

            cv2.imwrite(full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved_count += 1
            last_saved_sec = current_int_sec

            print(f"    Saved: {filename}  ({secs_total} s)")

            if txt_path:
                lat, lon = get_lat_lon_at_sec(gps_data, current_sec)  # Use current_sec for precision
                with open(txt_path, "a") as f:
                    f.write(f"{filename}\t{lat}\t{lon}\n")
                print(f"      Lat: {lat}  Lon: {lon}")

        frame_idx += 1

    cap.release()
    print(f"  Finished {video_path}. {saved_count} frames saved to: {os.path.abspath(output_dir)}")
    return saved_count


# Find all MP4 files recursively
mp4_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith('.mp4'):
            mp4_files.append(os.path.join(root, file))

print(f"Found {len(mp4_files)} MP4 files in {ROOT_DIR} and subdirectories.")

total_saved = 0

for mp4 in mp4_files:
    # Create unique prefix based on MP4 name and relative path
    rel_path = os.path.relpath(os.path.dirname(mp4), ROOT_DIR).replace('\\', '_').replace('/', '_')
    base_name = os.path.splitext(os.path.basename(mp4))[0]
    prefix = f"{rel_path}_{base_name}" if rel_path != '.' else base_name

    saved = extract_frames_every_second(
        mp4,
        OUTPUT_FOLDER,
        prefix,
        TXT_PATH,
        FPS_FALLBACK
    )
    total_saved += saved

print(f"\nAll processing complete. Total frames saved: {total_saved}")