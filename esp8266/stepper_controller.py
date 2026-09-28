"""
CSV trajectory follower with manual arena mapping.

Flow:
  1. You provide a CSV file with columns: x, y, orientation(degrees)
     (a header row is auto-detected and skipped).
  2. The camera feed opens. Click the 4 arena corners with the mouse
     (corner coordinates are shown next to each point). The CSV
     trajectory is then perspective-mapped so its bounding box fits
     exactly into the clicked arena quadrilateral.
  3. Press 's' to start. The bot visits every trajectory point in
     order, driving to each position and rotating to that point's
     orientation before moving on.

Controls (video window):
    During arena mapping:
        Left click   = add a corner (exactly 4 needed, in order)
        u            = undo last corner
        r            = reset corners
        ENTER or c   = confirm arena
        q            = quit
    After mapping:
        s            = start navigation
        q            = quit / abort (stops the bot)

Command meaning (as wired/observed on your bot):
    D = forward   (translate)
    A = backward  (translate)
    W = rotate right
    S = rotate left
    X = stop

Setup:
    pip install opencv-contrib-python numpy
    Set ESP_IP below to your board's IP address.

If the bot turns the wrong way (overshoots/oscillates instead of
converging), flip ROTATE_SIGN from 1 to -1 below.
"""

import csv
import math
import time
import socket

import cv2
import numpy as np

# ---------- Bot connection ----------
ESP_IP = "192.168.50.175"   # <-- set to your board's IP address
ESP_PORT = 4210
SEND_INTERVAL_S = 0.05      # matches the board's 300ms watchdog

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(cmd: str):
    sock.sendto(cmd.encode(), (ESP_IP, ESP_PORT))


# ---------- ArUco setup ----------
BOT_ID = 1  # ArUco ID of the marker on this bot

# DICT_4X4_50 holds the same patterns as the first 50 of DICT_4X4_250,
# so tag 1 still works. Fewer valid codes means fewer false detections.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.polygonalApproxAccuracyRate = 0.05
parameters.minMarkerPerimeterRate = 0.015
parameters.adaptiveThreshConstant = 10
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minCornerDistanceRate = 0.05
parameters.errorCorrectionRate = 1.0
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# ---------- Camera setup (Rapoo C280) ----------
CAMERA_INDEX = 1        # same index as calibrate_and_track.py
FRAME_W, FRAME_H = 2560, 1440
DISPLAY_W, DISPLAY_H = 1920, 1080   # size of the window on screen
FOCUS_STEP = 5
focus_value = 0         # press [ and ] in the window to tune this

cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(cv2.CAP_PROP_FOCUS, focus_value)

if not cam.isOpened():
    raise RuntimeError("Could not open camera. Try a different CAMERA_INDEX (0, 1, 2...).")

# Use the resolution the camera actually gave us, so clicks map correctly
ACTUAL_W = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
ACTUAL_H = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera running at {ACTUAL_W}x{ACTUAL_H}")


def detect_markers(img):
    """Grayscale + CLAHE, then ArUco detection. Returns (corners, ids)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def find_bot(corners, ids):
    """Return the corners of BOT_ID only, or None if it isn't visible."""
    if ids is None:
        return None
    for i, marker_id in enumerate(ids.flatten()):
        if int(marker_id) == BOT_ID:
            return corners[i][0]
    return None


def show(img):
    """Display a downscaled copy. INTER_AREA keeps edges sharp when shrinking."""
    disp = cv2.resize(img, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, disp)


def handle_focus_key(key):
    """'[' and ']' step the lens focus so you can find the sharpest setting."""
    global focus_value
    if key == ord(']'):
        focus_value += FOCUS_STEP
    elif key == ord('['):
        focus_value = max(0, focus_value - FOCUS_STEP)
    elif key == ord('p'):
        cam.set(cv2.CAP_PROP_SETTINGS, 1)  # opens the Windows camera settings panel
        return    
    else:
        return
    cam.set(cv2.CAP_PROP_FOCUS, focus_value)
    print(f"Focus requested: {focus_value} | camera reports: "
          f"{cam.get(cv2.CAP_PROP_FOCUS):.0f}")


# ---------- Navigation tuning ----------
POSITION_TOLERANCE_PX = 55  # was 15 at ~700 px wide, scaled for 2560 px
ANGLE_TOLERANCE_DEG = 8
ROTATE_SIGN = 1  # flip to -1 if the bot turns the wrong direction

WINDOW_NAME = "Navigation"
DEFAULT_CSV = "trajectory.csv"   # used if you just press ENTER at the prompt
NUM_ARENA_CORNERS = 4            # needed for a perspective mapping


def calculate_angle(corners):
    top_left, top_right, bottom_right, bottom_left = corners
    vector = top_right - top_left
    return math.degrees(math.atan2(vector[1], vector[0]))


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


# ---------- Trajectory / arena geometry ----------

def load_trajectory_csv(path):
    """Read rows of (x, y, orientation_deg); skip header/non-numeric rows."""
    import os
    if os.path.isdir(path):
        csvs = sorted(f for f in os.listdir(path) if f.lower().endswith(".csv"))
        if not csvs:
            raise SystemExit(f"'{path}' is a folder with no CSV files in it.")
        print(f"'{path}' is a folder. CSV files inside:")
        for i, name in enumerate(csvs, start=1):
            print(f"  {i}: {name}")
        choice = input(f"Pick 1-{len(csvs)} (or type a file name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(csvs):
            path = os.path.join(path, csvs[int(choice) - 1])
        else:
            path = os.path.join(path, choice)
    elif not path.lower().endswith(".csv") and os.path.isfile(path + ".csv"):
        path += ".csv"

    points = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                x, y, orientation = float(row[0]), float(row[1]), float(row[2])
            except ValueError:
                continue  # header or comment row
            points.append((x, y, orientation))
    if not points:
        raise SystemExit(f"No valid x,y,orientation rows found in {path!r} — "
                         "expected columns like: 50.0,20.0,90.0")
    return points


def map_trajectory_to_arena(points, arena_corners):
    """
    Perspective-map the CSV coordinate space onto the clicked arena quad.

    The CSV bounding box corners (min_x,min_y) -> (max_x,min_y) ->
    (max_x,max_y) -> (min_x,max_y) are mapped, in that order, onto the
    clicked arena corners. Positions go through the homography, and each
    orientation is rotated by mapping a unit step in the CSV direction.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    src = np.float32([[min_x, min_y], [max_x, min_y],
                      [max_x, max_y], [min_x, max_y]])
    dst = np.float32(arena_corners)
    H = cv2.getPerspectiveTransform(src, dst)

    mapped = []
    for x, y, theta in points:
        pt = np.float32([[[x, y]]])
        px, py = cv2.perspectiveTransform(pt, H)[0, 0]

        # Map a 1-unit step along the CSV heading to learn how the
        # homography rotates that heading into camera-pixel space.
        th = math.radians(theta)
        step = np.float32([[[x + math.cos(th), y + math.sin(th)]]])
        sx, sy = cv2.perspectiveTransform(step, H)[0, 0]
        new_theta = math.degrees(math.atan2(sy - py, sx - px))

        mapped.append((float(px), float(py), new_theta))
    return mapped


def map_arena():
    """Let the user click the arena corners; returns [(x, y) x 4]."""
    arena = []
    print(f"Arena mapping: left-click {NUM_ARENA_CORNERS} corners in order, "
          "ENTER/c = done, u = undo, r = reset, q = quit.")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(arena) < NUM_ARENA_CORNERS:
            arena.append((int(x * ACTUAL_W / DISPLAY_W), int(y * ACTUAL_H / DISPLAY_H)))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        success, img = cam.read()
        if not success:
            raise RuntimeError("Failed to grab frame during arena mapping")

        if len(arena) >= 3:
            cv2.polylines(img, [np.array(arena, np.int32)], True, (0, 255, 0), 2)
        elif len(arena) >= 2:
            cv2.polylines(img, [np.array(arena, np.int32)], False, (0, 255, 0), 2)

        for p in arena:
            cv2.circle(img, p, 5, (0, 0, 255), -1)
            cv2.putText(img, f"({p[0]},{p[1]})", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(img, f"corners: {len(arena)}/{NUM_ARENA_CORNERS}  "
                         "ENTER=done  u=undo  r=reset  q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        show(img)

        key = cv2.waitKey(1) & 0xFF
        handle_focus_key(key)
        if key in (13, ord('c')) and len(arena) == NUM_ARENA_CORNERS:
            break
        elif key == ord('u') and arena:
            arena.pop()
        elif key == ord('r'):
            arena.clear()
        elif key == ord('q'):
            raise SystemExit("Cancelled during arena mapping.")

    cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)  # ignore stray clicks later
    print(f"Arena corners (camera pixels):")
    for i, (x, y) in enumerate(arena, start=1):
        print(f"  corner {i}: ({x}, {y})")
    return arena


# ---------- Navigation ----------

class NavApp:
    def __init__(self, trajectory):
        self.trajectory = trajectory  # list of (x_px, y_px, orientation_deg)
        self.running = False
        self.current_index = 0
        self.state = "ROTATE_TO_HEADING"

    def start(self):
        self.running = True
        self.current_index = 0
        self.state = "ROTATE_TO_HEADING"

    def stop(self):
        self.running = False
        send('X')

    def draw_overlay(self, img):
        # trajectory polyline
        for i in range(len(self.trajectory) - 1):
            p1 = (int(self.trajectory[i][0]), int(self.trajectory[i][1]))
            p2 = (int(self.trajectory[i + 1][0]), int(self.trajectory[i + 1][1]))
            cv2.line(img, p1, p2, (255, 0, 0), 1)

        for i, (tx, ty, _) in enumerate(self.trajectory):
            color = (0, 165, 255) if (self.running and i == self.current_index) else (255, 0, 0)
            cv2.circle(img, (int(tx), int(ty)), 6, color, -1)
            cv2.putText(img, str(i + 1), (int(tx) + 8, int(ty) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if not self.running:
            msg = f"READY  |  {len(self.trajectory)} point(s)  |  s: start  q: quit"
        else:
            msg = (f"RUNNING  |  point {self.current_index + 1}/{len(self.trajectory)}  "
                   f"|  state:{self.state}  q: abort")
        cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def step(self, cx, cy, current_angle):
        """Decide the command for the current trajectory point. Returns cmd."""
        target_x, target_y, target_orientation = self.trajectory[self.current_index]
        dx = target_x - cx
        dy = target_y - cy
        distance = math.hypot(dx, dy)
        desired_heading = math.degrees(math.atan2(dy, dx))
        heading_error = ROTATE_SIGN * normalize_angle(desired_heading - current_angle)
        orientation_error = ROTATE_SIGN * normalize_angle(target_orientation - current_angle)
        cmd = 'X'

        if self.state == "ROTATE_TO_HEADING":
            if distance <= POSITION_TOLERANCE_PX:
                self.state = "ROTATE_TO_ORIENTATION"
            elif abs(heading_error) > ANGLE_TOLERANCE_DEG:
                cmd = 'W' if heading_error > 0 else 'S'
            else:
                self.state = "MOVE"

        if self.state == "MOVE":
            if distance <= POSITION_TOLERANCE_PX:
                self.state = "ROTATE_TO_ORIENTATION"
            elif abs(heading_error) > ANGLE_TOLERANCE_DEG * 2:
                self.state = "ROTATE_TO_HEADING"
            else:
                cmd = 'D'

        if self.state == "ROTATE_TO_ORIENTATION":
            if abs(orientation_error) <= ANGLE_TOLERANCE_DEG:
                print(f"Point {self.current_index + 1} reached at "
                      f"({target_x:.0f}, {target_y:.0f}), orient {target_orientation:.0f} deg.")
                self.current_index += 1
                self.state = "ROTATE_TO_HEADING"
            else:
                cmd = 'W' if orientation_error > 0 else 'S'

        if self.current_index >= len(self.trajectory):
            print("All trajectory points reached.")
            self.stop()

        return cmd


def main():
    csv_path = input(f"Trajectory CSV path [{DEFAULT_CSV}]: ").strip() or DEFAULT_CSV
    try:
        trajectory = load_trajectory_csv(csv_path)
    except FileNotFoundError:
        raise SystemExit(f"File not found: {csv_path!r} — check the path and try again.")
    except PermissionError:
        raise SystemExit(f"Permission denied: {csv_path!r} — that's a folder, not a CSV file. "
                         "Enter the full path to a .csv file, e.g. s_curve.csv")
    print(f"Loaded {len(trajectory)} trajectory point(s).")

    arena_corners = map_arena()
    trajectory = map_trajectory_to_arena(trajectory, arena_corners)
    print("Trajectory mapped to camera pixels (x, y, orientation_deg):")
    for i, (x, y, o) in enumerate(trajectory, start=1):
        print(f"  point {i}: ({x:.1f}, {y:.1f}), {o:.1f} deg")

    app = NavApp(trajectory)
    print("Press 's' to start, 'q' to quit.")

    try:
        while True:
            success, img = cam.read()
            if not success:
                print("Failed to grab frame")
                break

            corners, ids = detect_markers(img)
            bot_corners = find_bot(corners, ids)
            cmd = 'X'

            if bot_corners is not None:
                # Draw only the bot's marker, never false detections from the floor
                cv2.aruco.drawDetectedMarkers(img, [bot_corners[np.newaxis]],
                                              np.array([[BOT_ID]], dtype=np.int32))

            if app.running and bot_corners is not None:
                current_angle = calculate_angle(bot_corners)
                center = bot_corners.mean(axis=0)
                cmd = app.step(center[0], center[1], current_angle)

            elif app.running:
                cmd = 'X'  # marker not visible, stop for safety

            if app.running:
                send(cmd)

            app.draw_overlay(img)
            show(img)

            key = cv2.waitKey(1) & 0xFF
            handle_focus_key(key)
            if key == ord('q'):
                app.stop()
                print("Stopped.")
                break
            elif key == ord('s') and not app.running:
                app.start()
                print(f"Starting navigation through {len(app.trajectory)} point(s).")

            time.sleep(SEND_INTERVAL_S)

    finally:
        send('X')
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()