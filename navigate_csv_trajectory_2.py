"""
CSV trajectory follower with manual arena mapping.

Flow:
  1. You provide a CSV file with columns: x, y, orientation(degrees)
     (a header row is auto-detected and skipped).
  2. The camera feed opens. Click the 4 arena corners with the mouse
     (corner coordinates are shown next to each point). The CSV
     trajectory is then perspective-mapped so its bounding box fits
     exactly into the clicked arena quadrilateral.
  3. The mapped points are treated as control points of a smooth
     Catmull-Rom spline. Press 's' to start: the bot tracks the curve
     continuously with a pure-pursuit controller (steering toward a
     lookahead point on the path) — it does NOT stop at each point.
     It only stops at the final destination, where it rotates to the
     last point's orientation.

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
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cam = cv2.VideoCapture(0)  # change index if the wrong camera opens
cam.set(3, 700)
cam.set(4, 505)

if not cam.isOpened():
    raise RuntimeError("Could not open camera. Try a different index (0, 1, 2...).")

# ---------- Navigation tuning ----------
ANGLE_TOLERANCE_DEG = 8
ROTATE_SIGN = 1  # flip to -1 if the bot turns the wrong direction

# Pure-pursuit tuning
LOOKAHEAD_PX = 45       # how far ahead on the path to steer toward
STEER_ROTATE_DEG = 35   # |heading error| above this -> stop translating, rotate
STEER_DRIVE_DEG = 18    # |heading error| below this -> drive (hysteresis band between)
END_TOLERANCE_PX = 22   # distance to final point where the final orientation lock begins
SPLINE_SAMPLES_PER_SEG = 25  # spline samples between consecutive CSV points

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
            arena.append((x, y))

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
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1) & 0xFF
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


# ---------- Smooth path ----------

def catmull_rom_spline(points, samples_per_seg=SPLINE_SAMPLES_PER_SEG):
    """Smooth curve passing through all (x, y, ...) points.

    Uses a Catmull-Rom spline with duplicated endpoints. Returns a dense
    list of (x, y) samples in traversal order.
    """
    if len(points) < 3:
        return [(x, y) for x, y, _ in points]

    ext = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(len(ext) - 3):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append((points[-1][0], points[-1][1]))
    return out


def heading_deg(xs, ys, i):
    """Direction of travel at sample i via central differences."""
    n = len(xs)
    if i == 0:
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    elif i == n - 1:
        dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
    else:
        dx, dy = xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]
    return math.degrees(math.atan2(dy, dx))


def build_dense_path(mapped_points):
    """Spline the mapped waypoints and attach tangent headings."""
    smooth = catmull_rom_spline(mapped_points)
    xs = [p[0] for p in smooth]
    ys = [p[1] for p in smooth]
    return [(x, y, heading_deg(xs, ys, i)) for i, (x, y) in enumerate(smooth)]


# ---------- Navigation (pure pursuit) ----------

class NavApp:
    """
    Tracks a dense path continuously using pure pursuit: steer toward a
    point LOOKAHEAD_PX ahead on the path instead of stopping at each
    sample. Only the final destination triggers an orientation lock.
    """

    def __init__(self, path):
        self.path = path  # dense [(x_px, y_px, tangent_heading_deg)]
        self.running = False
        self.progress = 0        # index of nearest path sample (monotonic)
        self.turning = False     # hysteresis: currently rotating vs driving
        self.lookahead_pt = None # last lookahead target (for display)

    def start(self):
        self.running = True
        self.progress = 0
        self.turning = False

    def stop(self):
        self.running = False
        send('X')

    def _lookahead(self, cx, cy):
        """Advance progress to the nearest sample (never backwards), then
        walk arc-length forward to the lookahead target."""
        n = len(self.path)
        best_i = self.progress
        best_d = math.hypot(self.path[self.progress][0] - cx,
                            self.path[self.progress][1] - cy)
        for i in range(self.progress + 1, min(n, self.progress + 40)):
            d = math.hypot(self.path[i][0] - cx, self.path[i][1] - cy)
            if d < best_d:
                best_i, best_d = i, d
        self.progress = best_i

        acc = 0.0
        i = best_i
        while i < n - 1 and acc < LOOKAHEAD_PX:
            acc += math.hypot(self.path[i + 1][0] - self.path[i][0],
                              self.path[i + 1][1] - self.path[i][1])
            i += 1
        return self.path[i]

    def step(self, cx, cy, current_angle):
        end_x, end_y, end_theta = self.path[-1]
        dist_end = math.hypot(end_x - cx, end_y - cy)

        # Endgame: close to the destination -> rotate to final heading, then stop
        if dist_end <= END_TOLERANCE_PX:
            err = ROTATE_SIGN * normalize_angle(end_theta - current_angle)
            if abs(err) <= ANGLE_TOLERANCE_DEG:
                print("Destination reached.")
                self.stop()
                return 'X'
            return 'W' if err > 0 else 'S'

        # Cruise: pure pursuit toward the lookahead point
        lx, ly, _ = self._lookahead(cx, cy)
        self.lookahead_pt = (lx, ly)
        desired = math.degrees(math.atan2(ly - cy, lx - cx))
        err = ROTATE_SIGN * normalize_angle(desired - current_angle)

        if abs(err) > STEER_ROTATE_DEG:
            self.turning = True
        elif abs(err) < STEER_DRIVE_DEG:
            self.turning = False
        return ('W' if err > 0 else 'S') if self.turning else 'D'

    def draw_overlay(self, img):
        # dense path
        for i in range(len(self.path) - 1):
            p1 = (int(self.path[i][0]), int(self.path[i][1]))
            p2 = (int(self.path[i + 1][0]), int(self.path[i + 1][1]))
            cv2.line(img, p1, p2, (200, 200, 255), 1)

        # lookahead target
        if self.running and self.lookahead_pt is not None:
            cv2.circle(img, (int(self.lookahead_pt[0]), int(self.lookahead_pt[1])),
                       7, (0, 165, 255), -1)

        # final destination
        ex, ey, _ = self.path[-1]
        cv2.circle(img, (int(ex), int(ey)), 8, (255, 0, 0), -1)

        pct = 100.0 * self.progress / max(1, len(self.path) - 1)
        if not self.running:
            msg = f"READY  |  path of {len(self.path)} samples  |  s: start  q: quit"
        else:
            msg = f"RUNNING  |  path progress {pct:.0f}%  q: abort"
        cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


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
    print(f"Trajectory mapped to camera pixels ({len(trajectory)} control points).")

    path = build_dense_path(trajectory)
    app = NavApp(path)
    print(f"Smooth path built: {len(path)} samples. Press 's' to start, 'q' to quit.")

    try:
        while True:
            success, img = cam.read()
            if not success:
                print("Failed to grab frame")
                break

            corners, ids, _ = detector.detectMarkers(img)
            cmd = 'X'

            if app.running and ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners)
                marker_corners = corners[0][0]  # tracks the first marker seen
                current_angle = calculate_angle(marker_corners)
                center = marker_corners.mean(axis=0)
                cmd = app.step(center[0], center[1], current_angle)

            elif app.running:
                cmd = 'X'  # marker not visible, stop for safety

            if app.running:
                send(cmd)

            app.draw_overlay(img)
            cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                app.stop()
                print("Stopped.")
                break
            elif key == ord('s') and not app.running:
                app.start()
                print("Starting continuous path tracking.")

            time.sleep(SEND_INTERVAL_S)

    finally:
        send('X')
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
