"""
Navigate the bot to a target (x, y, orientation) using ArUco tracking,
with a manually mapped arena.

Flow:
  1. Camera feed opens. Click points with the mouse to trace the arena
     boundary (3 or more points). Left-click adds a point.
        ENTER or 'c' = finish arena
        'u'          = undo last point
        'r'          = clear all points
        'q'          = quit
  2. After the arena is confirmed, the terminal asks for the target
     X, Y and orientation.
  3. The bot navigates to the target while the arena boundary stays
     drawn on the video feed. Press 'q' in the video window to cancel.

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
converging), flip ROTATE_SIGN from 1 to -1 below — that's the only
thing likely to need calibrating for your camera/marker orientation.
"""

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
cam.set(3, 500)
cam.set(4, 505)

if not cam.isOpened():
    raise RuntimeError("Could not open camera. Try a different index (0, 1, 2...).")

# ---------- Navigation tuning ----------
POSITION_TOLERANCE_PX = 15
ANGLE_TOLERANCE_DEG = 8
ROTATE_SIGN = 1  # flip to -1 if the bot turns the wrong direction


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


def map_arena():
    """Show the live feed and let the user click the arena boundary."""
    arena = []
    print("Arena mapping: left-click boundary points (>= 3), "
          "ENTER/c = done, u = undo, r = reset, q = quit.")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            arena.append((x, y))

    cv2.namedWindow("Navigation")
    cv2.setMouseCallback("Navigation", on_mouse)

    while True:
        success, img = cam.read()
        if not success:
            raise RuntimeError("Failed to grab frame during arena mapping")

        pts = np.array(arena, np.int32)
        if len(arena) >= 3:
            # translucent fill so you can see the mapped region
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        elif len(arena) >= 2:
            cv2.polylines(img, [pts], False, (0, 255, 0), 2)

        for p in arena:
            cv2.circle(img, p, 5, (0, 0, 255), -1)
            cv2.putText(img, f"({p[0]},{p[1]})", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(img, f"points: {len(arena)}  ENTER=done  u=undo  r=reset  q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Navigation", img)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, ord('c')) and len(arena) >= 3:
            break
        elif key == ord('u') and arena:
            arena.pop()
        elif key == ord('r'):
            arena.clear()
        elif key == ord('q'):
            raise SystemExit("Cancelled during arena mapping.")

    cv2.setMouseCallback("Navigation", lambda *args: None)  # ignore stray clicks later
    print(f"Arena mapped with {len(arena)} points. Corner coordinates:")
    for i, (x, y) in enumerate(arena, start=1):
        print(f"  corner {i}: ({x}, {y})")
    return arena


def get_target():
    x = float(input("Target X (pixels): "))
    y = float(input("Target Y (pixels): "))
    orientation = float(input("Target orientation (degrees): "))
    return x, y, orientation


def main():
    arena = map_arena()
    arena_pts = np.array(arena, np.int32)

    target_x, target_y, target_orientation = get_target()

    # Warn (but don't block) if the target lies outside the mapped arena
    if cv2.pointPolygonTest(arena_pts, (target_x, target_y), False) < 0:
        print("WARNING: target is outside the mapped arena — proceeding anyway.")

    print(f"Navigating to ({target_x:.0f}, {target_y:.0f}) at {target_orientation:.0f} deg. "
          f"Press 'q' in the video window to cancel.")

    state = "ROTATE_TO_HEADING"

    try:
        while True:
            success, img = cam.read()
            if not success:
                print("Failed to grab frame")
                break

            corners, ids, _ = detector.detectMarkers(img)
            cmd = 'X'

            cv2.polylines(img, [arena_pts], True, (0, 255, 0), 2)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners)
                marker_corners = corners[0][0]  # tracks the first marker seen
                current_angle = calculate_angle(marker_corners)
                center = marker_corners.mean(axis=0)
                cx, cy = center[0], center[1]

                dx = target_x - cx
                dy = target_y - cy
                distance = math.hypot(dx, dy)
                desired_heading = math.degrees(math.atan2(dy, dx))
                heading_error = ROTATE_SIGN * normalize_angle(desired_heading - current_angle)
                orientation_error = ROTATE_SIGN * normalize_angle(target_orientation - current_angle)

                cv2.circle(img, (int(target_x), int(target_y)), 6, (255, 0, 0), -1)
                cv2.putText(img, f"dist:{distance:.0f} herr:{heading_error:.0f} state:{state}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                if state == "ROTATE_TO_HEADING":
                    if distance <= POSITION_TOLERANCE_PX:
                        state = "ROTATE_TO_ORIENTATION"
                    elif abs(heading_error) > ANGLE_TOLERANCE_DEG:
                        cmd = 'W' if heading_error > 0 else 'S'
                    else:
                        state = "MOVE"

                if state == "MOVE":
                    if distance <= POSITION_TOLERANCE_PX:
                        state = "ROTATE_TO_ORIENTATION"
                    elif abs(heading_error) > ANGLE_TOLERANCE_DEG * 2:
                        state = "ROTATE_TO_HEADING"
                    else:
                        cmd = 'D'

                if state == "ROTATE_TO_ORIENTATION":
                    if abs(orientation_error) <= ANGLE_TOLERANCE_DEG:
                        state = "DONE"
                    else:
                        cmd = 'W' if orientation_error > 0 else 'S'

                if state == "DONE":
                    send('X')
                    print("Target reached.")
                    cv2.imshow("Navigation", img)
                    cv2.waitKey(500)
                    break
            else:
                cmd = 'X'  # marker not visible, stop for safety

            send(cmd)

            cv2.imshow("Navigation", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Cancelled.")
                break

            time.sleep(SEND_INTERVAL_S)

    finally:
        send('X')
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
