"""
Generate trajectory CSV files (x, y, orientation_deg) for
navigate_csv_trajectory.py.

Available patterns:
    straight   line from left to right through the middle
    s_curve    smooth S (smoothstep) drifting down then up
    circle     closed loop
    figure8    lemniscate (two loops)
    square     closed perimeter, 90 deg turns at corners
    spiral     outward/inward Archimedean spiral
    zigzag     triangle wave across the field

Coordinates are in arbitrary CSV units (default field 100 x 100).
Orientation is the direction of travel: degrees(atan2(dy, dx)) with
the y-down image convention, i.e. 0 = +x (right), 90 = down,
180/-180 = left, -90 = up. The arena mapping in the follower scales
these units onto your clicked arena, so the absolute size doesn't
matter — only the shape.

Usage:
    python generate_trajectory.py --pattern s_curve --output s_curve.csv
    python generate_trajectory.py --pattern circle --n 60
    python generate_trajectory.py --list
"""

import argparse
import csv
import math

PATTERNS = ["straight", "s_curve", "circle", "figure8", "square", "spiral", "zigzag"]


def smoothstep(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def gen_straight(n, w, h):
    return [(5 + 90 * i / (n - 1), h / 2) for i in range(n)]


def gen_s_curve(n, w, h, amp):
    cx, cy = w / 2, h / 2
    return [(5 + 90 * i / (n - 1),
             cy + amp * (2 * smoothstep(i / (n - 1)) - 1)) for i in range(n)]


def gen_circle(n, w, h):
    cx, cy, r = w / 2, h / 2, 40
    return [(cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def gen_figure8(n, w, h):
    cx, cy = w / 2, h / 2
    return [(cx + 40 * math.sin(2 * math.pi * i / n),
             cy + 20 * math.sin(4 * math.pi * i / n)) for i in range(n)]


def gen_square(n, w, h):
    # Walk the closed perimeter; extra samples land on the straight
    # sections so the heading between corners stays exact.
    corners = [(10, 10), (90, 10), (90, 90), (10, 90)]
    per_side = max(2, n // 4)
    pts = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        for j in range(per_side):
            t = j / per_side
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return pts


def gen_spiral(n, w, h, turns):
    cx, cy = w / 2, h / 2
    pts = []
    for i in range(n):
        t = i / (n - 1)
        r = 5 + 38 * t
        th = turns * 2 * math.pi * t
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts


def gen_zigzag(n, w, h, peaks):
    pts = []
    for i in range(n):
        t = i / (n - 1)          # 0..1 left to right
        y = h / 2 + 35 * (2 / math.pi) * math.asin(math.sin(peaks * math.pi * t))
        pts.append((5 + 90 * t, y))
    return pts


def heading_deg(xs, ys, i):
    """Direction of travel at point i via central differences."""
    n = len(xs)
    if i == 0:
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    elif i == n - 1:
        dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
    else:
        dx, dy = xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]
    return math.degrees(math.atan2(dy, dx))


def write_csv(path, pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "orientation"])
        for i in range(len(pts)):
            writer.writerow([f"{xs[i]:.2f}", f"{ys[i]:.2f}", f"{heading_deg(xs, ys, i):.1f}"])
    print(f"Wrote {len(pts)} points to {path}")


def main():
    ap = argparse.ArgumentParser(description="Generate trajectory CSVs for the bot follower.")
    ap.add_argument("--pattern", choices=PATTERNS, default="s_curve")
    ap.add_argument("--output", default=None, help="output CSV file (default: <pattern>.csv)")
    ap.add_argument("--n", type=int, default=40, help="number of points (default 40)")
    ap.add_argument("--width", type=float, default=100.0, help="field width in CSV units")
    ap.add_argument("--height", type=float, default=100.0, help="field height in CSV units")
    ap.add_argument("--amp", type=float, default=30.0, help="s_curve amplitude")
    ap.add_argument("--turns", type=float, default=3.0, help="spiral turns")
    ap.add_argument("--peaks", type=int, default=4, help="zigzag peaks")
    ap.add_argument("--list", action="store_true", help="list patterns and exit")
    args = ap.parse_args()

    if args.list:
        print("Available patterns:", ", ".join(PATTERNS))
        return

    generators = {
        "straight": lambda: gen_straight(args.n, args.width, args.height),
        "s_curve": lambda: gen_s_curve(args.n, args.width, args.height, args.amp),
        "circle": lambda: gen_circle(args.n, args.width, args.height),
        "figure8": lambda: gen_figure8(args.n, args.width, args.height),
        "square": lambda: gen_square(args.n, args.width, args.height),
        "spiral": lambda: gen_spiral(args.n, args.width, args.height, args.turns),
        "zigzag": lambda: gen_zigzag(args.n, args.width, args.height, args.peaks),
    }
    pts = generators[args.pattern]()
    write_csv(args.output or f"{args.pattern}.csv", pts)


if __name__ == "__main__":
    main()
