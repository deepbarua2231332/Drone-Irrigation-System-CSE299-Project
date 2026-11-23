import urllib.request
import cv2
import numpy as np

BASE_URL = "http://192.168.4.1"
CAPTURE_URL = f"{BASE_URL}/capture"

# Standard ESP32 CameraWebServer framesize IDs:
# 4 = QVGA (320x240), 5 = CIF (400x296), 6 = VGA (640x480), 7 = SVGA (800x600), 8 = XGA (1024x768)
RESOLUTIONS = [
    (4, "QVGA", "320x240"),
    (5, "CIF",  "400x296"),
    (6, "VGA",  "640x480"),
    (7, "SVGA", "800x600"),
    (8, "XGA",  "1024x768"),
]

# Healthy threshold slider maps 0..9 -> 50%..95% (step 5%)
HEALTH_THRESH_MIN = 50
HEALTH_THRESH_STEP = 5
HEALTH_THRESH_MAX_POS = 9  # 50 + 9*5 = 95%

# Minimum connected component area (in pixels) to keep as a tree/foliage region
MIN_TREE_AREA = 150

VARI_BINS = [
    (-1.00, 0.05,  (35, 65, 135)),
    (0.05,  0.07,  (40, 85, 175)),
    (0.07,  0.11,  (45, 115, 215)),
    (0.11,  0.13,  (55, 145, 235)),
    (0.13,  0.15,  (70, 175, 245)),
    (0.15,  0.17,  (85, 205, 245)),
    (0.17,  0.19,  (95, 225, 235)),
    (0.19,  0.21,  (95, 220, 195)),
    (0.21,  0.23,  (85, 205, 150)),
    (0.23,  0.25,  (75, 185, 105)),
    (0.25,  0.27,  (65, 165, 65)),
    (0.27,  0.31,  (55, 145, 40)),
    (0.31,  0.35,  (65, 135, 25)),
    (0.35,  0.39,  (95, 130, 20)),
    (0.39,  0.45,  (130, 110, 25)),
    (0.45,  1.00,  (150, 75, 35)),
]

CTRL_WIN = "Tree Health & VARI Controller"


def set_control(var: str, val: int, timeout: float = 3.0) -> None:
    """Send a /control command to the ESP32-CAM."""
    url = f"{BASE_URL}/control?var={var}&val={val}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        resp.read()


def apply_camera_settings(framesize: int, brightness: int, contrast: int) -> None:
    """Push resolution, brightness (-2..2), and contrast (-2..2) to ESP32-CAM."""
    set_control("framesize", framesize)
    set_control("brightness", brightness)
    set_control("contrast", contrast)


def fetch_frame(url: str, timeout: float = 12.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        buf = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def detect_tree_and_health(frame_bgr: np.ndarray, healthy_thresh_pct: float) -> tuple[np.ndarray, dict]:
    """Detect tree/foliage pixels and evaluate health against healthy_thresh_pct."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    exg = (2.0 * g) - r - b

    foliage_raw = (
        (h >= 16) & (h <= 95) &
        (s >= 35) &
        (v >= 30) & (v <= 245) &
        (exg > -5.0)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(foliage_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    tree_mask = np.zeros_like(cleaned, dtype=bool)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= MIN_TREE_AREA:
            tree_mask[labels == label_id] = True

    healthy_mask = (
        tree_mask &
        (g > r * 1.15) &
        (g > b * 1.05) &
        (s >= 55) &
        (h >= 38) & (h <= 88)
    )

    yellow_mask = tree_mask & (~healthy_mask) & (h >= 18) & (h < 38) & (s >= 40)
    brown_mask = tree_mask & (~healthy_mask) & (~yellow_mask)

    total_px = int(np.sum(tree_mask))
    healthy_px = int(np.sum(healthy_mask))
    yellow_px = int(np.sum(yellow_mask))
    brown_px = int(np.sum(brown_mask))

    if total_px < MIN_TREE_AREA:
        status = "NO TREE DETECTED"
        color = (0, 200, 255)
        healthy_pct = yellow_pct = brown_pct = 0.0
    else:
        healthy_pct = (healthy_px / total_px) * 100.0
        yellow_pct = (yellow_px / total_px) * 100.0
        brown_pct = (brown_px / total_px) * 100.0

        mod_thresh = max(35.0, healthy_thresh_pct - 25.0)
        if healthy_pct >= healthy_thresh_pct and brown_pct < 6.0 and yellow_pct < 12.0:
            status = "HEALTHY"
            color = (0, 220, 0)
        elif healthy_pct >= mod_thresh and brown_pct < 18.0:
            status = "MODERATE STRESS"
            color = (0, 230, 255)
        else:
            status = "UNHEALTHY / STRESSED"
            color = (0, 0, 255)

    info = {
        "status": status,
        "color": color,
        "healthy_pct": healthy_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "tree_pixels": total_px,
        "healthy_thresh_pct": healthy_thresh_pct,
    }
    return tree_mask, info


def compute_vari(frame_bgr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    denom = g + r - b
    safe_denom = np.where(
        np.abs(denom) < eps,
        np.sign(denom) * eps + (denom == 0) * eps,
        denom,
    )
    return np.clip((g - r) / safe_denom, -1.0, 1.0)


def tree_vari_to_map(frame_bgr: np.ndarray, vari: np.ndarray, tree_mask: np.ndarray) -> np.ndarray:
    out = (frame_bgr.astype(np.float32) * 0.35).astype(np.uint8)
    for low, high, bgr in VARI_BINS:
        bin_mask = tree_mask & (vari >= low) & (vari < high)
        out[bin_mask] = bgr
    out[tree_mask & (vari >= VARI_BINS[-1][1])] = VARI_BINS[-1][2]
    return out


def build_legend_bar(width: int, height: int = 54) -> np.ndarray:
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    n_bins = len(VARI_BINS)
    sw_w = max(1, width // n_bins)
    cv2.putText(
        bar, "VARI (Trees Only)", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA
    )
    for idx, (low, high, bgr) in enumerate(VARI_BINS):
        x0 = idx * sw_w
        x1 = width if idx == n_bins - 1 else (idx + 1) * sw_w
        cv2.rectangle(bar, (x0, 22), (x1 - 1, 38), bgr, -1)
        if idx % 2 == 0:
            label = f"{high:.2f}" if idx < n_bins - 1 else "0.45+"
            cv2.putText(
                bar, label, (x0 + 2, 50), cv2.FONT_HERSHEY_PLAIN, 0.8, (30, 30, 30), 1, cv2.LINE_AA
            )
    return bar


def render_controller(
    res_name: str,
    res_dims: str,
    brightness: int,
    contrast: int,
    healthy_thresh: int,
) -> np.ndarray:
    ctrl = np.zeros((185, 650, 3), dtype=np.uint8)
    cv2.putText(
        ctrl, f"Res: {res_name} ({res_dims}) | Bright: {brightness:+d} | Contrast: {contrast:+d}",
        (16, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        ctrl, f"Healthy Threshold [h]: {healthy_thresh}%",
        (16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 235, 160), 2, cv2.LINE_AA
    )
    cv2.putText(
        ctrl, "[c/Enter] Snap   |   [r] Res   |   [h] Health Thresh   |   [q] Quit",
        (16, 164), cv2.FONT_HERSHEY_PLAIN, 1.05, (190, 230, 190), 1, cv2.LINE_AA
    )
    return ctrl


def capture_and_show(
    framesize: int,
    res_name: str,
    brightness: int,
    contrast: int,
    healthy_thresh: int,
) -> None:
    print(
        f"Applying settings ({res_name}, Bright={brightness:+d}, "
        f"Contrast={contrast:+d}, HealthyThresh={healthy_thresh}%)..."
    )
    try:
        apply_camera_settings(framesize, brightness, contrast)
    except Exception as exc:
        print(f"Warning: setting push issue ({exc}), attempting capture anyway...")

    frame = fetch_frame(CAPTURE_URL, timeout=12.0)
    if frame is None:
        print("Capture failed: empty image returned.")
        return

    tree_mask, health = detect_tree_and_health(frame, float(healthy_thresh))
    vari = compute_vari(frame)
    vari_map = tree_vari_to_map(frame, vari, tree_mask)
    legend = build_legend_bar(vari_map.shape[1])

    tree_px = health["tree_pixels"]
    total_px = frame.shape[0] * frame.shape[1]
    tree_pct = (tree_px / max(total_px, 1)) * 100.0

    if tree_px > 0:
        mean_tree_vari = float(np.mean(vari[tree_mask]))
        vari_line = f"Tree VARI: {mean_tree_vari:+.3f} ({tree_pct:.1f}% tree)"
    else:
        vari_line = "Tree VARI: N/A (0.0% tree)"

    health_line = (
        f"Health: {health['status']} [req>={healthy_thresh}%] | "
        f"G:{health['healthy_pct']:.0f}% Y:{health['yellow_pct']:.0f}% B:{health['brown_pct']:.0f}%"
    )

    cv2.rectangle(vari_map, (8, 8), (vari_map.shape[1] - 8, 66), (20, 20, 20), -1)
    cv2.putText(
        vari_map, vari_line, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        vari_map, health_line, (16, 57),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, health["color"], 2, cv2.LINE_AA
    )

    print(
        f"[{res_name} {frame.shape[1]}x{frame.shape[0]}] {vari_line} | {health['status']} "
        f"(req>={healthy_thresh}%, G:{health['healthy_pct']:.1f}%, Y:{health['yellow_pct']:.1f}%, B:{health['brown_pct']:.1f}%)"
    )

    vari_with_legend = np.vstack((vari_map, legend))
    rgb_resized = cv2.resize(frame, (frame.shape[1], vari_with_legend.shape[0]))
    combined = np.hstack((rgb_resized, vari_with_legend))

    max_disp_w = 1440
    if combined.shape[1] > max_disp_w:
        scale = max_disp_w / combined.shape[1]
        disp = cv2.resize(combined, (int(combined.shape[1] * scale), int(combined.shape[0] * scale)))
    else:
        disp = combined

    cv2.imshow(f"{res_name} Tree VARI + Plant Health ('c' snap, 'h' thresh, 'q' quit)", disp)


def _noop(_val: int) -> None:
    pass


def main() -> None:
    default_res_idx = 2      # VGA (640x480)
    default_thresh_pos = 6   # 50 + 6*5 = 80%

    framesize, res_name, res_dims = RESOLUTIONS[default_res_idx]
    try:
        set_control("framesize", framesize)
    except Exception as exc:
        print(f"Warning: could not set initial resolution automatically ({exc}).")

    cv2.namedWindow(CTRL_WIN)
    cv2.createTrackbar("Res (0=QVGA..4=XGA)", CTRL_WIN, default_res_idx, len(RESOLUTIONS) - 1, _noop)
    cv2.createTrackbar("Brightness (-2..+2)", CTRL_WIN, 2, 4, _noop)
    cv2.createTrackbar("Contrast (-2..+2)", CTRL_WIN, 2, 4, _noop)
    cv2.createTrackbar("Healthy % (50..95)", CTRL_WIN, default_thresh_pos, HEALTH_THRESH_MAX_POS, _noop)

    while True:
        r_pos = cv2.getTrackbarPos("Res (0=QVGA..4=XGA)", CTRL_WIN)
        b_pos = cv2.getTrackbarPos("Brightness (-2..+2)", CTRL_WIN)
        c_pos = cv2.getTrackbarPos("Contrast (-2..+2)", CTRL_WIN)
        h_pos = cv2.getTrackbarPos("Healthy % (50..95)", CTRL_WIN)

        framesize, res_name, res_dims = RESOLUTIONS[r_pos]
        brightness = b_pos - 2
        contrast = c_pos - 2
        healthy_thresh = HEALTH_THRESH_MIN + (h_pos * HEALTH_THRESH_STEP)

        cv2.imshow(
            CTRL_WIN,
            render_controller(res_name, res_dims, brightness, contrast, healthy_thresh),
        )

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            next_idx = (r_pos + 1) % len(RESOLUTIONS)
            cv2.setTrackbarPos("Res (0=QVGA..4=XGA)", CTRL_WIN, next_idx)
        elif key == ord("h"):
            next_h_pos = (h_pos + 1) % (HEALTH_THRESH_MAX_POS + 1)
            cv2.setTrackbarPos("Healthy % (50..95)", CTRL_WIN, next_h_pos)
        elif key in (ord("c"), 13):
            try:
                capture_and_show(framesize, res_name, brightness, contrast, healthy_thresh)
            except Exception as exc:
                print(f"{res_name} capture error: {exc}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()