import urllib.request
import cv2
import numpy as np

BASE_URL = "http://192.168.4.1"
CAPTURE_URL = f"{BASE_URL}/capture"
CTRL_WIN = "Plant Health Controller"


def set_control(var: str, val: int, timeout: float = 3.0) -> None:
    """Send a /control command to the ESP32-CAM."""
    url = f"{BASE_URL}/control?var={var}&val={val}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        resp.read()


def fetch_frame(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        buf = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def build_land_roi_mask(
    shape_hw: tuple[int, int],
    top_pct: int,
    bottom_pct: int,
    left_pct: int,
    right_pct: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Create a boolean mask for the selected land area inside the image."""
    h, w = shape_hw

    # Ensure valid non-empty box even if sliders cross
    y0 = int(round(h * min(top_pct, max(0, bottom_pct - 5)) / 100.0))
    y1 = int(round(h * max(bottom_pct, min(100, top_pct + 5)) / 100.0))
    x0 = int(round(w * min(left_pct, max(0, right_pct - 5)) / 100.0))
    x1 = int(round(w * max(right_pct, min(100, left_pct + 5)) / 100.0))

    y0 = max(0, min(h - 2, y0))
    y1 = max(y0 + 1, min(h, y1))
    x0 = max(0, min(w - 2, x0))
    x1 = max(x0 + 1, min(w, x1))

    land_mask = np.zeros((h, w), dtype=bool)
    land_mask[y0:y1, x0:x1] = True
    return land_mask, (x0, y0, x1, y1)


def assess_plant_health_land_only(
    frame_bgr: np.ndarray,
    land_mask: np.ndarray,
    roi_box: tuple[int, int, int, int],
    healthy_thresh_pct: float = 70.0,
) -> tuple[np.ndarray, dict]:
    """Analyze plant health strictly inside the selected land ROI mask."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 1. Broad vegetation mask restricted to the land area
    plant_mask = (
        land_mask &
        (h >= 18) & (h <= 95) &
        (s >= 35) &
        (v >= 35) & (v <= 245)
    )

    # 2. Healthy green foliage
    healthy_mask = plant_mask & (g > r * 1.08) & (g >= b * 0.95) & (h >= 35) & (h <= 88)

    # 3. Yellow / pale / chlorotic stress
    yellow_mask = plant_mask & (~healthy_mask) & (h >= 18) & (h < 35)

    # 4. Brown / dry / necrotic tissue inside the land area
    brown_mask = (
        land_mask &
        (~plant_mask) &
        (v >= 35) & (v <= 210) &
        (r > g * 1.15) & (g > b * 1.05) &
        (h >= 5) & (h < 22)
    )

    total_plant_px = int(np.sum(plant_mask))
    healthy_px = int(np.sum(healthy_mask))
    yellow_px = int(np.sum(yellow_mask))
    brown_px = int(np.sum(brown_mask))

    denom = max(total_plant_px + brown_px, 1)
    healthy_pct = (healthy_px / denom) * 100.0
    yellow_pct = (yellow_px / denom) * 100.0
    brown_pct = (brown_px / denom) * 100.0

    mod_thresh = max(35.0, healthy_thresh_pct - 25.0)
    if total_plant_px < 200:
        status = "NO CLEAR PLANT IN LAND"
        status_color = (0, 200, 255)
    elif healthy_pct >= healthy_thresh_pct and brown_pct < 12.0:
        status = "HEALTHY"
        status_color = (0, 220, 0)
    elif healthy_pct >= mod_thresh:
        status = "MODERATE / MILD STRESS"
        status_color = (0, 230, 255)
    else:
        status = "UNHEALTHY / STRESSED"
        status_color = (0, 0, 255)

    # Dim outside land area so unwanted space is visually excluded
    vis = (frame_bgr.astype(np.float32) * 0.35).astype(np.uint8)
    vis[land_mask] = frame_bgr[land_mask]

    overlay = vis.copy()
    overlay[healthy_mask] = (60, 200, 60)
    overlay[yellow_mask] = (0, 220, 255)
    overlay[brown_mask] = (40, 60, 210)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    # Draw land boundary rectangle
    x0, y0, x1, y1 = roi_box
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # Summary banner
    cv2.rectangle(vis, (8, 8), (430, 112), (20, 20, 20), -1)
    cv2.putText(vis, f"Status : {status}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Healthy: {healthy_pct:.1f}% (req>={healthy_thresh_pct:.0f}%) | Yellow: {yellow_pct:.1f}%",
                (16, 60), cv2.FONT_HERSHEY_PLAIN, 1.05, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Brown/Dry: {brown_pct:.1f}%  |  Land Foliage px: {total_plant_px}",
                (16, 88), cv2.FONT_HERSHEY_PLAIN, 1.05, (220, 220, 220), 1, cv2.LINE_AA)

    metrics = {
        "status": status,
        "healthy_pct": healthy_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "foliage_pixels": total_plant_px,
    }
    return vis, metrics


def render_controller(
    top_pct: int,
    bottom_pct: int,
    left_pct: int,
    right_pct: int,
    healthy_thresh: int,
    brightness: int,
    contrast: int,
) -> np.ndarray:
    ctrl = np.zeros((270, 620, 3), dtype=np.uint8)
    cv2.putText(
        ctrl, f"Land ROI -> Top:{top_pct}% Bot:{bottom_pct}% Left:{left_pct}% Right:{right_pct}%",
        (16, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        ctrl, f"Healthy Req: {healthy_thresh}% | Bright: {brightness:+d} | Contrast: {contrast:+d}",
        (16, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        ctrl, "[c/Enter] Snap Land Check   |   [q] Quit",
        (16, 255), cv2.FONT_HERSHEY_PLAIN, 1.15, (180, 230, 180), 1, cv2.LINE_AA
    )
    return ctrl


def _noop(_val: int) -> None:
    pass


def main():
    cv2.namedWindow(CTRL_WIN)
    # Land area crop sliders (0..100% of frame height/width)
    cv2.createTrackbar("Land Top %", CTRL_WIN, 0, 95, _noop)
    cv2.createTrackbar("Land Bottom %", CTRL_WIN, 100, 100, _noop)
    cv2.createTrackbar("Land Left %", CTRL_WIN, 0, 95, _noop)
    cv2.createTrackbar("Land Right %", CTRL_WIN, 100, 100, _noop)
    # Health & camera sliders
    cv2.createTrackbar("Healthy %", CTRL_WIN, 70, 95, _noop)
    cv2.createTrackbar("Brightness (-2..+2)", CTRL_WIN, 2, 4, _noop)
    cv2.createTrackbar("Contrast (-2..+2)", CTRL_WIN, 2, 4, _noop)

    print("Adjust Land ROI sliders to select only the land area, then press 'c' or ENTER | 'q' to quit")

    while True:
        top_pct = cv2.getTrackbarPos("Land Top %", CTRL_WIN)
        bot_pct = cv2.getTrackbarPos("Land Bottom %", CTRL_WIN)
        left_pct = cv2.getTrackbarPos("Land Left %", CTRL_WIN)
        right_pct = cv2.getTrackbarPos("Land Right %", CTRL_WIN)
        healthy_thresh = max(40, cv2.getTrackbarPos("Healthy %", CTRL_WIN))
        brightness = cv2.getTrackbarPos("Brightness (-2..+2)", CTRL_WIN) - 2
        contrast = cv2.getTrackbarPos("Contrast (-2..+2)", CTRL_WIN) - 2

        cv2.imshow(
            CTRL_WIN,
            render_controller(top_pct, bot_pct, left_pct, right_pct, healthy_thresh, brightness, contrast),
        )

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("c"), 13):
            try:
                try:
                    set_control("brightness", brightness)
                    set_control("contrast", contrast)
                except Exception as cam_exc:
                    print(f"Warning: camera setting push skipped ({cam_exc})")

                frame = fetch_frame(CAPTURE_URL)
                if frame is None:
                    print("Failed to decode captured image.")
                    continue

                land_mask, roi_box = build_land_roi_mask(
                    frame.shape[:2], top_pct, bot_pct, left_pct, right_pct
                )
                result_img, info = assess_plant_health_land_only(
                    frame, land_mask, roi_box, float(healthy_thresh)
                )

                # Show ROI outline on original left image too for quick alignment
                left_preview = frame.copy()
                x0, y0, x1, y1 = roi_box
                cv2.rectangle(left_preview, (x0, y0), (x1, y1), (0, 255, 255), 2)

                print(
                    f"Result -> {info['status']} | "
                    f"Healthy: {info['healthy_pct']:.1f}%, "
                    f"Yellow: {info['yellow_pct']:.1f}%, "
                    f"Brown/Dry: {info['brown_pct']:.1f}% | "
                    f"Land Foliage px: {info['foliage_pixels']}"
                )
                combined = np.hstack((left_preview, result_img))
                cv2.imshow("Single-Image Land Plant Health Check", combined)
            except Exception as exc:
                print(f"Error capturing/analyzing image: {exc}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()