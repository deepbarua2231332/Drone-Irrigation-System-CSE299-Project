import urllib.request
import cv2
import numpy as np

from roi_crop import build_land_roi, dim_outside_roi
from vari_map import compute_vari, render_vari_map, build_vari_legend
from health_classifier import classify_plant_health

BASE_URL = "http://192.168.4.1"
CAPTURE_URL = f"{BASE_URL}/capture"
CTRL_WIN = "Plant Health Controller"


def set_control(var: str, val: int, timeout: float = 3.0) -> None:
    with urllib.request.urlopen(f"{BASE_URL}/control?var={var}&val={val}", timeout=timeout) as resp:
        resp.read()


def fetch_frame(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        buf = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _noop(_val: int) -> None:
    pass


def main() -> None:
    cv2.namedWindow(CTRL_WIN)
    cv2.createTrackbar("Land Top %", CTRL_WIN, 0, 95, _noop)
    cv2.createTrackbar("Land Bottom %", CTRL_WIN, 100, 100, _noop)
    cv2.createTrackbar("Land Left %", CTRL_WIN, 0, 95, _noop)
    cv2.createTrackbar("Land Right %", CTRL_WIN, 100, 100, _noop)
    cv2.createTrackbar("Healthy %", CTRL_WIN, 70, 95, _noop)
    cv2.createTrackbar("Brightness (-2..+2)", CTRL_WIN, 2, 4, _noop)
    cv2.createTrackbar("Contrast (-2..+2)", CTRL_WIN, 2, 4, _noop)

    while True:
        top_pct = cv2.getTrackbarPos("Land Top %", CTRL_WIN)
        bot_pct = cv2.getTrackbarPos("Land Bottom %", CTRL_WIN)
        left_pct = cv2.getTrackbarPos("Land Left %", CTRL_WIN)
        right_pct = cv2.getTrackbarPos("Land Right %", CTRL_WIN)
        healthy_thresh = max(40, cv2.getTrackbarPos("Healthy %", CTRL_WIN))
        brightness = cv2.getTrackbarPos("Brightness (-2..+2)", CTRL_WIN) - 2
        contrast = cv2.getTrackbarPos("Contrast (-2..+2)", CTRL_WIN) - 2

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("c"), 13):
            try:
                set_control("brightness", brightness)
                set_control("contrast", contrast)
            except Exception as exc:
                print(f"Camera control warning: {exc}")

            frame = fetch_frame(CAPTURE_URL)
            if frame is None:
                print("Failed to decode captured frame.")
                continue

            roi_mask, roi_box = build_land_roi(frame.shape[:2], top_pct, bot_pct, left_pct, right_pct)
            health_vis, info = classify_plant_health(frame, roi_mask, float(healthy_thresh))

            vari = compute_vari(frame)
            vari_vis = render_vari_map(frame, vari, active_mask=roi_mask)
            legend = build_vari_legend(vari_vis.shape[1])
            vari_panel = np.vstack((vari_vis, legend))

            left_vis = dim_outside_roi(frame, roi_mask, roi_box)
            left_panel = cv2.resize(left_vis, (left_vis.shape[1], vari_panel.shape[0]))
            health_panel = cv2.resize(health_vis, (health_vis.shape[1], vari_panel.shape[0]))

            combined = np.hstack((left_panel, health_panel, vari_panel))
            print(
                f"{info['status']} | Healthy: {info['healthy_pct']:.1f}% | "
                f"Yellow: {info['yellow_pct']:.1f}% | Brown/Dry: {info['brown_pct']:.1f}%"
            )
            cv2.imshow("Land ROI | Health Classification | VARI Map", combined)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()