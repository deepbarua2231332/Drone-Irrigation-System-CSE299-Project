import cv2
import numpy as np


def build_land_roi(
    shape_hw: tuple[int, int],
    top_pct: int = 0,
    bottom_pct: int = 100,
    left_pct: int = 0,
    right_pct: int = 100,
    min_span_pct: int = 5,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (roi_mask, (x0, y0, x1, y1)) for a percentage-based land box."""
    h, w = shape_hw

    t = max(0, min(100 - min_span_pct, top_pct))
    b = max(t + min_span_pct, min(100, bottom_pct))
    l = max(0, min(100 - min_span_pct, left_pct))
    r = max(l + min_span_pct, min(100, right_pct))

    y0 = int(round(h * t / 100.0))
    y1 = int(round(h * b / 100.0))
    x0 = int(round(w * l / 100.0))
    x1 = int(round(w * r / 100.0))

    y0 = max(0, min(h - 2, y0))
    y1 = max(y0 + 1, min(h, y1))
    x0 = max(0, min(w - 2, x0))
    x1 = max(x0 + 1, min(w, x1))

    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask, (x0, y0, x1, y1)


def crop_to_roi(frame: np.ndarray, roi_box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame or single-channel map to (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = roi_box
    return frame[y0:y1, x0:x1].copy()


def dim_outside_roi(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray,
    roi_box: tuple[int, int, int, int],
    dim_factor: float = 0.35,
) -> np.ndarray:
    """Dim pixels outside the selected land ROI and draw the ROI border."""
    vis = (frame_bgr.astype(np.float32) * dim_factor).astype(np.uint8)
    vis[roi_mask] = frame_bgr[roi_mask]
    x0, y0, x1, y1 = roi_box
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return vis