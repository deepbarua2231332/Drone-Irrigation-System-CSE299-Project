import cv2
import numpy as np


def classify_plant_health(
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray | None = None,
    healthy_thresh_pct: float = 70.0,
    min_foliage_px: int = 200,
) -> tuple[np.ndarray, dict]:
    """Classify foliage inside roi_mask into Healthy, Yellow/Chlorotic, or Brown/Dry."""
    h_img, w_img = frame_bgr.shape[:2]
    if roi_mask is None:
        roi_mask = np.ones((h_img, w_img), dtype=bool)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Broad vegetation candidate inside land ROI
    plant_mask = (
        roi_mask &
        (h >= 18) & (h <= 95) &
        (s >= 35) &
        (v >= 35) & (v <= 245)
    )

    # Strict healthy green: excludes pale/washed-out or dull leaves
    healthy_mask = (
        plant_mask &
        (g > r * 1.12) &
        (g >= b * 1.00) &
        (s >= 50) &
        (h >= 36) & (h <= 88)
    )

    # Yellow / pale / chlorotic foliage
    yellow_mask = plant_mask & (~healthy_mask) & (h >= 18) & (h < 36)

    # Brown / dry / necrotic tissue inside land ROI
    brown_mask = (
        roi_mask &
        (~plant_mask) &
        (v >= 35) & (v <= 210) &
        (r > g * 1.15) & (g > b * 1.05) &
        (h >= 5) & (h < 22)
    )

    plant_px = int(np.sum(plant_mask))
    healthy_px = int(np.sum(healthy_mask))
    yellow_px = int(np.sum(yellow_mask))
    brown_px = int(np.sum(brown_mask))

    denom = max(plant_px + brown_px, 1)
    healthy_pct = (healthy_px / denom) * 100.0
    yellow_pct = (yellow_px / denom) * 100.0
    brown_pct = (brown_px / denom) * 100.0

    mod_thresh = max(35.0, healthy_thresh_pct - 25.0)
    if plant_px < min_foliage_px:
        status = "NO CLEAR PLANT IN ROI"
        color = (0, 200, 255)
    elif healthy_pct >= healthy_thresh_pct and brown_pct < 12.0:
        status = "HEALTHY"
        color = (0, 220, 0)
    elif healthy_pct >= mod_thresh:
        status = "MODERATE / MILD STRESS"
        color = (0, 230, 255)
    else:
        status = "UNHEALTHY / STRESSED"
        color = (0, 0, 255)

    # Overlay on dimmed non-ROI background
    vis = (frame_bgr.astype(np.float32) * 0.35).astype(np.uint8)
    vis[roi_mask] = frame_bgr[roi_mask]

    overlay = vis.copy()
    overlay[healthy_mask] = (60, 200, 60)
    overlay[yellow_mask] = (0, 220, 255)
    overlay[brown_mask] = (40, 60, 210)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    cv2.rectangle(vis, (8, 8), (430, 108), (20, 20, 20), -1)
    cv2.putText(vis, f"Status : {status}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Healthy: {healthy_pct:.1f}% (req>={healthy_thresh_pct:.0f}%) | Yellow: {yellow_pct:.1f}%",
                (16, 60), cv2.FONT_HERSHEY_PLAIN, 1.05, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Brown/Dry: {brown_pct:.1f}%  |  Foliage px: {plant_px}",
                (16, 88), cv2.FONT_HERSHEY_PLAIN, 1.05, (220, 220, 220), 1, cv2.LINE_AA)

    metrics = {
        "status": status,
        "healthy_pct": healthy_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "foliage_pixels": plant_px,
    }
    return vis, metrics