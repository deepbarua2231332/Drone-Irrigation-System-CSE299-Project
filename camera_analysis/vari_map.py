import cv2
import numpy as np

# (min_vari, max_vari, BGR_color)
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


def compute_vari(frame_bgr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute VARI = (G - R) / (G + R - B) clipped to [-1.0, 1.0]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    denom = g + r - b
    safe_denom = np.where(
        np.abs(denom) < eps,
        np.sign(denom) * eps + (denom == 0.0) * eps,
        denom,
    )
    return np.clip((g - r) / safe_denom, -1.0, 1.0)


def render_vari_map(
    frame_bgr: np.ndarray,
    vari: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Colorize VARI bins over active_mask (or full image if None)."""
    if active_mask is None:
        active_mask = np.ones(vari.shape, dtype=bool)

    out = (frame_bgr.astype(np.float32) * 0.35).astype(np.uint8)
    for low, high, bgr in VARI_BINS:
        m = active_mask & (vari >= low) & (vari < high)
        out[m] = bgr
    out[active_mask & (vari >= VARI_BINS[-1][1])] = VARI_BINS[-1][2]
    return out


def build_vari_legend(width: int, height: int = 52) -> np.ndarray:
    """Create a horizontal color legend bar for VARI bins."""
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    n = len(VARI_BINS)
    sw = max(1, width // n)
    cv2.putText(bar, "VARI Index", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    for i, (_, high, bgr) in enumerate(VARI_BINS):
        x0 = i * sw
        x1 = width if i == n - 1 else (i + 1) * sw
        cv2.rectangle(bar, (x0, 21), (x1 - 1, 36), bgr, -1)
        if i % 2 == 0:
            lbl = f"{high:.2f}" if i < n - 1 else "0.45+"
            cv2.putText(bar, lbl, (x0 + 2, 48), cv2.FONT_HERSHEY_PLAIN, 0.8, (30, 30, 30), 1, cv2.LINE_AA)
    return bar