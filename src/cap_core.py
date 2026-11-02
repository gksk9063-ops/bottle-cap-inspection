"""병뚜껑 검사 v3.1 핵심 규칙 — 사진 검사(07)와 실시간(08)이 같이 쓰는 함수 모음."""
import cv2
import numpy as np

DARK_LIMIT = 50      # 평균 밝기 < 50 → RETAKE
TILT_LIMIT = 12      # 기울기(°) > 12 → RETAKE
THRESHOLD = 0.52     # 세로/가로 엣지 비율 ≥ 0.52 → OK (Calibration 32장)
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def resize_w1000(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (1000, int(h * 1000 / w)))


def silhouette(img):
    """CLAHE → Sobel 엣지 → 구멍 메우기 → 가장 큰 덩어리 = 병."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = _clahe.apply(cv2.GaussianBlur(gray, (5, 5), 0))
    mag = cv2.magnitude(cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1))
    mask = (mag > 20).astype(np.uint8) * 255
    mask[:15, :] = 0; mask[-5:, :] = 0; mask[:, :5] = 0; mask[:, -5:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    obj = np.zeros_like(mask)
    cv2.drawContours(obj, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return obj


def _y_top(obj):
    rows = np.where((obj > 0).sum(1) > 50)[0]
    return rows[0] if len(rows) else None


def tilt_angle(obj):
    """실루엣 줄마다 가운데 x → 직선 기울기(°). 잴 수 없으면 None."""
    H, W = obj.shape
    y_top = _y_top(obj)
    if y_top is None:
        return None
    ys, cs = [], []
    for y in range(y_top + 10, min(y_top + 400, H - 1), 5):
        xs = np.where(obj[y] > 0)[0]
        if len(xs) == 0 or xs[0] <= 6 or xs[-1] >= W - 7:
            continue
        ys.append(y); cs.append((xs[0] + xs[-1]) / 2)
    if len(ys) < 10:
        return None
    return float(np.degrees(np.arctan(np.polyfit(ys, cs, 1)[0])))


def find_roi(obj):
    """병 맨 윗부분 폭 w 기준 ROI (1.2w × 1.1w)."""
    H, W = obj.shape
    y_top = _y_top(obj)
    if y_top is None:
        return None
    L, R = [], []
    for row in obj[y_top + 20:y_top + 80]:
        xs = np.where(row > 0)[0]
        if len(xs):
            L.append(xs[0]); R.append(xs[-1])
    if not L:
        return None
    x1, x2 = int(np.median(L)), int(np.median(R))
    w, cx = x2 - x1, (x1 + x2) // 2
    if w < 30:
        return None
    rx, ry = max(cx - int(0.6 * w), 0), max(y_top - int(0.05 * w), 0)
    return rx, ry, min(int(1.2 * w), W - rx), min(int(1.1 * w), H - ry)


def edge_ratio(crop_bgr):
    """세로선 세기 ÷ 가로선 세기 (뚜껑 줄무늬=세로, 병 나사산=가로)."""
    g = _clahe.apply(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
    gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)).sum()
    gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)).sum()
    return gx / (gy + 1e-6)


def judge(img):
    """사진 1장 → dict(verdict, reason, roi, tilt, ratio). img는 폭 1000으로 맞춘 BGR."""
    out = {"verdict": "RETAKE", "reason": "", "roi": None, "tilt": None, "ratio": None}
    if cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < DARK_LIMIT:
        out["reason"] = "too dark"; return out
    obj = silhouette(img)
    if obj is None:
        out["reason"] = "no bottle"; return out
    out["tilt"] = tilt_angle(obj)
    if out["tilt"] is not None and abs(out["tilt"]) > TILT_LIMIT:
        out["reason"] = "tilted"; return out
    roi = find_roi(obj)
    if roi is None:
        out["reason"] = "no roi"; return out
    x, y, w, h = roi
    out["roi"] = roi
    out["ratio"] = edge_ratio(img[y:y + h, x:x + w])
    out["verdict"] = "OK" if out["ratio"] >= THRESHOLD else "NG"
    return out
