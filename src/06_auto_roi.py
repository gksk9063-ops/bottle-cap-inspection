import cv2, glob, os, sys
import numpy as np

SET = sys.argv[1] if len(sys.argv) > 1 else "test"
BOX_DIR, CROP_DIR = f"results/{SET}_autobox", f"results/{SET}_autocrop"
os.makedirs(BOX_DIR, exist_ok=True); os.makedirs(CROP_DIR, exist_ok=True)
TILT_LIMIT = 12        # 이 각도(°)보다 기울면 RETAKE

def silhouette(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blur)
    mag = cv2.magnitude(cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1))
    mask = (mag > 20).astype(np.uint8) * 255
    mask[:15, :] = 0; mask[-5:, :] = 0; mask[:, :5] = 0; mask[:, -5:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obj = np.zeros_like(mask)
    cv2.drawContours(obj, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return obj

def tilt_angle(obj):
    H, W = obj.shape
    y_top = np.where((obj > 0).sum(1) > 50)[0][0]
    ys, cs = [], []
    for y in range(y_top + 10, min(y_top + 400, H - 1), 5):
        xs = np.where(obj[y] > 0)[0]
        if len(xs) == 0 or xs[0] <= 6 or xs[-1] >= W - 7:   # 사진 끝에 닿은 줄은 제외
            continue
        ys.append(y); cs.append((xs[0] + xs[-1]) / 2)
    if len(ys) < 10:
        return None
    slope = np.polyfit(ys, cs, 1)[0]                 # 아래로 1px 갈 때 가운데가 옆으로 움직인 양
    return float(np.degrees(np.arctan(slope)))

def find_roi(obj):
    H, W = obj.shape
    y_top = np.where((obj > 0).sum(1) > 50)[0][0]
    L, R = [], []
    for row in obj[y_top + 20:y_top + 80]:
        xs = np.where(row > 0)[0]
        if len(xs): L.append(xs[0]); R.append(xs[-1])
    x1, x2 = int(np.median(L)), int(np.median(R))
    w, cx = x2 - x1, (x1 + x2) // 2
    rx, ry = max(cx - int(0.6 * w), 0), max(y_top - int(0.05 * w), 0)
    return rx, ry, min(int(1.2 * w), W - rx), min(int(1.1 * w), H - ry)

for path in sorted(glob.glob(f"data/{SET}/*.png")):
    name = os.path.basename(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (1000, int(img.shape[0] * 1000 / img.shape[1])))
    if cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < 50:
        print(name, "RETAKE (너무 어두움)")
        continue
    obj = silhouette(img)
    angle = tilt_angle(obj)
    if angle is not None and abs(angle) > TILT_LIMIT:
        print(name, f"RETAKE (기울어짐 {angle:.1f})")
        continue
    tilt_txt = "None" if angle is None else f"{angle:.1f}"
    x, y, w, h = find_roi(obj)
    box = img.copy()
    cv2.rectangle(box, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.putText(box, f"tilt {tilt_txt}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(f"{BOX_DIR}/{name}", box)
    cv2.imwrite(f"{CROP_DIR}/{name}", img[y:y + h, x:x + w])
    print(name, (x, y, w, h), f"tilt {tilt_txt}")