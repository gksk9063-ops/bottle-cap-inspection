import cv2, glob, os, sys, csv
import numpy as np

SET = sys.argv[1] if len(sys.argv) > 1 else "test"
MODE = sys.argv[2] if len(sys.argv) > 2 else "auto"        # fixed 또는 auto
CROP_DIR = f"results/{SET}_autocrop" if MODE == "auto" else f"results/{SET}_crop"
DARK_LIMIT = 50        # 평균 밝기가 이보다 낮으면 RETAKE
THRESHOLD = 0.52       # 세로/가로 엣지 비율 (test 24장으로 정함)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

rows, correct, judged = [], 0, 0
for path in sorted(glob.glob(f"{CROP_DIR}/*.png")):
    name = os.path.basename(path)
    gt = "OK" if name.startswith("cap_on") else "NG"
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mean = gray.mean()

    g = clahe.apply(gray)
    gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)).sum()
    gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)).sum()
    ratio = gx / (gy + 1e-6)

    if mean < DARK_LIMIT:
        pred = "RETAKE"
    else:
        pred = "OK" if ratio >= THRESHOLD else "NG"
        judged += 1
        correct += (pred == gt)
    rows.append([name, gt, pred, round(mean, 1), round(ratio, 3)])

os.makedirs("reports", exist_ok=True)
with open(f"reports/v3_{SET}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "gt", "pred", "mean", "ratio"]); w.writerows(rows)

print(f"[{SET}] v3 정확도: {correct}/{judged}  (RETAKE {len(rows) - judged}장)")
for r in rows:
    print(r)