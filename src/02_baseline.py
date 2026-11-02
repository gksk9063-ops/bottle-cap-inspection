import cv2, csv, os, glob, sys
import numpy as np

SET = sys.argv[1] if len(sys.argv) > 1 else "test"
CROP_DIR = f"results/{SET}_crop"
OUT_DIR = f"results/v1_{SET}"
THRESHOLD = 0.3608                     # 수업 기준값 그대로
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
os.makedirs(OUT_DIR, exist_ok=True)

rows, correct = [], 0
for path in sorted(glob.glob(f"{CROP_DIR}/*.png")):
    name = os.path.basename(path)
    gt = "OK" if name.startswith("cap_on") else "NG"
    _, _, color, light, _ = name[:-4].split("_")

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    ratio = np.count_nonzero(binary) / binary.size
    pred = "OK" if ratio >= THRESHOLD else "NG"
    correct += (pred == gt)
    rows.append([name, gt, pred, color, light, round(t, 1), round(ratio, 4)])
    cv2.imwrite(f"{OUT_DIR}/{name}", binary)

with open(f"reports/v1_{SET}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "gt", "pred", "color", "light", "otsu_t", "area_ratio"])
    w.writerows(rows)

print(f"[{SET}] v1 정확도: {correct}/{len(rows)}")
for r in sorted(rows, key=lambda r: r[6]):
    print(r)