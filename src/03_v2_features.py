import cv2, glob, os
import numpy as np

rows = []
for path in sorted(glob.glob("results/roi_crop/*.png")):
    name = os.path.basename(path)
    _, state, color, light, _ = name[:-4].split("_")
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    mean = gray.mean()                                   # ① 평균 밝기
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)                                # 어두운 사진도 대비 올리기
    gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)).sum()   # 세로선 세기
    gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)).sum()   # 가로선 세기
    edge_ratio = gx / (gy + 1e-6)                        # 세로 ÷ 가로
    rows.append((name, state, color, light, round(mean, 1), round(edge_ratio, 3)))


print("이름 / 상태 / 색 / 조명 / 평균밝기 / 엣지비율")
for r in sorted(rows, key=lambda r: r[5]):              # 엣지비율 작은 순
    print(r)