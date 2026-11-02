import cv2, glob, os

TEST_DIR = "data/test"
OUT_DIR = "results/grid"
STEP = 50                     # 격자 간격(px)
os.makedirs(OUT_DIR, exist_ok=True)

for path in [f"{TEST_DIR}/1.png"]: 
    img = cv2.imread(path)
    h, w = img.shape[:2]
    img = cv2.resize(img, (1000, int(h * 1000 / w)))   # v1·v2와 같은 크기
    h, w = img.shape[:2]

    for x in range(0, w, STEP):
        thick = 2 if x % 100 == 0 else 1                 # 100px마다 굵게
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), thick)
        if x % 100 == 0:
            cv2.putText(img, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for y in range(0, h, STEP):
        thick = 2 if y % 100 == 0 else 1
        cv2.line(img, (0, y), (w, y), (0, 255, 0), thick)
        if y % 100 == 0:
            cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.imwrite(f"{OUT_DIR}/{os.path.basename(path)}", img)

print("저장 완료:", OUT_DIR)