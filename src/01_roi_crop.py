import cv2, glob, os, sys

SET = sys.argv[1] if len(sys.argv) > 1 else "test"
TEST_DIR = f"data/{SET}"
BOX_DIR, CROP_DIR = f"results/{SET}_box", f"results/{SET}_crop"
ROI = (90, 240, 800, 800)          # test와 holdout 모두 같은 ROI
os.makedirs(BOX_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

x, y, w, h = ROI
for path in sorted(glob.glob(f"{TEST_DIR}/*.png")):
    name = os.path.basename(path)
    img = cv2.imread(path)
    H, W = img.shape[:2]
    img = cv2.resize(img, (1000, int(H * 1000 / W)))
    crop = img[y:y + h, x:x + w]
    box = img.copy()
    cv2.rectangle(box, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite(f"{BOX_DIR}/{name}", box)
    cv2.imwrite(f"{CROP_DIR}/{name}", crop)

print("완료:", SET)