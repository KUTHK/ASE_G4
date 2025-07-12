import cv2
import os
import glob

IMAGE_DIR = r"C:\Users\ryoma\ASE\img"
OUTPUT_DIR = r"C:\Users\ryoma\ASE\anno"
CLASS_INDEX = 0  # 必要に応じて変更

points = []
annotations = []

def mouse_callback(event, x, y, flags, param):
    global points, img_disp
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img_disp, (x, y), 5, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.line(img_disp, points[-2], points[-1], (255, 0, 0), 2)
        if len(points) == 4:
            cv2.line(img_disp, points[3], points[0], (255, 0, 0), 2)
        cv2.imshow("image", img_disp)

def normalize_point(pt, w, h):
    return pt[0] / w, pt[1] / h

def annotate_image(img_path):
    global points, img_disp, annotations
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    annotations = []
    while True:
        img_disp = img.copy()
        points.clear()
        cv2.imshow("image", img_disp)
        cv2.setMouseCallback("image", mouse_callback)
        # 4点入力待ち
        while len(points) < 4:
            cv2.imshow("image", img_disp)
            key = cv2.waitKey(1)
            if key == 27:  # ESCで画像スキップ
                return
            if key == 13:  # Enterで次画像へ
                break
        if len(points) == 4:
            norm_points = [normalize_point(pt, w, h) for pt in points]
            line = f"{CLASS_INDEX} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_points)
            annotations.append(line)
        # Enterで次画像へ
        key = cv2.waitKey(0)
        if key == 13:  # Enter
            break
        if key == 27:  # ESC
            return
    # 保存
    if annotations:
        base = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(OUTPUT_DIR, base + ".txt")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(txt_path, "w") as f:
            for line in annotations:
                f.write(line + "\n")

def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    for img_path in image_paths:
        print(f"Annotating: {img_path}")
        annotate_image(img_path)
    cv2.destroyAllWindows()
    print("アノテーション完了")

if __name__ == "__main__":
    main()