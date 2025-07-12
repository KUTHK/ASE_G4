import cv2
import os
import glob

IMAGE_DIR = r"C:\Users\ryoma\ase\img"
OUTPUT_DIR = r"C:\Users\ryoma\ase\anno"
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
    
    print(f"画像: {os.path.basename(img_path)}")
    print("操作方法:")
    print("- 左クリックで4点を選択してオブジェクトをアノテーション")
    print("- Spaceキー: 次のオブジェクトをアノテーション")
    print("- Enterキー: この画像のアノテーション完了（次の画像へ）")
    print("- ESCキー: この画像をスキップ")
    
    while True:
        img_disp = img.copy()
        
        # 既存のアノテーションを描画
        for i, line in enumerate(annotations):
            parts = line.split()
            coords = [float(x) for x in parts[1:]]
            pts = [(int(coords[i] * w), int(coords[i+1] * h)) for i in range(0, len(coords), 2)]
            for j in range(len(pts)):
                cv2.line(img_disp, pts[j], pts[(j+1) % len(pts)], (0, 255, 255), 2)  # 黄色で既存アノテーション
            # アノテーション番号を表示
            cv2.putText(img_disp, f"#{i+1}", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        points.clear()
        cv2.imshow("image", img_disp)
        cv2.setMouseCallback("image", mouse_callback)
        
        # 4点入力待ち
        while len(points) < 4:
            cv2.imshow("image", img_disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCで画像スキップ
                return
            if key == 13:  # Enterで次画像へ
                break
            if key == 32:  # Spaceで次のオブジェクト（4点未満の場合は無効）
                if len(points) == 0:
                    break
        
        if len(points) == 4:
            norm_points = [normalize_point(pt, w, h) for pt in points]
            line = f"{CLASS_INDEX} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_points)
            annotations.append(line)
            print(f"オブジェクト #{len(annotations)} をアノテーションしました")
            
            # 4点完了後の待機
            print("Spaceキー: 次のオブジェクト, Enterキー: 次の画像, ESCキー: スキップ")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 32:  # Spaceで次のオブジェクト
                    break
                elif key == 13:  # Enterで次画像へ
                    return
                elif key == 27:  # ESCで画像スキップ
                    return
        else:
            # 4点未満でEnterまたはSpaceが押された場合
            break
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