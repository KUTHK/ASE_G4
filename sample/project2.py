import math
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from sklearn.decomposition import PCA

model = YOLO(r"../flask/yolov8n-seg.pt")
obb_model = YOLO(r"../flask/best.pt")

PILLAR_DISTANCE = 3.5 # m
PIXEL_PER_METER = None

def detect_lines(gray):
    # # コントラスト制御
    # clane = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clane.apply(gray)
    # gray = cv2.equalizeHist(gray)
    
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    gray = np.uint8(np.clip(lap, 0, 255))
    # 大津の二値化
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ノイズ除去
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # blur = cv2.GaussianBlur(thresh, (5, 5), 0)

    # エッジ検出
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Hough変換で直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        # まず縦線を抽出し、座標を保存
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 100:
                vertical_lines.append(((x1, y1), (x2, y2)))
                cv2.line(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)

    show_image(gray)
    # show_image(apply)

    return vertical_lines, horizontal_lines

def detect_lines2(gray):
    # コントラスト制御
    clane = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clane.apply(gray)

    # ノイズ除去
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobelフィルタで縦方向エッジ（x方向の変化）を抽出
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(np.clip(sobelx, 0, 255))

    # 二値化（必要なら）
    _, edges = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)

    # Hough変換で直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    vertical_lines = []
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 100:
                vertical_lines.append(((x1, y1), (x2, y2)))
                cv2.line(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)

    show_image(edges)
    return vertical_lines, horizontal_lines

def resize(image, target_size):
    """
    create a square image with resizing and padding
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # cv2.imwrite("resized_image.jpg", padded)
    print(f"Resized image shape: {padded.shape}")
    return padded

def mask(image, result):
    if not hasattr(result, 'masks') or result.masks is None:
        print("No masks found in the result.")
        return image

    # extract the segmented area from the YOLO result
    mask_tensor = result.masks.data
    mask_np = mask_tensor.cpu().numpy() if hasattr(mask_tensor, "cpu") else mask_tensor 
    
    combined_mask = np.zeros(mask_np[0].shape, dtype=bool)
    
    for m in mask_np:
        combined_mask |= (m > 0.5)

    masked_img = image.copy()
    masked_img[combined_mask, :] = 0
    
    # extract max and min y cordinate from the mask
    max_y = np.max(np.where(combined_mask)[0])
    min_y = np.min(np.where(combined_mask)[0])
    # print(f"Mask Y coordinates: min={min_y}, max={max_y}")
    return masked_img, max_y, min_y
    
def inference(image):
    results = model(image)
    annoted = results[0].plot()
    # cv2.imshow('Annotated Image', annoted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    centroids = []
    if hasattr(results[0], "masks") and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # shape: (N, H, W)
        for mask in masks:
            mask_uint8 = (mask * 255).astype(np.uint8)
            M = cv2.moments(mask_uint8)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
            else:
                centroids.append(None)
    print("重心座標リスト:", centroids)
    return results, centroids

def pillar_inference(image):
    results = obb_model(image)
    annoted = results[0].plot()
    
    vertical_lines = []

    obb_boxes = results[0].obb.xyxyxyxy  # shape: (N, 8)
    if obb_boxes is not None:
        for box in obb_boxes:
            if hasattr(box, "cpu"):
                box = box.cpu().numpy()
            pts = np.array(box, dtype=np.int32).reshape(4, 2)
            # 4辺の長さを計算
            dists = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
            # 短辺2本のインデックスを取得
            short_idx = np.argsort(dists)[:2]
            # 各短辺の中点を計算
            midpoints = []
            for idx in short_idx:
                p1 = pts[idx]
                p2 = pts[(idx+1)%4]
                mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                midpoints.append(mid)
            # 2つの中点を結ぶ線を描画
            if len(midpoints) == 2:
                cv2.line(annoted, midpoints[0], midpoints[1], (0, 0, 255), 2)
            vertical_lines.append(midpoints)

    show_image(annoted)
    return vertical_lines

def pillar_inference_pca(image):
    results = obb_model(image)
    annoted = results[0].plot()
    
    vertical_lines = []
    obb_boxes = results[0].obb.xyxyxyxy  # shape: (N, 8)

    if obb_boxes is not None:
        for box in obb_boxes:
            if hasattr(box, "cpu"):
                box = box.cpu().numpy()
            pts = np.array(box, dtype=np.float32).reshape(4, 2)

            # PCA を実行
            pca = PCA(n_components=2)
            pca.fit(pts)

            center = np.mean(pts, axis=0)  # 重心（中心点）
            direction = pca.components_[1]  # 第2主成分：短辺方向（副軸）

            length = 50  # 線の長さ（見た目調整用）

            pt1 = (int(center[0] - direction[0] * length), int(center[1] - direction[1] * length))
            pt2 = (int(center[0] + direction[0] * length), int(center[1] + direction[1] * length))

            # 中心から短辺方向へ線を描く（＝柱の縦線）
            cv2.line(annoted, pt1, pt2, (0, 0, 255), 2)
            vertical_lines.append([pt1, pt2])

    return vertical_lines

def make_line(img, vertical_lines):
    print("縦線の数:", len(vertical_lines))
    Y1 = []
    Y2 = []
    tops = []  # (x, y)座標を記録
    for (x1, y1), (x2, y2) in vertical_lines:
        Y1.append(y1)
        Y2.append(y2)
        # 始点（yが小さい方）を記録
        if y1 < y2:
            tops.append((x1, y1))
        else:
            tops.append((x2, y2))
    if np.sum(Y1) > np.sum(Y2):
        print("上端のY座標:", Y2)
    else:
        print("上端のY座標:", Y1)

    # x座標順にソート
    tops = sorted(tops, key=lambda p: p[0])
    print("始点座標リスト（x昇順）:", tops)

    if len(tops) >= 2:
        x0, y0 = tops[0]
        x1, y1 = tops[-1]
        # 傾き計算
        if x1 != x0:
            slope = (y1 - y0) / (x1 - x0)
        else:
            slope = 0  # 垂直の場合

        # 画像の左端と右端でのy座標を計算
        x_left = 0
        y_left = int(y0 + slope * (x_left - x0))
        x_right = img.shape[1] - 1
        y_right = int(y0 + slope * (x_right - x0))

        # 画像範囲内にクリップ
        y_left = np.clip(y_left, 0, img.shape[0] - 1)
        y_right = np.clip(y_right, 0, img.shape[0] - 1)

        # 端から端まで直線を描画
        cv2.line(img, (x_left, y_left), (x_right, y_right), (0, 0, 255), 2)
    # show_image(img)
    return img, (x_left, y_left), (x_right, y_right)
    
def calc_angle(vertical_lines):
    """
    各縦線が垂直（90度）から何度ずれているかを計算し、リストで返す
    """
    angles = []
    for (x1, y1), (x2, y2) in vertical_lines:
        dx = x2 - x1
        dy = y2 - y1
        # 線分の傾きの角度（ラジアン→度）
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        # 垂直線（90度）との差
        diff = abs(90 - abs(angle_deg))
        angles.append(diff)
        print(f"線(({x1},{y1})-({x2},{y2})) の傾き: {angle_deg:.2f}度, 垂直との差: {diff:.2f}度")
    return angles

def distances_and_bike_distances(img, centroids, angle, start, end, pixel_per_meter):
    """
    粉色線交点を計算し、さらに相邻自行车間の距離も計算
    """
    cross = []
    # 屋根直線の方程式: y = m1*x + b1
    x0, y0 = start
    x1, y1 = end
    if x1 != x0:
        m1 = (y1 - y0) / (x1 - x0)
    else:
        m1 = float('inf')  # 垂直
    b1 = y0 - m1 * x0 if m1 != float('inf') else None

    # angleは垂直からのずれ（度）なので、直線の傾きm2を計算
    theta = np.deg2rad(90 - angle)  # 画像上での角度
    m2 = np.tan(theta)

    # 各自転車について粉色線との交点を計算
    for i, centroid in enumerate(centroids):
        if centroid is None:
            continue
        
        cx, cy = centroid
        if cx is None or cy is None:
            continue

        # 直線1: 屋根直線 y = m1*x + b1
        # 直線2: y = m2*(x - cx) + cy
        if m1 == m2:
            print("平行なので交点なし")
            continue

        if m1 == float('inf'):
            # 屋根直線が垂直
            x_cross = x0
            y_cross = m2 * (x_cross - cx) + cy
        elif np.isinf(m2):
            # 自転車からの線が垂直
            x_cross = cx
            y_cross = m1 * x_cross + b1
        else:
            x_cross = (m2 * cx - m1 * x0 + y0 - cy) / (m2 - m1)
            y_cross = m1 * x_cross + b1

        # 交点を記録
        cross.append((x_cross, y_cross))
        
        # 粉色線を描画
        cv2.line(img, (int(cx), int(cy)), (int(x_cross), int(y_cross)), (255, 0, 255), 2)
        
        # 自転車と交点に標記を追加
        cv2.circle(img, (int(cx), int(cy)), 8, (0, 255, 0), -1)  # 緑色の自転車重心
        cv2.circle(img, (int(x_cross), int(y_cross)), 8, (255, 0, 255), -1)  # 粉色の交点
        cv2.putText(img, f"B{i+1}", (int(cx), int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"P{i+1}", (int(x_cross), int(y_cross) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        print(f"自転車{i+1}: ({cx},{cy}) → 屋根交点: ({int(x_cross)},{int(y_cross)})")

    # 交点間の相邻距離を計算
    if len(cross) >= 2:
        print("\n=== 相邻自転車間の距離計算（粉色線交点基準）===")
        
        # 交点をx座標順でソート
        cross_with_index = [(i, cross[i]) for i in range(len(cross))]
        cross_with_index.sort(key=lambda x: x[1][0])  # x座標でソート
        
        bike_distances = []
        
        for i in range(len(cross_with_index) - 1):
            idx1, (x1, y1) = cross_with_index[i]
            idx2, (x2, y2) = cross_with_index[i + 1]
            
            # 像素距離を計算
            pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 実際距離を計算
            real_distance = pixel_distance / abs(pixel_per_meter)
            
            bike_distances.append(real_distance)
            
            # 距離線を描画
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
            
            # 距離標記
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            distance_text = f"{real_distance:.2f}m"
            cv2.putText(img, distance_text, (mid_x, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            print(f"自転車{idx1+1}→自転車{idx2+1} (交点基準): "
                  f"像素距離={pixel_distance:.2f}px, 実際距離={real_distance:.2f}m")
        
        print(f"\n相邻自転車距離リスト: {bike_distances}")
        return img, cross, bike_distances
    else:
        print("交点が不足しているため、距離計算できません")
        return img, cross, []

def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    global PIXEL_PER_METER
    
    # 画像の読み込み
    # img = cv2.imread('sample.jpg')
    # img = cv2.imread('sample2.jpg')
    img = cv2.imread(r"C:\Users\Light6\ASE_G4\sample\sample.jpg")
    
    if img is None:
        print("画像が読み込めませんでした。パスを確認してください。")
        return
    
    print(f"画像サイズ: {img.shape}")
    
    # Pillar検出
    print("=== Pillar検出 ===")
    vertical_lines = pillar_inference(img)
    
    # 自転車検出
    print("=== 自転車検出 ===")
    results, centroids = inference(img)
    
    # マスク表示
    masked, max_y, min_y = mask(img, results[0])
    show_image(masked)
    
    print(f"検出された自転車重心: {centroids}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    item = len(vertical_lines)
    print(f"検出された縦線の数: {item}")
    
    if item > 1:
        PIXEL_PER_METER = (vertical_lines[0][0][0] - vertical_lines[-1][0][0]) / ((item-1) * PILLAR_DISTANCE)
        print(f"PIXEL_PER_METER: {PIXEL_PER_METER}")

        # Pillar線を描画
        for (x1, y1), (x2, y2) in vertical_lines:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Roof線を描画
        img, start, end = make_line(img, vertical_lines)
        
        # 角度計算
        angles = calc_angle(vertical_lines)
        angle = np.max(angles[0:2]) if len(angles) >= 2 else 0
        print(f"最大の角度差: {angle:.2f}度")
        
        # 粉色線描画 + 自転車間距離計算
        final_img, cross_points, bike_distances = distances_and_bike_distances(
            img, centroids, angle*(-1), start, end, PIXEL_PER_METER
        )
        
        # 結果表示
        show_image(final_img)
        
        # 結果保存
        cv2.imwrite("result_with_bike_distances.jpg", final_img)
        print("結果画像を保存しました: result_with_bike_distances.jpg")
        
        # 最終サマリー
        print(f"\n=== 最終結果サマリー ===")
        print(f"検出された自転車数: {len([c for c in centroids if c is not None])}")
        print(f"計算された距離数: {len(bike_distances)}")
        if bike_distances:
            print("相邻自転車間距離:")
            for i, dist in enumerate(bike_distances):
                print(f"  距離{i+1}: {dist:.2f}メートル")
    else:
        print("Pillar数が不足しています")

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()