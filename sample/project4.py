import math
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from sklearn.decomposition import PCA

model = YOLO(r"../flask/yolov8n-seg.pt")
obb_model = YOLO(r"../flask/best.pt")

PILLAR_DISTANCE = 3.5 # m
PIXEL_PER_METER_ARRAY = None
PILLAR_X_COORDS = None

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

def calculate_pillar_roof_intersections(vertical_lines, start, end):
    """
    各柱と屋根のラインの交点を計算し、ピクセルとメートルの対応関係を確立します。
    """
    global PIXEL_PER_METER_ARRAY, PILLAR_X_COORDS
    
    if len(vertical_lines) < 2:
        print("Pillar数が不足しています（2個以上必要）")
        return None, None
    
    # roof線の方程式: y = m*x + b
    x0, y0 = start
    x1, y1 = end
    if x1 != x0:
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
    else:
        print("Roof線が垂直なため計算できません")
        return None, None
    
    # 各pillarとroof線の交点を計算
    pillar_roof_points = []
    for i, ((px1, py1), (px2, py2)) in enumerate(vertical_lines):
        # pillar線の中点のx座標を使用
        pillar_x = (px1 + px2) / 2
        # roof線上でのy座標を計算
        roof_y = m * pillar_x + b
        pillar_roof_points.append((pillar_x, roof_y))
        print(f"Pillar {i+1} とroof線の交点: ({pillar_x:.1f}, {roof_y:.1f})")
    
    # x座標順にソート
    pillar_roof_points.sort(key=lambda p: p[0])
    pillar_x_coords = [p[0] for p in pillar_roof_points]
    
    # 相邻pillar间的像素距离を計算
    pixel_distances = []
    for i in range(len(pillar_x_coords) - 1):
        pixel_dist = abs(pillar_x_coords[i+1] - pillar_x_coords[i])
        pixel_distances.append(pixel_dist)
    
    # 各区间的pixel_per_meter比例を計算
    pixel_per_meter_array = []
    for pixel_dist in pixel_distances:
        ppm = pixel_dist / PILLAR_DISTANCE
        pixel_per_meter_array.append(ppm)
    
    print(f"\nPillar roof線交点のx座標: {[f'{x:.1f}' for x in pillar_x_coords]}")
    print(f"相邻pillar间像素距離: {[f'{d:.1f}' for d in pixel_distances]}")
    print(f"各区間のpixel_per_meter: {[f'{ppm:.2f}' for ppm in pixel_per_meter_array]}")
    
    PIXEL_PER_METER_ARRAY = pixel_per_meter_array
    PILLAR_X_COORDS = pillar_x_coords
    
    return pixel_per_meter_array, pillar_x_coords

def interpolate_pixel_per_meter(x_position):
    """
    x座標位置を補間して、その位置のpixel_per_meter値を取得します
    """
    global PIXEL_PER_METER_ARRAY, PILLAR_X_COORDS
    
    if PILLAR_X_COORDS is None or PIXEL_PER_METER_ARRAY is None:
        return None
    
    if len(PILLAR_X_COORDS) < 2 or len(PIXEL_PER_METER_ARRAY) == 0:
        return None
    
    # x_positionが最初の柱の前の場合は、最初の間隔の値を使用します
    if x_position <= PILLAR_X_COORDS[0]:
        return PIXEL_PER_METER_ARRAY[0]
    
    # x_positionが最後の柱の後にある場合は、最後の間隔の値を使用します
    if x_position >= PILLAR_X_COORDS[-1]:
        return PIXEL_PER_METER_ARRAY[-1]

    # x_positionが属する区間を見つける
    for i in range(len(PILLAR_X_COORDS) - 1):
        if PILLAR_X_COORDS[i] <= x_position <= PILLAR_X_COORDS[i+1]:
            # その区間のpixel_per_meter値を使用
            if i < len(PIXEL_PER_METER_ARRAY):
                return PIXEL_PER_METER_ARRAY[i]
            else:
                return PIXEL_PER_METER_ARRAY[-1]
    
    return PIXEL_PER_METER_ARRAY[-1]

def distances_and_bike_distances_improved(img, centroids, angle, start, end):
    """
    遠近法の補正に基づく距離計算の改善 - ここでは距離ラベルを描画しません
    """
    cross = []
    distance_info = []  # 後続の描画用に詳細情報を保存

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

    # 交点間の相邻距離を計算（優化版）
    if len(cross) >= 2:
        print("\n=== 相邻自転車間の距離計算（優化版）===")
        
        # 交点をx座標順でソート
        cross_with_index = [(i, cross[i]) for i in range(len(cross))]
        cross_with_index.sort(key=lambda x: x[1][0])  # x座標でソート
        
        bike_distances = []
        
        print("使用方法: 水平距離のみで計算")
        
        for i in range(len(cross_with_index) - 1):
            idx1, (x1, y1) = cross_with_index[i]
            idx2, (x2, y2) = cross_with_index[i + 1]
            
            # 水平距離のみ（y成分を無視）
            horizontal_pixel_distance = abs(x2 - x1)
            
            # 两个交点的pixel_per_meter值を取得
            ppm1 = interpolate_pixel_per_meter(x1)
            ppm2 = interpolate_pixel_per_meter(x2)
            
            if ppm1 and ppm2:
                # 平均値を変換率として使用する
                avg_ppm = (ppm1 + ppm2) / 2
                real_distance = horizontal_pixel_distance / avg_ppm
                
                # 採用結果
                bike_distances.append(real_distance)
                
                # 距離情報を保存（後で绘制用）
                distance_info.append({
                    'point1': (x1, y1),
                    'point2': (x2, y1),  # 水平線なのでy1を使用
                    'bike1_idx': idx1,
                    'bike2_idx': idx2,
                    'horizontal_pixel_distance': horizontal_pixel_distance,
                    'avg_ppm': avg_ppm,
                    'original_distance': real_distance
                })
                
                print(f"自転車{idx1+1}→自転車{idx2+1}:")
                print(f"  水平像素距離: {horizontal_pixel_distance:.2f}px")
                print(f"  平均PPM: {avg_ppm:.2f}")
                print(f"  実際距離: {real_distance:.2f}m")
            else:
                print(f"自転車{idx1+1}→自転車{idx2+1}: pixel_per_meter取得失敗")
                bike_distances.append(None)
                distance_info.append(None)

        # None値をフィルタリング
        valid_distances = [d for d in bike_distances if d is not None]
        print(f"\n相邻自転車距離リスト: {valid_distances}")
        return img, cross, valid_distances, distance_info
    else:
        print("交点が不足しているため、距離計算できません")
        return img, cross, [], []

def draw_distances_on_image(img, distance_info, distances, is_corrected=False):
    """
    指定された距離で距離線と標注を描画
    """
    result_img = img.copy()
    
    for i, (info, distance) in enumerate(zip(distance_info, distances)):
        if info is None or distance is None:
            continue
            
        x1, y1 = info['point1']
        x2, y2 = info['point2']
        
        # 距離線を描画（水平線）
        cv2.line(result_img, (int(x1), int(y1)), (int(x2), int(y1)), (0, 255, 255), 3)
        
        # 距離標記
        mid_x = int((x1 + x2) / 2)
        mid_y = int(y1)
        distance_text = f"{distance:.2f}m"
        
        if is_corrected:
            # 校正済みの場合は背景付きで描画
            text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(result_img, 
                         (mid_x - text_size[0]//2 - 5, mid_y - text_size[1] - 15),
                         (mid_x + text_size[0]//2 + 5, mid_y - 5),
                         (0, 0, 0), -1)  # 黒い背景
            
            cv2.putText(result_img, distance_text, (mid_x - text_size[0]//2, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 校正済みマークを追加（日本語の代わりに英語を使用）
            cv2.putText(result_img, "Corrected", (mid_x - 40, mid_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # 通常の描画
            cv2.putText(result_img, distance_text, (mid_x, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return result_img

def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    global PIXEL_PER_METER_ARRAY, PILLAR_X_COORDS
    
    # 画像の読み込み
    # img = cv2.imread(r"C:\Users\Light6\ASE_G4\sample\output.jpg")
    img = cv2.imread(r"output.jpg")
    
    if img is None:
        print("画像が読み込めませんでした。パスを確認してください。")
        return
    
    print(f"画像サイズ: {img.shape}")
    print(f"Pillar間距離: {PILLAR_DISTANCE}m (固定)")
    
    # Pillar検出
    print("\n=== Pillar検出 ===")
    vertical_lines = pillar_inference(img)
    
    # 自転車検出
    print("\n=== 自転車検出 ===")
    results, centroids = inference(img)
    
    # マスク表示
    masked, max_y, min_y = mask(img, results[0])
    show_image(masked)
    
    print(f"検出された自転車重心: {centroids}")

    item = len(vertical_lines)
    print(f"検出された縦線の数: {item}")
    
    if item > 1:
        # 基本要素を描画（Pillar線、Roof線）
        base_img = img.copy()
        for (x1, y1), (x2, y2) in vertical_lines:
            cv2.line(base_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        base_img, start, end = make_line(base_img, vertical_lines)
        
        # Pillarとroof線の交点を計算し、透視校正用の配列を作成
        print(f"\n=== 透視校正計算 ===")
        pixel_per_meter_array, pillar_x_coords = calculate_pillar_roof_intersections(vertical_lines, start, end)
        
        if pixel_per_meter_array is not None:
            # 角度計算
            angles = calc_angle(vertical_lines)
            angle = np.max(angles[0:2]) if len(angles) >= 2 else 0
            print(f"最大の角度差: {angle:.2f}度")
            
            # 粉色線描画と距離計算（距離線は描画しない）
            detection_img, cross_points, bike_distances, distance_info = distances_and_bike_distances_improved(
                base_img.copy(), centroids, angle*(-1), start, end
            )
            
            if bike_distances:
                print(f"\n=== 結果評価 ===")
                avg_distance = np.mean(bike_distances)
                print(f"平均検出距離: {avg_distance:.2f}m")
                
                # キャリブレーションオプション
                apply_correction = input(f"\n校正係数を適用しますか？ (y/n, デフォルト:n): ").strip().lower()
                
                if apply_correction == 'y':
                    try:
                        print("\n校正係数の参考値:")
                        print("  0.5-0.6: 大幅な校正（距離が実際の約2倍の場合）")
                        print("  0.7-0.8: 中程度の校正（距離が実際の約1.5倍の場合）")
                        print("  0.85-0.9: 軽微な校正（距離が実際の約1.2倍の場合）")
                        
                        correction_factor = float(input("校正係数を入力してください (0.5-1.0): "))
                        
                        if correction_factor < 0.5 or correction_factor > 1.0:
                            print("警告: 校正係数は0.5-1.0の範囲で入力してください")
                            correction_factor = max(0.5, min(1.0, correction_factor))
                            print(f"自動調整: {correction_factor}")
                        
                        corrected_distances = [d * correction_factor for d in bike_distances]
                        
                        print(f"\n=== 校正後の結果 ===")
                        print(f"使用した校正係数: {correction_factor}")
                        print("校正前 → 校正後:")
                        for i, (orig, corr) in enumerate(zip(bike_distances, corrected_distances)):
                            print(f"  距離{i+1}: {orig:.2f}m → {corr:.2f}m")
                        
                        # 校正後の距離で画像を描画
                        final_img = draw_distances_on_image(detection_img, distance_info, corrected_distances, is_corrected=True)
                        final_distances = corrected_distances
                        
                        # 校正済みファイル名で保存
                        cv2.imwrite("result_corrected_distances.jpg", final_img)
                        print("校正後結果画像を保存しました: result_corrected_distances.jpg")
                        
                    except ValueError:
                        print("無効な入力です。校正をスキップします。")
                        # 原始距离で画像を描画
                        final_img = draw_distances_on_image(detection_img, distance_info, bike_distances, is_corrected=False)
                        final_distances = bike_distances
                        cv2.imwrite("result_original_distances.jpg", final_img)
                        print("元の結果画像を保存しました: result_original_distances.jpg")
                else:
                    # 原始距离で画像を描画
                    final_img = draw_distances_on_image(detection_img, distance_info, bike_distances, is_corrected=False)
                    final_distances = bike_distances
                    cv2.imwrite("result_original_distances.jpg", final_img)
                    print("元の結果画像を保存しました: result_original_distances.jpg")
                
                # 結果表示
                show_image(final_img)
                
                # 最終サマリー
                print(f"\n=== 最終結果サマリー ===")
                print(f"検出された自転車数: {len([c for c in centroids if c is not None])}")
                print(f"計算された距離数: {len(final_distances)}")
                
                if final_distances:
                    print("相邻自転車間距離:")
                    for i, dist in enumerate(final_distances):
                        print(f"  距離{i+1}: {dist:.2f}メートル")
                    
                    # 間隔値を昇順で出力します
                    print(f"\n=== 間距値昇順出力 ===")
                    sorted_distances = sorted(final_distances)
                    for i, dist in enumerate(sorted_distances):
                        if i == 0:
                            print(f"最小間距: {dist:.2f}m")
                        elif i == len(sorted_distances) - 1:
                            print(f"最大間距: {dist:.2f}m")
                        else:
                            print(f"第{i+1}小間距: {dist:.2f}m")

                    # 統計情報
                    print(f"\n=== 統計情報 ===")
                    print(f"平均間距: {np.mean(final_distances):.2f}m")
                    print(f"間距範囲: {min(final_distances):.2f}m - {max(final_distances):.2f}m")
                    print(f"標準偏差: {np.std(final_distances):.2f}m")
            else:
                print("距離計算結果がありません")
        else:
            print("透視校正計算が失敗しました")
    else:
        print("Pillar数が不足しています")

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()