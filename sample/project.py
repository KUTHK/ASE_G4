import math
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from sklearn.decomposition import PCA
from numpy.polynomial import Polynomial

model = YOLO(r"../flask/yolov8n-seg.pt")
obb_model = YOLO(r"../flask/best.pt")

PILLAR_DISTANCE = 2.8 # m
PIXEL_PER_METER = None
PILLER_X_COORDS = []

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
            # 重心も描画（任意）
            # cx, cy = np.mean(pts, axis=0).astype(int)
            # cv2.circle(annoted, (cx, cy), 5, (255, 0, 0), -1)
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
    # cv2.imwrite("result.jpg", img)
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

def distances(img, centroids, angle, start, end):
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

    for i, (cx, cy) in enumerate(centroids):
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

        # 線を描画
        cross.append((x_cross, y_cross))
        cv2.line(img, (int(cx), int(cy)), (int(x_cross), int(y_cross)), (255, 0, 255), 2)
        print(f"自転車{i+1}: ({cx},{cy}) → 屋根交点: ({int(x_cross)},{int(y_cross)})")

    show_image(img)
    cv2.imwrite("result.jpg", img)
    return img, cross


def calculate_pillar_roof_intersections(vertical_lines, start, end):

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
    # PIXEL_PER_METER_ARRAY = [ppm * 0.01 for ppm in pixel_per_meter_array] 
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

def world_distance(x1, x2):
    global PIXEL_PER_METER_ARRAY, PILLAR_X_COORDS

    if PILLAR_X_COORDS is None or PIXEL_PER_METER_ARRAY is None:
        print("Pillar情報が初期化されていません")
        return None
    
    if len(PILLAR_X_COORDS) < 2 or len(PIXEL_PER_METER_ARRAY) == 0:
        print("Pillar情報が不足しています")
        return None
    
    # x座標を昇順にソート
    start_x = min(x1, x2)
    end_x = max(x1, x2)
    
    # ピクセル距離を計算
    pixel_distance = end_x - start_x
    
    # 2つの座標が同じ区間にある場合
    for i in range(len(PILLAR_X_COORDS) - 1):
        if PILLAR_X_COORDS[i] <= start_x <= PILLAR_X_COORDS[i+1] and \
           PILLAR_X_COORDS[i] <= end_x <= PILLAR_X_COORDS[i+1]:
            # 同じ区間なので、その区間のpixel_per_meter値を使用
            if i < len(PIXEL_PER_METER_ARRAY):
                ppm = PIXEL_PER_METER_ARRAY[i]
                real_distance = pixel_distance / ppm
                print(f"区間 {i+1} での距離計算: {pixel_distance:.1f}px / {ppm:.2f} = {real_distance:.2f}m")
                return real_distance
    
    # 複数の区間にまたがる場合
    total_real_distance = 0.0
    current_x = start_x
    
    print(f"複数区間にまたがる計算: {start_x:.1f} から {end_x:.1f}")
    
    for i in range(len(PILLAR_X_COORDS) - 1):
        # 現在の区間の範囲
        section_start = PILLAR_X_COORDS[i]
        section_end = PILLAR_X_COORDS[i+1]
        
        # この区間に重複があるかチェック
        overlap_start = max(current_x, section_start)
        overlap_end = min(end_x, section_end)
        
        if overlap_start < overlap_end:
            # 重複部分のピクセル距離
            section_pixel_distance = overlap_end - overlap_start
            
            # この区間のpixel_per_meter値
            if i < len(PIXEL_PER_METER_ARRAY):
                ppm = PIXEL_PER_METER_ARRAY[i]
                section_real_distance = section_pixel_distance / ppm
                total_real_distance += section_real_distance
                
                print(f"区間 {i+1} ({section_start:.1f}-{section_end:.1f}): "
                      f"{section_pixel_distance:.1f}px / {ppm:.2f} = {section_real_distance:.2f}m")
            
            current_x = overlap_end
            
            # 終点に達した場合は終了
            if current_x >= end_x:
                break
    
    print(f"総実世界距離: {total_real_distance:.2f}m")
    return total_real_distance

def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 画像の読み込み
    img = cv2.imread('sample.jpg')
    # img = cv2.imread('sample2.jpg')
    # img = cv2.imread(r"C:\Users\ryoma\修士科目\ASE\images\img0\2025-07-05T06-35-14.214929_336.jpg")
    vertical_lines = pillar_inference(img)
    # vertical_lines = pillar_inference_pca(img)
    results, centroids = inference(img)
    masked, max_y, min_y = mask(img, results[0])
    show_image(masked)
    print(centroids)
    # res = resize(img, 640)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # vertical_lines, horizontal_lines = detect_lines(gray)

    item = len(vertical_lines)
    print(f"検出された縦線の数: {item}")
    print(vertical_lines)
    # x_coords = []
    
    PIXEL_PER_METER = (vertical_lines[0][0][0] - vertical_lines[-1][0][0]) / ((item-1) * PILLAR_DISTANCE)
    print(f"PIXEL_PER_METER: {PIXEL_PER_METER}")

    # 検出結果の描画
    for (x1, y1), (x2, y2) in vertical_lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # for (x1, y1), (x2, y2) in horizontal_lines:
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # project_front(img, vertical_lines, horizontal_lines)
    img, start, end = make_line(img, vertical_lines)
    result1, result2 = calculate_pillar_roof_intersections(vertical_lines, start, end)
    print(f"pixel_per_meter_array: {result1}")
    print(f"pillar_x_coords: {result2}")

    real_pillar = [i* PILLAR_DISTANCE for i in range(len(result2))]
    p = Polynomial.fit(result2, real_pillar, deg=2)
    print(f"Polynomial coefficients: {p.coef}")

    angles = calc_angle(vertical_lines)
    # print("各縦線の垂直からの角度差:", angles)
    angle = np.max(angles[0:2])
    print(f"最大の角度差: {angle:.2f}度")
    img, cross = distances(img, centroids, angle*(-1), start, end)
    cross = sorted(cross, key=lambda p: p[0])  # x座標順にソート
    print(cross)
    for i in range(len(cross)-1):
        x1, y1 = cross[i]
        x2, y2 = cross[i+1]
        # 交点間の距離を計算
        # dist = world_distance(x1, x2)
        dist = p(x2) - p(x1)
        if dist is not None:
            print(f"distance between bicycle {i+1} and bicycle {i+2}: {dist:.2f} m")
        else:
            print(f"distance between bicycle {i+1} and bicycle {i+2}: failed")


if __name__ == "__main__":
    main()
    
    cv2.destroyAllWindows()