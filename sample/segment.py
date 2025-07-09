from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolov8n-seg.pt")
# model = YOLO("yolov8m-seg.pt")

def scan_area(image, target_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    regions = []
    in_region = False
    region_start = None

    # 各列を左から右にスキャン
    for col in range(width):
        # 該当列の全画素が黒かどうかチェック
        if np.all(gray[:, col] == 0):
            if not in_region:
                in_region = True
                region_start = col
        else:
            if in_region:
                regions.append((region_start, col - 1))
                in_region = False
    if in_region:
        regions.append((region_start, width - 1))

    if len(regions) == 0:
        print("連続した黒の領域が見つかりませんでした。")
        return None, None, None

    # 複数の領域があれば、最も幅の広い領域を抽出
    best_region = max(regions, key=lambda r: r[1] - r[0])
    start, end = best_region
    # 該当領域を画像から抽出（全高さを含む）
    area = image[:, start:end+1]
    return area, start, end

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

def main():
    img1 = cv2.imread(r"1000015655.jpg")
    img2 = cv2.imread(r"1000015657.jpg") 
    
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return

    # target resolution for inputting to the model
    target_size = 640
    resize_img1 = resize(img1, target_size)

    results1 = model(resize_img1)
    
    # results2 = model(img2)

    annotated1 = results1[0].plot()
    # annotated2 = results2[0].plot()

    mask1, max_y1, min_y1 = mask(resize_img1, results1[0])
    
    # make two red lines on max_y and min_y in the masked image
    cv2.line(mask1, (0, max_y1), (mask1.shape[1], max_y1), (0, 0, 255), 2)
    cv2.line(mask1, (0, min_y1), (mask1.shape[1], min_y1), (0, 0, 255), 2)

    cv2.imwrite("masked_image1.jpg", mask1)
    
    roi = mask1[min_y1:max_y1, :]
    # cv2.imwrite("roi1.jpg", roi)
    area, start, end = scan_area(roi, target_size)
    # print(f"Scanned area: start={start}, end={end}, width={end - start + 1}")
    # print(f"Scanned area shape: {area.shape}")
    
    
    # cv2.imwrite("annotated1.jpg", annotated1)
    # cv2.imwrite("annotated2.jpg", annotated2)

    # cv2.imshow("Inference Result 1", annotated1)
    # cv2.imshow("Inference Result 2", annotated2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()