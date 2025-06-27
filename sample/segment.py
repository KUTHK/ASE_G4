from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-seg.pt")  # YOLOv8のモデルをロード

def main():
    img1 = cv2.imread(r"1000015655.jpg")
    img2 = cv2.imread(r"1000015657.jpg") 
    
    if img1 is None or img2 is None:
        print("Error: いずれかの画像が読み込めませんでした")
        return

    # 推論実施
    results1 = model(img1)
    results2 = model(img2)

    # 推論結果を描画した画像を取得
    annotated1 = results1[0].plot()  # 最初の画像の結果を描画
    annotated2 = results2[0].plot()  # 次の画像の結果を描画
    
    cv2.imwrite("annotated1.jpg", annotated1)  # 結果をファイルに保存
    cv2.imwrite("annotated2.jpg", annotated2)  # 結果をファイルに保存

    # # annotated image をウィンドウで表示
    # cv2.imshow("Inference Result 1", annotated1)
    # cv2.imshow("Inference Result 2", annotated2)

    # # キー入力待ち（何か押すと終了）
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()