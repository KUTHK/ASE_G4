import sys
import os
import threading
import concurrent.futures

import grpc
import cv2
import numpy as np
import base64

import torch
from ultralytics import YOLO
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify

# gRPCのプロトコルモジュールをインポート
import capture_pb2
import capture_pb2_grpc

import space as sp

app = Flask(__name__, template_folder='templates')

# 受信した画像データを保持するグローバル変数

latest_images = {
    "camera1": None,
    "camera2": None
}
latest_times = {
    "camera1": None,
    "camera2": None
}
masked_images = {
    "camera1": None,
    "camera2": None
}
processed_images = {
    "camera1": None,
    "camera2": None
}
parking_arrays = [1, -1, -1, -1, -1, -1, -1, -1, -1]
# parking_arrays = [2, -1, -1, -1, -1, -1, -1, -1, -1]

# model = YOLO("yolov8n.pt")  # YOLOv8のモデルをロード

img_path_1 = r"../img/camera1/"
img_path_2 = r"../img/camera2/"

seg_model = YOLO("yolov8n-seg.pt")  # セグメンテーションモデル
obb_model = YOLO(r"best.pt")  # OBBモデル
sd = sp.SpaceDetector(seg_model=seg_model, obb_model=obb_model)


def capture_image(camera_id):
    global latest_images
    # 受信済み画像があればそれを返す、なければ黒画像を生成
    if camera_id in latest_images:
        return latest_images[camera_id]
    # 黒い画像（480x640）を生成
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    retval, buffer = cv2.imencode('.jpg', black_img)
    if retval:
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    return None

def create_black():
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    retval, buffer = cv2.imencode('.jpg', black_img)
    if retval:
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    return None

black = create_black()
latest_images["camera1"] = black
latest_images["camera2"] = black


# Flaskのルート定義

@app.route('/')
def index():
    # image1 = create_black()
    # image2 = create_black()
    # latest_images["camera1"] = image1
    # latest_images["camera2"] = image2
    return render_template('ParkingMap2.html')

@app.route('/archive')
def archive():
    return render_template('ParkingMapArchive.html')

@app.route('/api/parking_stats')
def calculate_parking():
    # 仮の配列を直接返す
    # -1: 灰色, 0: 赤色, 1: 黄色, 2: 緑色
    # dummy_data = [2, -1, -1, -1, -1, -1, 0, -1, 1]  # 長さ9の配列
    # dummy_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 長さ9の配列
    return jsonify(parking_arrays)

@app.route('/show_images')
def show_images():
    title = "取得画像表示"
    # image1 = capture_image("camera1")
    # image2 = capture_image("camera2")
    image1 = latest_images["camera1"] if latest_images["camera1"] else create_black()
    image2 = latest_images["camera2"] if latest_images["camera2"] else create_black()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # +9hする（日本時間）
    # current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return render_template('show_2camera.html', title=title, image1=image1, image2=image2, current_time=current_time)

@app.route('/mask_images')
def mask_images():
    title = "マスク画像表示"
    image1 = masked_images["camera1"] if masked_images["camera1"] else create_black()
    image2 = masked_images["camera2"] if masked_images["camera2"] else create_black()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # +9hする（日本時間）
    # current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return render_template('mask_2camera.html', title=title, image1=image1, image2=image2, current_time=current_time)

@app.route('/result_images')
def result_images():
    title = "結果画像表示"
    image1 = processed_images["camera1"] if processed_images["camera1"] else create_black()
    image2 = processed_images["camera2"] if processed_images["camera2"] else create_black()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # +9hする（日本時間）
    # current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return render_template('result_2camera.html', title=title, image1=image1, image2=image2, current_time=current_time)

@app.route('/api/camera1_image')
def get_camera1_image():
    image = latest_images.get("camera1")
    if not image:
        image = create_black()
    return jsonify({"image": image})

@app.route('/capture_image', methods=['GET'])
def get_all_images():
    title = "受信した画像表示"
    # image1 = capture_image("camera1")
    # image2 = capture_image("camera2")
    # original, masked, processed, parking_array = process_images(image1)
    # original2, masked2, processed2, parking_array2 = process_images(image2)
    image1 = latest_images["camera1"] if latest_images["camera1"] else create_black()
    image2 = latest_images["camera2"] if latest_images["camera2"] else create_black()
    # print(f"Parking array for camera1: {parking_array}")
    # current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # return jsonify({
    #     # "camera1": {"image": image1, "current_time": current_time},
    #     # "camera2": {"image": image2, "current_time": current_time}
    #     "camera1": {"image": image1, "current_time": current_time},
    #     "camera2": {"image": image2, "current_time": current_time}
    # })
    return render_template('index_2camera.html', title=title, image1=image1, image2=image2, current_time=current_time)

def process_images(image, camera_id):
    """3つの画像（元画像、マスク画像、処理済み画像）を生成"""
    # global latest_image
    # print("process_images called")
    # if latest_image is None:
    #     print("No latest_image available")
    #     return None, None, None
    
    try:
        # base64文字列をバイト配列に戻す
        # img_data = base64.b64decode(latest_image)
        img_data = base64.b64decode(image)
        nparr = np.frombuffer(img_data, np.uint8)
        # JPEGバイト列からNumPy配列に変換（BGR形式）
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            print("Failed to decode image")
            return None, None, None
        
        print(f"Image shape: {img_np.shape}")
        ind = None
        if camera_id == "camera1":
            ind = 2
        elif camera_id == "camera2":
            ind = 3
        # SpaceDetectorで解析実行
        original_img, masked_img, processed_img, parking_array = sd.analyze(img_np.copy(), ind)
        
        print("Analysis completed, encoding images...")
        
        # 各画像をbase64エンコード
        def encode_image(img):
            if img is None:
                print("Image is None, cannot encode")
                return None
            retval, buffer = cv2.imencode('.jpg', img)
            if retval:
                return base64.b64encode(buffer).decode('utf-8')
            print("Failed to encode image")
            return None
        
        original_b64 = encode_image(original_img)
        masked_b64 = encode_image(masked_img)
        processed_b64 = encode_image(processed_img)
        
        print(f"Encoding results - Original: {'OK' if original_b64 else 'FAIL'}, Masked: {'OK' if masked_b64 else 'FAIL'}, Processed: {'OK' if processed_b64 else 'FAIL'}")
        
        return original_b64, masked_b64, processed_b64, parking_array
        
    except Exception as e:
        print(f"Error in process_images: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# gRPCサーバ実装
class ImageCaptureServicer(capture_pb2_grpc.ImageCaptureServicer):
    def StreamData(self, request_iterator, context):
        global latest_images, latest_times
        # img_pathの存在チェックと作成
        if not os.path.exists(img_path_1):
            os.makedirs(img_path_1)
        if not os.path.exists(img_path_2):
            os.makedirs(img_path_2)
        for capture_data in request_iterator:
            # 受信したJPEGバイト列をbase64エンコードして保存
            # camera_id = capture_data.camera_id
            camera_id = capture_data.message
            img_b64 = base64.b64encode(capture_data.image_data).decode('utf-8')
            latest_images[camera_id] = img_b64
            latest_times[camera_id] = capture_data.timestamp
            # ファイル名に使用できない文字（:）を-に置換してファイルパスを生成
            sanitized_time = capture_data.timestamp.replace(":", "-")
            if camera_id == "camera1":
                filename = os.path.join(img_path_1, f"{sanitized_time}_{capture_data.id}.jpg")
            else:
                filename = os.path.join(img_path_2, f"{sanitized_time}_{capture_data.id}.jpg")
            with open(filename, "wb") as f:
                f.write(capture_data.image_data)
            print(f"Saved image to {filename}")
            print(f"Received image id: {camera_id} at {capture_data.timestamp}")
            try:
                original, masked, processed, parking_array = process_images(img_b64, camera_id)
                if original is not None:
                    latest_images[camera_id] = original
                    processed_images[camera_id] = processed
                    masked_images[camera_id] = masked
                    parking_arrays = parking_array
                    print(f"Processed image for {camera_id} successfully")
                else:
                    print(f"Failed to process image for {camera_id}")
                
            except Exception as e:
                print(f"Error processing image: {e}")
                original, masked, processed, parking_array = None, None, None, None

        # ストリーム完了時、最後の画像を応答として返す
        return capture_pb2.CaptureData(
            id=capture_data.id,
            camera_id=capture_data.camera_id,
            command="ack",
            image_data=capture_data.image_data,
            message="Image received",
            timestamp=capture_data.timestamp
        )

def start_grpc_server():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    capture_pb2_grpc.add_ImageCaptureServicer_to_server(ImageCaptureServicer(), server)
    server.add_insecure_port('[::]:443')
    server.start()
    print("gRPC server started on port 443")
    # サーバをブロックせずに動作させるため無限ループ（Ctrl+Cで終了）
    server.wait_for_termination()

def main():
    # gRPCサーバを別スレッドで起動
    grpc_thread = threading.Thread(target=start_grpc_server, daemon=True)
    grpc_thread.start()
    
    print("Starting display server...")
    # Flaskサーバ起動 (use_reloader=False によりプロセスの重複防止)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    main()