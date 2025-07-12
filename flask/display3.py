import sys
import os
import threading
import concurrent.futures

import grpc
import cv2
import numpy as np
import base64

from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify

# gRPCのプロトコルモジュールをインポート
import capture_pb2
import capture_pb2_grpc

import space as sp
from ultralytics import YOLO

app = Flask(__name__, template_folder='templates')

# 受信した画像データを保持するグローバル変数
latest_image = None
latest_time = None

img_path = r"../img/"

seg_model = YOLO("yolov8l-seg.pt")  # YOLOv8のモデルをロード
obb_model = YOLO(r"best_l.pt")
sd = sp.SpaceDetector(seg_model=seg_model, obb_model=obb_model)

def capture_image():
    global latest_image
    # 受信済み画像があればそれを返す、なければ黒画像を生成
    if latest_image is not None:
        return latest_image
    # 黒い画像（480x640）を生成
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    retval, buffer = cv2.imencode('.jpg', black_img)
    if retval:
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    return None

# Flaskのルート定義

@app.route('/')
def index():
    title = "受信した画像表示"
    image = capture_image()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # +9hする（日本時間）
    current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index.html', title=title, image=image, current_time=current_time)

@app.route('/capture_image', methods=['GET'])
def get_new_image():
    # gRPCで受信した画像（なければ黒画像）と現在時間を返す
    image = capture_image()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"image": image, "current_time": current_time})

def process_images():
    """3つの画像（元画像、マスク画像、処理済み画像）を生成"""
    global latest_image
    if latest_image is None:
        return None, None, None
    
    # base64文字列をバイト配列に戻す
    img_data = base64.b64decode(latest_image)
    nparr = np.frombuffer(img_data, np.uint8)
    # JPEGバイト列からNumPy配列に変換（BGR形式）
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        return None, None, None
    
    # SpaceDetectorで解析実行
    original_img, masked_img, processed_img = sd.analyze(img_np.copy())
    
    # 各画像をbase64エンコード
    def encode_image(img):
        retval, buffer = cv2.imencode('.jpg', img)
        if retval:
            return base64.b64encode(buffer).decode('utf-8')
        return None
    
    original_b64 = encode_image(original_img)
    masked_b64 = encode_image(masked_img)
    processed_b64 = encode_image(processed_img)
    
    return original_b64, masked_b64, processed_b64

@app.route('/process_images', methods=['GET'])
def get_processed_images():
    """3つの処理済み画像を返す"""
    original_img, masked_img, processed_img = process_images()
    if original_img is None:
        return jsonify({"error": "画像がありません"}), 404
    
    return jsonify({
        "original": original_img,
        "masked": masked_img,
        "processed": processed_img
    })

@app.route('/three_images')
def three_images():
    """3つの画像を表示するページ"""
    current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return render_template('three_images.html', 
                         original_image=None, 
                         masked_image=None, 
                         processed_image=None, 
                         current_time=current_time)

# gRPCサーバ実装
class ImageCaptureServicer(capture_pb2_grpc.ImageCaptureServicer):
    def StreamData(self, request_iterator, context):
        global latest_image, latest_time
        # img_pathの存在チェックと作成
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        for capture_data in request_iterator:
            # 受信したJPEGバイト列をbase64エンコードして保存
            latest_image = base64.b64encode(capture_data.image_data).decode('utf-8')
            latest_time = capture_data.timestamp
            # ファイル名に使用できない文字（:）を-に置換してファイルパスを生成
            sanitized_time = capture_data.timestamp.replace(":", "-")
            filename = os.path.join(img_path, f"{sanitized_time}_{capture_data.id}.jpg")
            # count = count + 1
            # if count % 5 == 0:
            #     with open(filename, "wb") as f:
            #         f.write(capture_data.image_data)
            #     print(f"Saved image to {filename}")
            with open(filename, "wb") as f:
                f.write(capture_data.image_data)
            print(f"Saved image to {filename}")
            print(f"Received image id: {capture_data.id} at {capture_data.timestamp}")
        # ストリーム完了時、最後の画像を応答として返す
        return capture_pb2.CaptureData(
            id=capture_data.id,
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