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

app = Flask(__name__, template_folder='templates')

# 受信した画像データを保持するグローバル変数
latest_images = {}  # カメラIDをキー、base64画像を値に
latest_times = {}
# count = 0

model = YOLO("yolov8n.pt")  # YOLOv8のモデルをロード

img_path_1 = r"../img/camera1/"
img_path_2 = r"../img/camera2/"

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

def inference(base64_img):
    # base64文字列をバイト配列に戻す
    img_data = base64.b64decode(base64_img)
    nparr = np.frombuffer(img_data, np.uint8)
    # JPEGバイト列からNumPy配列に変換（BGR形式）
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        return None
    # YOLO推論（画像の形式はBGRでOK）
    result = model(img_np, verbose=False)
    # 結果を画像に描画（annotated image）
    annotated_img = result[0].plot()
    # annotated image をJPEGにエンコードしてbase64文字列に変換
    retval, buffer = cv2.imencode('.jpg', annotated_img)
    if retval:
        return base64.b64encode(buffer).decode('utf-8')
    return None

# Flaskのルート定義

@app.route('/')
def index():
    title = "受信した画像表示"
    image1 = capture_image("camera1")
    image2 = capture_image("camera2")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # +9hする（日本時間）
    current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index_2camera.html', title=title, image1=image1, image2=image2, current_time=current_time)

@app.route('/capture_image', methods=['GET'])
def get_all_images():
    image1 = capture_image("camera1")
    image2 = capture_image("camera2")
    current_time = (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
        "camera1": {"image": image1, "current_time": current_time},
        "camera2": {"image": image2, "current_time": current_time}
    })

# 推論結果を表示するルート追加
@app.route('/inference', methods=['GET'])
def inference_image():
    results = {}
    for camera_id in ["camera1", "camera2"]:
        orig_image = capture_image(camera_id)
        if not orig_image:
            results[camera_id] = {"error": "画像がありません"}
            continue
        inf_image = inference(orig_image)
        results[camera_id] = {"image": inf_image}
    return jsonify(results)

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