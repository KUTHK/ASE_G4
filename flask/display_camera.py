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

import capture_pb2
import capture_pb2_grpc

import space as sp

app = Flask(__name__, template_folder='templates')

# 受信した画像データを保持するグローバル変数（camera1のみ）
latest_image = None
latest_time = None
masked_image = None
processed_image = None
parking_array = [-1, -1, -1, -1, -1, -1, -1, -1, -1]

img_path_1 = r"../img/camera1/"

seg_model = YOLO("yolov8l-seg.pt")
obb_model = YOLO(r"best.pt")
sd = sp.SpaceDetector(seg_model=seg_model, obb_model=obb_model)


def create_black():
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    retval, buffer = cv2.imencode('.jpg', black_img)
    if retval:
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    return None


black = create_black()
latest_image = black


# Flaskのルート定義

@app.route('/')
def index():
    return render_template('ParkingMapOnecamera.html')


@app.route('/archive')
def archive():
    return render_template('ParkingMapArchive.html')


@app.route('/api/parking_stats')
def calculate_parking():
    return jsonify(parking_array)


@app.route('/show_image')
def show_image():
    title = "取得画像表示"
    image = latest_image if latest_image else create_black()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('show_1camera.html', title=title, image=image, current_time=current_time)


@app.route('/mask_image')
def mask_image():
    title = "マスク画像表示"
    image = masked_image if masked_image else create_black()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('mask_1camera.html', title=title, image=image, current_time=current_time)


@app.route('/result_image')
def result_image():
    title = "結果画像表示"
    image = processed_image if processed_image else create_black()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('result_1camera.html', title=title, image=image, current_time=current_time)


@app.route('/api/camera1_image')
def get_camera1_image():
    image = latest_image
    if not image:
        image = create_black()
    return jsonify({"image": image})


@app.route('/capture_image', methods=['GET'])
def get_image():
    title = "受信した画像表示"
    image = latest_image if latest_image else create_black()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index_1camera.html', title=title, image=image, current_time=current_time)


def process_image(image, camera_id):
    """3つの画像（元画像、マスク画像、処理済み画像）を生成"""
    try:
        img_data = base64.b64decode(image)
        nparr = np.frombuffer(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            print("Failed to decode image")
            return None, None, None
        
        # cameraidのidの部分の数字を正規表現で抽出
        import re
        match = re.search(r'camera(\d+)', camera_id)
        if match:
            ind = int(match.group(1)) - 1
        else:
            print(f"Invalid camera_id format: {camera_id}")
            return None, None, None

        # SpaceDetectorで解析実行
        original_img, masked_img, processed_img, parking_arr = sd.analyze(img_np.copy(), ind)
        
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
        return original_b64, masked_b64, processed_b64, parking_arr
    except Exception as e:
        print(f"Error in process_image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


class ImageCaptureServicer(capture_pb2_grpc.ImageCaptureServicer):
    def StreamData(self, request_iterator, context):
        global latest_image, latest_time, masked_image, processed_image, parking_array
        if not os.path.exists(img_path_1):
            os.makedirs(img_path_1)
        for capture_data in request_iterator:
            camera_id = capture_data.message
            img_b64 = base64.b64encode(capture_data.image_data).decode('utf-8')
            latest_image = img_b64
            latest_time = capture_data.timestamp
            sanitized_time = capture_data.timestamp.replace(":", "-")
            filename = os.path.join(img_path_1, f"{sanitized_time}_{capture_data.id}.jpg")
            with open(filename, "wb") as f:
                f.write(capture_data.image_data)
            print(f"Saved image to {filename}")
            print(f"Received image id: {camera_id} at {capture_data.timestamp}")
            try:
                original, masked, processed, parking_arr = process_image(img_b64, camera_id)
                if original is not None:
                    latest_image = original
                    processed_image = processed
                    masked_image = masked
                    parking_array = parking_arr
                    print(f"Processed image for {camera_id} successfully")
                else:
                    print(f"Failed to process image for {camera_id}")
            except Exception as e:
                print(f"Error processing image: {e}")
                original, masked, processed, parking_arr = None, None, None, None

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
    server.wait_for_termination()


def main():
    grpc_thread = threading.Thread(target=start_grpc_server, daemon=True)
    grpc_thread.start()
    print("Starting display server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()