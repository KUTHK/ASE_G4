import time
import cv2
import grpc
import datetime
import capture_pb2
import capture_pb2_grpc
import os

CAMERA_ID = "camera1"

# please edit if you use AWS
host = 'localhost'  # Use 'localhost' for local testing or replace with your server's IP address
# host = '100.69.96.32'
# host = '54.211.126.57'


def capture_data_generator(cap):
    msg_id = 0
    
    # imgディレクトリの作成
    img_dir = "img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created directory: {img_dir}")
    
    while True:
        try:
            # カメラからフレームをキャプチャ
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # フレームをJPEG形式にエンコード
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()
            
            # 現在の日時を取得
            now = datetime.datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            
            # 画像をimgディレクトリに保存
            filename = f"{img_dir}/frame_{timestamp_str}_{msg_id:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved image: {filename}")
            
            # CaptureDataメッセージを作成
            capture_data = capture_pb2.CaptureData(
                id=msg_id,
                # camera_id=CAMERA_ID,
                command="request",
                image_data=image_data,
                message=CAMERA_ID,
                timestamp=now.isoformat()
            )
            
            yield capture_data
            
            msg_id += 1
            time.sleep(5)
        except Exception as e:
            print(f"Error during capture: {e}")
            break

def main():
    # カメラの初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # gRPCチャンネルの作成
    # channel = grpc.insecure_channel(f'{host}:50051')
    channel = grpc.insecure_channel(f'{host}:443')
    # channel = grpc.insecure_channel(f'{host}:50051', options=[('grpc.enable_http_proxy', 0)])
    stub = capture_pb2_grpc.ImageCaptureStub(channel)
    
    # クライアントストリーミングRPCを呼び出し、サーバからの応答を受信
    response = stub.StreamData(capture_data_generator(cap))
    print("Response from server:", response)

    cap.release()

if __name__ == "__main__":
    main()