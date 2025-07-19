import time
import cv2
import grpc
import datetime
import capture_pb2
import capture_pb2_grpc

# please edit if you use AWS
# host = '172.21.48.46'
# host = '54.211.126.57'
host = '100.69.96.32'

CAMERA_ID_1 = "camera1"
CAMERA_ID_2 = "camera2"

def capture_data_generator(cap0, cap1):
    msg_id = 0
    while True:
        try:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret0:
                print("Error: Could not read frame from camera 0.")
                break
            if not ret1:
                print("Error: Could not read frame from camera 1.")
                break
            
            # カメラ0のフレームをJPEG形式にエンコード
            _, buffer0 = cv2.imencode('.jpg', frame0)
            image_data0 = buffer0.tobytes()

            # 現在の日時を取得
            now = datetime.datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")

            # 画像をimgディレクトリに保存
            filename = f"{img_dir}/frame_{timestamp_str}_{msg_id:04d}_{CAMERA_ID_1}.jpg"
            cv2.imwrite(filename, frame0)
            print(f"Saved image: {filename}")
            
            # CaptureDataメッセージを作成（カメラ0用）
            capture0 = capture_pb2.CaptureData(
                id=msg_id,
                command="request",
                image_data=image_data0,
                message=CAMERA_ID_1,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            # カメラ1のフレームをJPEG形式にエンコード
            _, buffer1 = cv2.imencode('.jpg', frame1)
            image_data1 = buffer1.tobytes()

            # 現在の日時を取得
            now = datetime.datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")

            # 画像をimgディレクトリに保存
            filename = f"{img_dir}/frame_{timestamp_str}_{msg_id:04d}_{CAMERA_ID_2}.jpg"
            cv2.imwrite(filename, frame1)
            print(f"Saved image: {filename}")

            # CaptureDataメッセージを作成（カメラ1用）
            capture1 = capture_pb2.CaptureData(
                id=msg_id + 1,
                command="request",
                image_data=image_data1,
                message=CAMERA_ID_2,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            # 2つのメッセージを順次送信
            yield capture0
            yield capture1
            
            msg_id += 2
            time.sleep(5)
        except Exception as e:
            print(f"Error during capture: {e}")
            break

def main():
    # 2台のカメラの初期化
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open video.")
        return
    
    # gRPCチャンネルの作成
    channel = grpc.insecure_channel(f'{host}:443')
    stub = capture_pb2_grpc.ImageCaptureStub(channel)
    
    # カスタマーライベントリーミングRPCを呼び出し、サーバからの応答を受信
    response = stub.StreamData(capture_data_generator(cap0, cap1))
    print("Response from server:", response)

    cap0.release()
    cap1.release()

if __name__ == "__main__":
    main()