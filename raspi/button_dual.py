import time
import cv2
import grpc
import datetime
import capture_pb2
import capture_pb2_grpc
import msvcrt     # 追加

# please edit if you use AWS
# host = '172.21.48.46'
host = '54.211.126.57'

def capture_data_generator(cap0, cap1):
    msg_id = 0
    print("Press 't' to capture images, 'q' to quit.")
    while True:
        # キー入力があれば取得
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b't':
                # カメラ0キャプチャ
                ret0, frame0 = cap0.read()
                # カメラ1キャプチャ
                ret1, frame1 = cap1.read()
                if not ret0 or not ret1:
                    print("Error: Could not read frame(s).")
                    continue

                # JPEGエンコード＆メッセージ作成（camera 0）
                _, buf0 = cv2.imencode('.jpg', frame0)
                yield capture_pb2.CaptureData(
                    id=msg_id,
                    command="request",
                    image_data=buf0.tobytes(),
                    message="camera 0",
                    timestamp=datetime.datetime.now().isoformat()
                )

                # JPEGエンコード＆メッセージ作成（camera 1）
                _, buf1 = cv2.imencode('.jpg', frame1)
                yield capture_pb2.CaptureData(
                    id=msg_id + 1,
                    command="request",
                    image_data=buf1.tobytes(),
                    message="camera 1",
                    timestamp=datetime.datetime.now().isoformat()
                )

                msg_id += 2
                print(f"Sent capture #{msg_id//2}")
            elif key == b'q':
                print("Quitting capture.")
                break
        time.sleep(0.1)

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

    # キー操作に応じてキャプチャ→送信
    response = stub.StreamData(capture_data_generator(cap0, cap1))
    print("Response from server:", response)

    cap0.release()
    cap1.release()

if __name__ == "__main__":
    main()