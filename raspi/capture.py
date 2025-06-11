import time
import cv2
import grpc
import datetime
import capture_pb2
import capture_pb2_grpc


# please edit if you use AWS
host = '172.21.48.46'

def capture_data_generator(cap):
    msg_id = 0
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
            
            # CaptureDataメッセージを作成
            capture_data = capture_pb2.CaptureData(
                id=msg_id,
                command="request",
                image_data=image_data,
                message="Frame captured",
                timestamp=datetime.datetime.now().isoformat()
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
    channel = grpc.insecure_channel(f'{host}:50051')
    stub = capture_pb2_grpc.ImageCaptureStub(channel)
    
    # クライアントストリーミングRPCを呼び出し、サーバからの応答を受信
    response = stub.StreamData(capture_data_generator(cap))
    print("Response from server:", response)

    cap.release()

if __name__ == "__main__":
    main()