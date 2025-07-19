import time
import os
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
# CAMERA_ID_2 = "camera2"

# img_dir1 = r"../img/raspi1"
# img_dir2 = r"../img/raspi2"

def capture_data_generator(img_dir):
    msg_id = 0
    while True:
        try:
            # jpgファイルのみ抽出
            jpg_files = [img for img in os.listdir(img_dir) if img.endswith(".jpg")]
            # ファイル名の最後の3文字の数字で昇順ソート
            jpg_files.sort(key=lambda x: int(x[-7:-4]))
            print(f"Found {len(jpg_files)} images in {img_dir}")
            for img in jpg_files:
                print(f"Processing image: {img}")
                # 最後の3文字の数字をが490未満ならスキップ
                if int(img[-7:-4]) < 490:
                    print(f"Skipping image: {img} (id < 490)")
                    continue
                with open(os.path.join(img_dir, img), 'rb') as f:
                    image_data = f.read()
                    now = datetime.datetime.now()
                    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                    yield capture_pb2.CaptureData(
                        id=0,
                        command="send",
                        image_data=image_data,
                        message=CAMERA_ID_1,
                        timestamp=timestamp_str
                    )
                    time.sleep(10)
        except Exception as e:
            print(f"Error during capture: {e}")
            break

def main():
    # 2台のカメラの初期化
    img_dir = r"../../img/"
    # gRPCチャンネルの作成
    channel = grpc.insecure_channel(f'{host}:443')
    stub = capture_pb2_grpc.ImageCaptureStub(channel)
    
    # カスタマーライベントリーミングRPCを呼び出し、サーバからの応答を受信
    response = stub.StreamData(capture_data_generator(img_dir))
    print("Response from server:", response)

if __name__ == "__main__":
    main()