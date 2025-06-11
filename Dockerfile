FROM ultralytics/yolov5:latest

# 依存関係のインストール（OpenCV表示用に必要）
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev

# 必要なPythonライブラリをインストール
RUN pip install --no-cache-dir grpcio grpcio-tools flask opencv-python
