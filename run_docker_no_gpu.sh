#!/bin/bash

# ホスト側の相対パスを絶対パスに変換
HOST_IMG_DIR="$(pwd)/img"
HOST_DATA_DIR="$(pwd)/work"

# コンテナ内のマウント先
CONTAINER_IMG_DIR="/app/img"
CONTAINER_DATA_DIR="/app/work"

# ポート設定
GRPC_PORT=50051
WEB_PORT=5000

# Dockerイメージ名
IMAGE_NAME="yolov5-server:v2"

# コンテナ名
CONTAINER_NAME="yolo_server_container_no_gpu"

# 既存のコンテナがあれば削除
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
  echo "既存のコンテナ '${CONTAINER_NAME}' を削除しています..."
  docker rm -f "${CONTAINER_NAME}"
fi

docker run -it \
  --name "${CONTAINER_NAME}" \
  -p ${WEB_PORT}:5000 \
  -p ${GRPC_PORT}:50051 \
  -v "${HOST_IMG_DIR}:${CONTAINER_IMG_DIR}" \
  -v "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
  -w /app \
  ${IMAGE_NAME}