# NIT_train — 학습/라벨링 API
#
# base 이미지가 torch/CUDA/opencv/ultralytics 를 제공하므로 여기서는 앱 추가 의존성만 깐다.
# (중복 설치하면 이미지가 커지고 CUDA 버전이 어긋날 수 있다.)
FROM ultralytics/ultralytics:latest

WORKDIR /app

# 의존성 레이어를 먼저 굳혀 소스 변경 시 재설치를 피한다.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY test_model /app/test_model

# 모든 상태(영상·프레임·라벨·데이터셋·학습결과)가 여기 하나에 모인다.
# 반드시 볼륨으로 마운트할 것. 안 하면 컨테이너를 지울 때 라벨링 작업이 사라진다.
ENV NIT_TRAIN_WORKSPACE=/app/workspace \
    NIT_TRAIN_MODEL_DIR=/app/test_model \
    NIT_TRAIN_HOST=0.0.0.0 \
    NIT_TRAIN_PORT=8888 \
    PYTHONUNBUFFERED=1
VOLUME ["/app/workspace"]

EXPOSE 8888

# 전처리 가중치 등 CWD 상대 경로 자원이 없더라도, 개발표준대로 앱 루트에서 실행한다.
WORKDIR /app/src/v1
CMD ["python", "main.py"]
