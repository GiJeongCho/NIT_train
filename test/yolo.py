import os
import sys

from ultralytics import YOLO

# OBB(Oriented Bounding Box) 데이터셋으로 YOLO26-OBB 3종(n/s/m) 학습.
# 데이터셋 라벨: class x1 y1 x2 y2 x3 y3 x4 y4 (4 꼭짓점) — OBB 포맷.
# 전처리(preprocessed) OBB 데이터셋으로 학습.
DATA = r"c:\project\tracker_py\train_data\preprocessed_obb\data.yaml"

# 학습할 사전학습 가중치(로컬 test_model 폴더).
MODELS = [
    r"C:\project\NIT_train\test_model\yolo26l.pt",
]

EPOCHS = 100
IMGSZ = 640
BATCH = 16
DEVICE = 0  # Blackwell GPU (cu128). CPU 로 하려면 "cpu"
# 워커를 너무 크게 잡으면(=8) Windows spawn 특성상 워커마다 RAM을 크게 물어
# 후반 epoch 에서 시스템 RAM 이 바닥나 DataLoader 가 죽는다(ArrayMemoryError).
# RAM 부족이 재발하면 4 → 2 로 더 낮춘다.
WORKERS = 4
# 이전에 중단된 학습을 last.pt 부터 이어서 할지 여부.
#   python yolo.py resume        → 전체를 이어서
#   python yolo.py resume 1      → 인덱스 1 모델만 이어서
RESUME = False

# 학습 결과 저장 루트. 모델별로 하위 폴더(train_model/<모델명>/)에 저장된다.
PROJECT = r"c:\project\tracker_py\train_model"


def _tag(mp: str) -> str:
    return mp.replace("\\", "/").split("/")[-1].replace(".pt", "")  # 예: yolo26n-obb


def train_one(mp: str, resume: bool = False) -> None:
    """모델 1개 학습. resume=True 면 중단된 last.pt 부터 이어서 학습한다."""
    tag = _tag(mp)
    print(f"\n{'='*70}\n[train] {tag}  ({mp}){'  (resume)' if resume else ''}\n{'='*70}", flush=True)
    if resume:
        # 이어서 학습할 때는 저장된 weights/last.pt 를 로드하고 resume=True.
        last = os.path.join(PROJECT, tag, "weights", "last.pt")
        if not os.path.exists(last):
            print(f"[skip] resume 대상 없음: {last}", flush=True)
            return
        model = YOLO(last)
        model.train(resume=True)
    else:
        model = YOLO(mp)
        model.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT,
            name=tag,          # train_model/yolo26n-obb, ...-s-obb, ...-m-obb
        )
    print(f"[done] {tag}", flush=True)


def train_sequential(indices, resume: bool = False) -> None:
    """지정한 모델들을 같은 GPU 에서 '하나씩 순차' 학습."""
    for i in indices:
        train_one(MODELS[i], resume=resume)
    print("\n[all done] 순차 학습 완료.", flush=True)


# Windows 멀티프로세싱(spawn) 대응: 진입점은 반드시 이 가드 안에 둔다.
# 사용법:
#   python yolo.py            → 전체(n, s, m) 순차 학습
#   python yolo.py 0 2        → 지정 인덱스만 순차 학습(예: n 과 m 만)
#   python yolo.py resume     → 중단된 학습을 last.pt 부터 이어서(전체)
#   python yolo.py resume 1   → 인덱스 1 모델만 이어서 학습
if __name__ == "__main__":
    args = sys.argv[1:]
    resume = RESUME
    if args and args[0].lower() == "resume":
        resume = True
        args = args[1:]
    indices = [int(a) for a in args] if args else list(range(len(MODELS)))
    train_sequential(indices, resume=resume)
