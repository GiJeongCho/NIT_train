# external_datasets — 외부에서 받은 원본 데이터셋 보관소

외부(예: Roboflow)에서 받은 **원본 데이터셋(raw)** 을 프로젝트 안에서 관리하는 곳입니다.
학습에 실제로 쓰이는 변환본은 등록 시 `workspace/datasets/<id>/` 로 굳혀지므로,
여기 폴더는 **재등록·감사·백업용 원본**입니다.

## 현재 보관 (UAV / 전차 · Roboflow COCO)
| 폴더 | 형식 | 클래스 |
|---|---|---|
| nit_03_01, nit_03_02 | COCO(`_annotations.coco.json`) | Ikv_91_105 · Jagdtiger · Panther_II · Strv_101 · Strv_103A · Tiger_II_10.5_ · VIDAR |
| nit_04_01 ~ nit_04_04 | COCO | (위와 동일, 일부 6종) |
| nit_unreal_01, nit_unreal_02 | COCO | challenger1 · chunma2 · t34-85 · t55 |

## 다시 등록하려면
프론트 **＋ 데이터 → 경로 붙여넣기 → 확인 → 등록**, 또는 4·데이터셋의 "기존 폴더 등록".
경로 예: `C:\project\NIT_train\external_datasets\nit_03_01`

- COCO 폴더는 등록 시 **YOLO-OBB 로 자동 변환**되어 `workspace/datasets/` 에 복사(굳힘)됩니다.
- 이미 등록된 8종의 원본 위치(`source_dir`)는 이 폴더로 갱신돼 있습니다.
