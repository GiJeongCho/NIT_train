"""
preprocess_vendor
=================

추론 서비스(**tracker_py**)의 프레임 전처리를 NIT_train 안으로 떼어 온(vendored) 패키지.

목적: 학습 이미지를 **추론 입력과 똑같은 방식**으로 전처리하기 위함이다. tracker_py 는
추론 직전에 `fit_input(640x480) → Stage1(저조도/안개) → Stage2(CLAHE) → Stage3(웨이블릿,
기본 OFF)` 파이프라인을 돌린다. 같은 전처리를 학습 데이터에도 적용해야 분포가 맞고
실전 성능이 나온다.

여기에는 tracker_py 가 설치/실행돼 있지 않아도 되도록 **가중치가 필요 없는** 부분을
그대로 복사했다.

- `utils_image` : float/uint8 변환 헬퍼 (원본 preprocess/utils_image.py)
- `clahe_lab`   : Stage2 화질향상 CLAHE(LAB L 채널) — cv2 (원본 modules/quality/clahe_lab.py)
- `dcp_dehaze`  : Stage1 안개 제거 DCP — cv2/numpy. 추론 기본(GPU DCP)과 **같은 알고리즘**을
                  CPU 로 돈다(원본 modules/fog/dcp_dehaze.py). GPU 유무와 무관하게 결과 동일.
- `classify`    : auto 모드 다중지표(밝기·대비·채도·선명도) dark/fog 판정
                  (원본 preprocess/pipeline.py 의 classify_conditions 이식)
- `zero_dce`    : Stage1 저조도 보정 Zero-DCE++ (torch, **가중치 필요**). 가중치 파일이
                  `zero_dce_weights/Epoch99.pth` 에 있으면 추론과 동일하게 사용하고,
                  없으면 상위(services.preprocess)가 가중치 불필요한 대체(감마+CLAHE)로 폴백한다.

원본 경로: c:/project/tracker_py/src/v1/preprocess/
"""
