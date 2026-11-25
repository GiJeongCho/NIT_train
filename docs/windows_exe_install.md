# Windows 설치 및 이동

**학습 해상도 조정:** `5. 학습 → 학습 시작 → 학습 해상도 (px)`에서 이번 학습에 쓸 입력 크기(`imgsz`)를 지정합니다. 기본값은 640이며 960, 1280 또는 다른 32의 배수를 입력할 수 있습니다. 실행 후에는 학습 run 상세의 설정에서 적용값을 확인하세요.

이 설정은 저장된 이미지나 라벨을 변경하지 않습니다. 예를 들어 640×480으로 추출한 프레임으로 1280 학습을 하면 기존 프레임을 확대합니다. 원본의 작은 객체 디테일까지 쓰려면 원본 영상을 보관하고, 같은 프레임을 더 높은 해상도로 추출해 라벨과 함께 새 데이터셋으로 구성하세요. 해상도를 높인 뒤 GPU 메모리가 부족하면 batch를 줄이세요.

다른 Windows x64 PC에는 `installer/release/NIT_train_Windows.zip` 한 파일을 복사합니다. 전체 압축을 풀고 `NIT_train_Setup.exe`에서 **설치하고 실행**을 누르세요. 옆에 있는 `runtime.zip`은 설치기가 직접 풉니다. Docker, Anaconda, pip 설치나 인터넷 연결은 필요 없습니다.

설치 위치는 `%LOCALAPPDATA%\NIT_train`입니다. 다음부터는 그 안의 `NIT_train.exe`를 열고 **시작 / 브라우저 열기**를 누릅니다. API는 8888, 웹 화면은 8890 포트를 사용합니다. 실행기가 내장 PyTorch로 CUDA 사용 가능 여부를 확인해 기본 장치를 GPU 0 또는 CPU로 지정합니다. CPU에서는 학습 시간이 길어집니다.

작업 데이터는 배포 ZIP에 포함하지 않습니다. 기존 영상, 라벨, 학습 결과도 옮기려면 서비스를 종료하고 기존 `workspace`를 설치 폴더로 복사하세요. 실행 오류 로그는 `logs/api.log`와 `logs/front.log`에 남습니다.

개발 PC에서 재빌드: NIT_train에서 `powershell -ExecutionPolicy Bypass -File installer/build_installer.ps1` 실행 후 `python installer/bundle.py`를 실행합니다. 빌드 PC에 PyInstaller와 conda-pack이 필요합니다. 원본 환경이 기본 anaconda3/envs/NIT 밖에 있으면 NIT_RUNTIME_PREFIX에 경로를 지정합니다.
