# SFA3D + SimpleTrack C++ LiDAR Detection and Tracking

이 저장소는 SFA3D 기반 LiDAR 3D object detection과 SimpleTrack 기반 multi-object tracking을 하나의 C++ 파이프라인으로 묶은 프로젝트입니다.

- Detection:
  원래 PyTorch 추론 경로를 배포 지향 구조로 바꿨습니다. 현재 저장소에는 `HailoRT + HEF` 실행 경로와 `ONNX Runtime` 기준 레퍼런스 실행기가 같이 들어 있습니다.
- Tracking:
  원래 Python 기반이던 SimpleTrack 로직을 C++17로 다시 구현했습니다.
- Deployment target:
  임베디드 보드에서 돌릴 수 있도록 BEV 전처리, 후처리, tracker 연결을 C++ 쪽으로 정리해 둔 구조입니다. AGX Orin 같은 환경에서는 동일한 전처리/후처리/트래킹 흐름에 ONNX/TensorRT 추론기를 연결해서 사용할 수 있습니다.

## Overview

- 입력: `float32 x 4` 포맷의 LiDAR `.bin` 파일 시퀀스 (`x, y, z, intensity`)
- 클래스 순서: `0 = Pedestrian`, `1 = Car`, `2 = Cyclist`
- BEV 범위:
  - `x: [-25, 75] m`
  - `y: [-50, 50] m`
  - `z: [-4.1, 1.4] m`
- BEV 해상도: `1216 x 1216 x 3`
- 출력:
  - 프레임별 timing log
  - `result.png` (매 프레임 마지막 저장본으로 덮어씀)
  - `result.mp4` 또는 지정한 비디오 파일

## Demo

- MP4 demo file: [assets/hailo_tracking_demo.mp4](/home/a/sfa3d_adcm/assets/hailo_tracking_demo.mp4)
- GitHub README에서 안정적으로 인라인 재생하려면 이 파일을 Issue/Discussion에 첨부해서 생성된 `github.com/user-attachments/...` URL로 바꿔 넣는 방식을 권장합니다.

## Pipeline

1. `.bin` LiDAR 프레임을 로드하고 파일명 기준으로 정렬합니다.
2. ROI 밖의 포인트를 제거하고 z를 boundary 기준으로 정규화합니다.
3. `1216 x 1216 x 3` INT8 BEV feature map을 생성합니다.
4. Detection head를 추론합니다.
   - 루트 앱: `sfa.hef`를 HailoRT로 실행
   - 레퍼런스 앱: `sfa.onnx`를 ONNX Runtime으로 실행
5. SFA3D head를 fusion/decode 하고 클래스별 NMS를 적용합니다.
6. Detection 결과를 `FrameData`로 바꾼 뒤 `Car / Cyclist / Pedestrian` tracker를 각각 독립적으로 돌립니다.
7. BEV 위에 detection/tracking 결과를 그려 video/image로 저장합니다.

## Repository Layout

- `src/main.cpp`
  루트 실행기. LiDAR 로딩, BEV 생성, Hailo 추론, 후처리, SimpleTrack 연동, 시각화까지 전체 파이프라인을 담당합니다.
- `src/hailo_inference.cpp`, `include/hailo_inference.h`
  HailoRT wrapper, multi-head fusion, decode helper가 들어 있습니다.
- `SimpleTrack/`
  SimpleTrack C++ 포팅 코드, 설정 파일, ONNX Runtime 레퍼런스 실행기입니다.
- `SimpleTrack/configs/tracker_params.yaml`
  클래스별 tracking 파라미터를 정의합니다.
- `SimpleTrack/configs/waymo_configs/*.yaml`
  tracker base config입니다.
- `SimpleTrack/tools/inference_sfa3d_at128_onnx.cpp`
  ONNX Runtime 기반 레퍼런스 추론 + tracking 실행기입니다.
- `sfa.hef`
  Hailo용 컴파일된 detection model입니다.

## Dependencies

### Root Hailo build

- CMake `>= 3.15`
- C++17 compiler
- HailoRT SDK / dev package
- Eigen3
- Boost
- OpenCV (`core`, `imgproc`, `videoio`, `imgcodecs`, `highgui`)
- `yaml-cpp`
- OpenMP
- POSIX Threads

### ONNX reference build

- 위 라이브러리 중 HailoRT 대신 ONNX Runtime dev package 필요
- `onnxruntime_cxx_api.h` 헤더와 `libonnxruntime.so` 경로가 필요

## Build

### 1. HailoRT executable

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DHailoRT_DIR=/path/to/HailoRT/cmake
cmake --build . -j"$(nproc)"
```

`find_package(HailoRT)`가 실패하면 `HailoRT_DIR` 또는 `CMAKE_PREFIX_PATH`를 HailoRT CMake config가 있는 경로로 잡아주면 됩니다.

### 2. ONNX Runtime reference executable

```bash
cmake -S SimpleTrack -B build_onnx -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXRUNTIME_LIB=/path/to/onnxruntime/lib/libonnxruntime.so
cmake --build build_onnx -j"$(nproc)"
```

## Run

### HailoRT path

루트 실행 파일은 기본적으로 repo root의 `sfa.hef`를 찾고, 없으면 상위 경로도 한 번 더 확인합니다.

```bash
cd build
./HailoPerceptionProject \
  --data-glob "../at128/*.bin" \
  --config-path ../SimpleTrack/configs/waymo_configs/vc_kf_giou.yaml \
  --class-config ../SimpleTrack/configs/tracker_params.yaml \
  --video-path result.mp4 \
  --video-fps 10 \
  --no-gui
```

`--data-glob`에는 glob 패턴뿐 아니라 directory path, single `.bin` file path도 넣을 수 있습니다.

### ONNX Runtime reference path

```bash
./build_onnx/inference_sfa3d_at128_onnx \
  --model-path /path/to/sfa.onnx \
  --data-glob "./at128/*.bin" \
  --config-path ./SimpleTrack/configs/waymo_configs/vc_kf_giou.yaml \
  --class-config ./SimpleTrack/configs/tracker_params.yaml \
  --video-path result_onnx.mp4
```

## Useful Options

루트 실행기 기준 주요 옵션은 아래와 같습니다.

- `--data-glob <pattern|dir|file>`
  입력 point cloud 지정
- `--config-path <path>`
  tracker base config
- `--class-config <path>`
  클래스별 tracker 파라미터
- `--video-path <path>`
  출력 비디오 경로
- `--video-fps <fps>`
  출력 비디오 FPS
- `--start-frame <idx>`
  시작 프레임 index
- `--end-frame <idx>`
  종료 프레임 index
- `--max-frames <N>`
  처리 프레임 수 제한
- `--detection-vis <bool>`
  raw detection 박스 시각화 on/off
- `--tracking-vis <bool>`
  tracking 박스 시각화 on/off
- `--debug-assoc`
  association debug log
- `--debug-life`
  tracker lifecycle debug log
- `--debug-all`
  전체 debug log
- `--debug-coords`
  BEV/ego 좌표 변환 log
- `--debug-det`
  detection -> FrameData 변환 log
- `--debug-yaw`
  pre-NMS yaw log

도움말은 아래처럼 확인할 수 있습니다.

```bash
./HailoPerceptionProject --help
```

## Tracker Configuration

클래스별 파라미터는 `SimpleTrack/configs/tracker_params.yaml`에서 조정합니다.

- `giou_asso`
- `giou_redund`
- `max_age`
- `min_hits`
- `score_thr`
- `nms_th`
- `post_nms_iou`
- `measurement_noise`

현재 기본 설정은 vehicle / cyclist / pedestrian 각각 다른 association threshold, lifecycle, score threshold를 사용합니다.

## Notes

- 루트 앱은 클래스별로 tracker를 3개 따로 돌립니다.
- 입력 `.bin` 파일은 파일명 순으로 정렬해서 처리합니다.
- `result.png`는 매 프레임 저장되므로 마지막 프레임 결과만 남습니다.
- `SimpleTrack/LICENSE`는 원본 SimpleTrack의 MIT 라이선스입니다. 공개 저장소로 올릴 때는 attribution을 유지하는 편이 안전합니다.

## GitHub Upload

이 폴더는 현재 Git 저장소가 아니므로, 처음 올릴 때는 아래 순서로 하면 됩니다.

### 1. 업로드 전 확인

- `sfa.hef`를 그대로 올릴지 먼저 결정하세요.
  - 현재 파일 크기는 GitHub 일반 업로드 한도(100MB) 안쪽이지만, 모델 배포 정책이나 라이선스 이슈가 있으면 release asset이나 Git LFS로 분리하는 편이 낫습니다.
- `.gitignore`에는 build 산출물, 로그, `result.png`, `result.mp4`, Python cache를 제외하도록 넣어뒀습니다.

### 2. 새 GitHub repo 만들고 push

```bash
cd /home/a/sfa3d_adcm
git init
git add .
git commit -m "Initial import: SFA3D + SimpleTrack C++ pipeline"
git branch -M main
git remote add origin https://github.com/<your-id>/<repo-name>.git
git push -u origin main
```

이미 GitHub에서 빈 repo를 만들어 둔 상태라면 위 명령 그대로 쓰면 됩니다.

### 3. GitHub CLI를 쓰는 경우

```bash
cd /home/a/sfa3d_adcm
git init
git add .
git commit -m "Initial import: SFA3D + SimpleTrack C++ pipeline"
gh repo create <repo-name> --private --source=. --remote=origin --push
```

`--private` 대신 `--public`으로 바꾸면 공개 저장소로 올릴 수 있습니다.

## License

- `SimpleTrack/` 원본 프로젝트는 MIT 라이선스를 따릅니다.
- 이 저장소의 나머지 코드에 대해서는 공개 전 별도 라이선스 파일을 정리하는 것을 권장합니다.
