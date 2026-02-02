# LTX-2 IC-LoRA Pose Control Demo

[LTX-2](https://github.com/Lightricks/LTX-2)의 IC-LoRA(In-Context LoRA) 파이프라인을 활용한 **포즈 기반 비디오 생성** 데모입니다. 입력 영상에서 [DWPose](https://github.com/IDEA-Research/DWPose)로 스켈레톤을 추출하고, 텍스트 프롬프트와 결합하여 포즈를 따라가는 새로운 비디오를 생성합니다.

## Overview

```
Input Video  ──→  DWPose Skeleton  ──→  ICLoraPipeline  ──→  Generated Video
                  (body/hands/face)      + Text Prompt
```

- **Video-to-Video (V2V)**: 원본 영상의 포즈 모션을 유지하면서 프롬프트에 맞는 새 영상 생성
- **Image-to-Video (I2V)**: 참조 이미지와 포즈 컨트롤을 결합하여 영상 생성
- **DWPose**: Body(18) + Hands(42) + Face(68) keypoints를 추출하는 whole-body 포즈 추정
- **Two-stage pipeline**: Stage 1에서 IC-LoRA 기반 생성 후, Stage 2에서 2x 업스케일 및 디테일 보정

## Requirements

### Hardware

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU VRAM | 24GB (FP8) | 48GB+ |
| RAM | 32GB | 64GB+ |
| Storage | 60GB | 100GB+ |

### Software

- Docker + NVIDIA Container Toolkit
- CUDA 12.8+
- Python 3.12+ (Docker 없이 실행 시)

## Quick Start

### 1. Download Models

자동 다운로드 스크립트 사용:

```bash
chmod +x download_models.sh
./download_models.sh /path/to/models
```

또는 직접 다운로드:

```bash
mkdir -p /path/to/models

# LTX-2 base models
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled.safetensors \
    ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --local-dir /path/to/models

# Pose IC-LoRA
huggingface-cli download Lightricks/LTX-2-19b-IC-LoRA-Pose-Control \
    ltx-2-19b-ic-lora-pose-control.safetensors \
    --local-dir /path/to/models

# Gemma text encoder
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir /path/to/models/gemma-3-12b-it-qat-q4_0-unquantized
```

다운로드 후 디렉토리 구조:

```
/path/to/models/
├── ltx-2-19b-distilled.safetensors           (~38GB)
├── ltx-2-spatial-upscaler-x2-1.0.safetensors (~2GB)
├── gemma-3-12b-it-qat-q4_0-unquantized/     (~12GB)
│   └── model-*.safetensors
└── ltx-2-19b-ic-lora-pose-control.safetensors (~1.5GB)
```

### 2. Run with Docker

```bash
export MODELS_PATH=/path/to/models
docker compose up --build
```

브라우저에서 `http://localhost:7862` 접속.

### 3. Run without Docker

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# LTX-2 packages (monorepo에서 설치)
pip install 'git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core'
pip install 'git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines'

export MODELS_ROOT=/path/to/models
python app.py
```

## Usage

### Gradio UI

1. **Video Input** 탭에서 컨디셔닝 영상을 업로드 (또는 **Image Input** 탭에서 참조 이미지 업로드)
2. **Prompt**에 원하는 출력 영상을 설명
3. **Generate Video** 클릭
4. 오른쪽에서 생성된 영상과 포즈 스켈레톤 프리뷰 확인

### Python API

```python
from pipeline import generate_video

output_path = generate_video(
    prompt="A dancer in traditional clothing, cinematic lighting",
    video_path="input_dance.mp4",
    width=768,
    height=512,
    num_frames=97,
    seed=42,
)
```

### Preprocessing Only (포즈 스켈레톤 추출만)

```bash
python preprocessing.py input_video.mp4 output_skeleton.mp4
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_ROOT` | `/models` | 모델 가중치 디렉토리 |
| `OUTPUTS_DIR` | `/app/outputs` | 생성 영상 저장 경로 |
| `GRADIO_SERVER_PORT` | `7860` | Gradio 서버 포트 |
| `GRADIO_SHARE` | `false` | 외부 공유 링크 활성화 |

### Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `width` | 768 | 256-1280 | 영상 너비 (32의 배수) |
| `height` | 512 | 256-1280 | 영상 높이 (32의 배수) |
| `num_frames` | 97 | 9-257 | 프레임 수 (8k+1) |
| `frame_rate` | 24 | 8-30 | 출력 FPS |
| `ic_lora_strength` | 1.0 | 0.5-1.0 | IC-LoRA 가중치 강도 |
| `conditioning_strength` | 1.0 | 0.1-1.0 | 포즈 컨디셔닝 강도 |
| `seed` | 42 | - | 재현성을 위한 시드 |

> `ic_lora_strength`는 공식 문서 기준 1.0 사용을 권장합니다. `conditioning_strength`는 `ICLoraPipeline`의 `video_conditioning` 파라미터에 전달되어, 포즈 스켈레톤이 생성 결과에 미치는 영향도를 조절합니다.

## Project Structure

```
ltx2-iclora-demo/
├── app.py               # Gradio UI
├── pipeline.py          # LTX2PosePipeline (ICLoraPipeline wrapper)
├── preprocessing.py     # DWPose skeleton extraction + video I/O
├── config.py            # Configuration (paths, defaults, validation)
├── requirements.txt     # Python dependencies
├── download_models.sh   # Model download script
├── Dockerfile
├── docker-compose.yml
├── .env.sample
└── README.md
```

### Architecture

```
app.py (Gradio UI)
  └── pipeline.py (LTX2PosePipeline)
        ├── preprocessing.py (PoseExtractor - DWPose)
        │     └── controlnet-dwpose (ONNX)
        ├── ltx_pipelines.ic_lora.ICLoraPipeline
        │     ├── Stage 1: IC-LoRA conditioned generation
        │     └── Stage 2: 2x spatial upscale + refinement
        └── config.py (GenerationConfig, paths)
```

### ICLoraPipeline API

이 프로젝트가 래핑하는 `ICLoraPipeline.__call__`의 실제 시그니처:

```python
def __call__(
    self,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    images: list[tuple[str, int, float]],          # (path, frame_idx, strength)
    video_conditioning: list[tuple[str, float]],    # (video_path, strength)
    enhance_prompt: bool = False,
    tiling_config: TilingConfig | None = None,
) -> tuple[Iterator[torch.Tensor], torch.Tensor]
```

> `negative_prompt`, `num_inference_steps`, `cfg_guidance_scale` 파라미터는 이 API에 **존재하지 않습니다**. Distilled 모델의 추론 스텝과 CFG 설정은 파이프라인 내부에서 처리됩니다.

## Troubleshooting

### Out of Memory (OOM)

- FP8 모드가 기본 활성화되어 있습니다 (`fp8_transformer=True`)
- 해상도를 줄여보세요: 512x384
- 프레임 수를 줄여보세요: 49 프레임 (~2초)

### DWPose 모델 다운로드

`controlnet-dwpose`는 첫 실행 시 ONNX 모델(`yolox_l.onnx`, `dw-ll_ucoco_384.onnx`)을 자동 다운로드합니다. 오프라인 환경에서는 사전 다운로드가 필요합니다.

### CUDA 확인

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 모델 경로 검증

```bash
python -c "from config import validate_model_paths; validate_model_paths()"
```

## References

- [LTX-2](https://github.com/Lightricks/LTX-2) - Base model repository
- [LTX-2-19b-IC-LoRA-Pose-Control](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control) - Pose LoRA weights
- [DWPose](https://github.com/IDEA-Research/DWPose) - Whole-body pose estimation (ICCV 2023)
- [IC-LoRA Documentation](https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra) - Official usage guide

## License

This project uses LTX-2 models which are subject to their own license terms. Please refer to the [LTX-2 repository](https://github.com/Lightricks/LTX-2) for licensing information.
