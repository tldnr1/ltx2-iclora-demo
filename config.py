"""
LTX-2 IC-LoRA Demo Configuration
Pose-only control using DWPose for skeleton extraction.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# Paths Configuration
# ============================================================================

# Base paths - these should be mounted via docker-compose
MODELS_ROOT = os.getenv("MODELS_ROOT", "/models")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/app/outputs")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "/app/uploads")

# Model paths
CHECKPOINT_PATH = os.path.join(MODELS_ROOT, "ltx-2-19b-distilled.safetensors")
SPATIAL_UPSAMPLER_PATH = os.path.join(MODELS_ROOT, "ltx-2-spatial-upscaler-x2-1.0.safetensors")
GEMMA_ROOT = os.path.join(MODELS_ROOT, "gemma-3-12b-it-qat-q4_0-unquantized")

# IC-LoRA path (pose-only)
IC_LORA_POSE_PATH = os.path.join(MODELS_ROOT, "ltx-2-19b-ic-lora-pose-control.safetensors")

# ============================================================================
# Generation Defaults
# ============================================================================

@dataclass
class GenerationConfig:
    """
    Generation parameters for ICLoraPipeline.

    Note: ICLoraPipeline.__call__ does NOT accept negative_prompt,
    num_inference_steps, or cfg_guidance_scale. These are handled
    internally by the two-stage pipeline.
    """
    # Video dimensions (must be divisible by 32)
    width: int = 768
    height: int = 512

    # Frame settings (num_frames must be 8k + 1)
    num_frames: int = 97  # ~4 seconds at 24fps
    frame_rate: int = 24

    # IC-LoRA settings
    # Official docs: IC-LoRAs are designed to work at full strength (1.0)
    ic_lora_strength: float = 1.0

    # Video conditioning strength
    # Controls how strongly the pose skeleton guides generation
    conditioning_strength: float = 1.0

    # Seed
    seed: int = 42

    # Memory optimization
    fp8_transformer: bool = True

    @property
    def duration(self) -> float:
        """Calculate video duration in seconds"""
        return (self.num_frames - 1) / self.frame_rate


# ============================================================================
# Preprocessing Configuration
# ============================================================================

@dataclass
class PreprocessingConfig:
    """DWPose preprocessing settings for skeleton extraction."""
    # DWPose ONNX model files
    # These are auto-downloaded by controlnet-dwpose if not present
    dwpose_det_model: str = "yolox_l.onnx"
    dwpose_pose_model: str = "dw-ll_ucoco_384.onnx"

    # Device for DWPose inference
    dwpose_device: str = "cuda"


# ============================================================================
# Server Configuration
# ============================================================================

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist"""
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)


def validate_model_paths() -> dict[str, bool]:
    """Check which model files exist"""
    paths = {
        "checkpoint": CHECKPOINT_PATH,
        "spatial_upsampler": SPATIAL_UPSAMPLER_PATH,
        "gemma": GEMMA_ROOT,
        "ic_lora_pose": IC_LORA_POSE_PATH,
    }

    results = {}
    for name, path in paths.items():
        exists = os.path.exists(path)
        results[name] = exists
        if not exists:
            print(f"Missing: {name} at {path}")

    return results


def get_valid_resolution(height: int, width: int) -> tuple[int, int]:
    """Adjust resolution to be divisible by 32 (VAE requirement)"""
    height = height - (height % 32)
    width = width - (width % 32)
    return height, width


def get_valid_num_frames(num_frames: int) -> int:
    """Adjust num_frames to be 8k + 1"""
    # num_frames should be 8k + 1 for some integer k
    k = (num_frames - 1) // 8
    return k * 8 + 1
