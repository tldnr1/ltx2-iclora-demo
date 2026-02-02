"""
LTX-2 IC-LoRA Pipeline wrapper (Pose-only).
Provides a simplified interface to the ICLoraPipeline for pose control.

ICLoraPipeline.__call__ signature (from LTX-2 repo):
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union, Callable
from datetime import datetime
import gc
import tempfile

from config import (
    CHECKPOINT_PATH,
    SPATIAL_UPSAMPLER_PATH,
    GEMMA_ROOT,
    IC_LORA_POSE_PATH,
    OUTPUTS_DIR,
    GenerationConfig,
    get_valid_resolution,
    get_valid_num_frames,
)
from preprocessing import (
    PoseExtractor,
    load_video_frames,
    save_video_frames,
    resize_frames,
    pad_or_trim_frames,
)


class LTX2PosePipeline:
    """
    Wrapper for LTX-2 ICLoraPipeline with pose-only control.
    Uses DWPose to extract skeleton frames from input video.
    """

    def __init__(
        self,
        checkpoint_path: str = CHECKPOINT_PATH,
        spatial_upsampler_path: str = SPATIAL_UPSAMPLER_PATH,
        gemma_root: str = GEMMA_ROOT,
        ic_lora_path: str = IC_LORA_POSE_PATH,
        ic_lora_strength: float = 1.0,
        fp8_transformer: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.gemma_root = gemma_root
        self.ic_lora_path = ic_lora_path
        self.ic_lora_strength = ic_lora_strength
        self.fp8_transformer = fp8_transformer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._pipeline = None
        self._pose_extractor = None
        self._tiling_config = None
        self._temp_files: list[str] = []

    def _lazy_load_pipeline(self):
        """Lazy load the pipeline on first use."""
        if self._pipeline is not None:
            return

        print(f"Loading LTX-2 ICLoraPipeline (Pose control)...")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  IC-LoRA: {self.ic_lora_path}")
        print(f"  IC-LoRA strength: {self.ic_lora_strength}")
        print(f"  FP8: {self.fp8_transformer}")

        # Import here to avoid loading at module import time
        from ltx_pipelines.ic_lora import ICLoraPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.model.video_vae import TilingConfig

        if not os.path.exists(self.ic_lora_path):
            raise ValueError(f"IC-LoRA pose model not found: {self.ic_lora_path}")

        # Prepare LoRA config
        loras = [
            LoraPathStrengthAndSDOps(
                path=self.ic_lora_path,
                strength=self.ic_lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        # Initialize pipeline
        self._pipeline = ICLoraPipeline(
            checkpoint_path=self.checkpoint_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            gemma_root=self.gemma_root,
            loras=loras,
            device=self.device,
            fp8transformer=self.fp8_transformer,
        )

        # Default tiling config
        self._tiling_config = TilingConfig.default()

        print("Pipeline loaded successfully!")

    def _get_pose_extractor(self) -> PoseExtractor:
        """Get or create pose extractor."""
        if self._pose_extractor is None:
            self._pose_extractor = PoseExtractor()
        return self._pose_extractor

    def _create_temp_video(self, frames: np.ndarray, fps: int = 24) -> str:
        """
        Save frames to a temporary video file and track it for cleanup.

        Returns:
            Path to temporary video file.
        """
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        save_video_frames(frames, temp_path, fps=fps)
        self._temp_files.append(temp_path)
        return temp_path

    def _cleanup_temp_files(self):
        """Remove all temporary files created during generation."""
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()

    def generate(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Generate video with pose-controlled IC-LoRA.

        Args:
            prompt: Text prompt for generation
            video_path: Path to conditioning video (for V2V)
            image_path: Path to conditioning image (for I2V)
            config: Generation configuration
            progress_callback: Optional callback(step, total, message)

        Returns:
            Tuple of (output_video_path, control_preview_path or None)
        """
        self._lazy_load_pipeline()

        config = config or GenerationConfig()

        # Validate dimensions
        height, width = get_valid_resolution(config.height, config.width)
        num_frames = get_valid_num_frames(config.num_frames)

        if progress_callback:
            progress_callback(0, 100, "Preparing inputs...")

        # Prepare video conditioning: list[tuple[str, float]]
        video_conditioning: list[tuple[str, float]] = []
        control_preview_path: Optional[str] = None

        if video_path:
            conditioning_path, preview_path = self._prepare_pose_conditioning(
                video_path, num_frames, height, width
            )
            video_conditioning = [(conditioning_path, config.conditioning_strength)]
            control_preview_path = preview_path

        # Prepare image conditioning: list[tuple[str, int, float]]
        images: list[tuple[str, int, float]] = []
        if image_path:
            # (path, frame_index, strength)
            # frame_index=0 means the image conditions the first frame
            images = [(image_path, 0, 1.0)]

        if progress_callback:
            progress_callback(10, 100, "Starting generation...")

        # Generate video
        from ltx_core.model.video_vae import get_video_chunks_number
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_pipelines.utils.media_io import encode_video

        video_chunks_number = get_video_chunks_number(num_frames, self._tiling_config)

        with torch.inference_mode():
            video, audio = self._pipeline(
                prompt=prompt,
                seed=config.seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=config.frame_rate,
                images=images,
                video_conditioning=video_conditioning,
                tiling_config=self._tiling_config,
            )

            if progress_callback:
                progress_callback(90, 100, "Encoding video...")

            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUTS_DIR, f"generated_{timestamp}.mp4")

            encode_video(
                video=video,
                fps=config.frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )

        if progress_callback:
            progress_callback(100, 100, "Done!")

        # Cleanup temp files (but not the control preview)
        if control_preview_path and control_preview_path in self._temp_files:
            self._temp_files.remove(control_preview_path)
        self._cleanup_temp_files()
        self._cleanup_memory()

        return output_path, control_preview_path

    def _prepare_pose_conditioning(
        self,
        video_path: str,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[str, str]:
        """
        Prepare pose conditioning from input video.

        1. Load video frames
        2. Resize to target dimensions
        3. Pad/trim to target frame count
        4. Extract DWPose skeleton
        5. Save as temporary video

        Returns:
            Tuple of (conditioning_video_path, preview_video_path)
            Both are saved temporary files.
        """
        # Load video frames
        frames, fps = load_video_frames(video_path, max_frames=num_frames)

        # Resize to target resolution
        frames = resize_frames(frames, height, width)

        # Ensure exact frame count
        frames = pad_or_trim_frames(frames, num_frames)

        # Extract pose skeleton
        extractor = self._get_pose_extractor()
        pose_frames = extractor.generate_pose(frames)

        # Save conditioning video
        conditioning_path = self._create_temp_video(pose_frames, fps=24)

        # Save preview (separate copy for UI display)
        preview_path = self._create_temp_video(pose_frames, fps=24)

        return conditioning_path, preview_path

    def _cleanup_memory(self):
        """Clean up GPU memory after generation."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self):
        """Unload the pipeline to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        if self._pose_extractor is not None:
            self._pose_extractor.release()
            self._pose_extractor = None

        self._cleanup_temp_files()
        self._cleanup_memory()
        print("Pipeline unloaded.")


# =============================================================================
# Convenience Functions
# =============================================================================

_global_pipeline: Optional[LTX2PosePipeline] = None


def get_pipeline(force_reload: bool = False) -> LTX2PosePipeline:
    """
    Get or create a global pipeline instance.

    Args:
        force_reload: Force reload even if pipeline exists

    Returns:
        Pipeline instance
    """
    global _global_pipeline

    if _global_pipeline is None or force_reload:
        if _global_pipeline is not None:
            _global_pipeline.unload()
        _global_pipeline = LTX2PosePipeline()

    return _global_pipeline


def generate_video(
    prompt: str,
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convenience function to generate video with pose control.

    Args:
        prompt: Text prompt
        video_path: Path to conditioning video
        image_path: Path to conditioning image
        **kwargs: Additional arguments passed to GenerationConfig

    Returns:
        Path to generated video
    """
    pipeline = get_pipeline()
    config = GenerationConfig(**kwargs)

    output_path, _ = pipeline.generate(
        prompt=prompt,
        video_path=video_path,
        image_path=image_path,
        config=config,
    )

    return output_path


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    # Simple test
    print("Testing pipeline initialization...")

    # Check if models exist
    from config import validate_model_paths
    results = validate_model_paths()

    all_exist = all(results.values())
    if not all_exist:
        print("\nSome models are missing. Please download them first.")
        print("Required models:")
        print("  - ltx-2-19b-distilled.safetensors")
        print("  - ltx-2-spatial-upscaler-x2-1.0.safetensors")
        print("  - gemma-3-12b-it-qat-q4_0-unquantized/")
        print("  - ltx-2-19b-ic-lora-pose-control.safetensors")
        sys.exit(1)

    print("\nAll models found!")
    print("\nTo test generation, use:")
    print('  python -c "from pipeline import generate_video; generate_video(\'A person dancing\', video_path=\'input.mp4\')"')
