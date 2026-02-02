"""
Preprocessing utilities for IC-LoRA pose control.
Extracts DWPose skeleton frames from input videos/images.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, Union
import tempfile
import imageio

from config import PreprocessingConfig


class PoseExtractor:
    """Extract DWPose skeleton frames from videos or images."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._detector = None

    def _load_detector(self):
        """Lazy load DWPose detector."""
        if self._detector is not None:
            return

        print("Loading DWPose detector...")

        from controlnet_dwpose import DWposeDetector

        self._detector = DWposeDetector(
            model_det=self.config.dwpose_det_model,
            model_pose=self.config.dwpose_pose_model,
            device=self.config.dwpose_device,
        )

        print("DWPose detector loaded.")

    def generate_pose(self, frames: Union[np.ndarray, list[np.ndarray]]) -> np.ndarray:
        """
        Generate DWPose skeleton maps from video frames.

        Args:
            frames: Video frames as numpy array (T, H, W, C) RGB uint8
                    or list of (H, W, C) frames

        Returns:
            Skeleton maps as numpy array (T, H, W, C) RGB uint8
        """
        self._load_detector()

        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)

        from controlnet_dwpose.util import draw_pose

        pose_frames = []
        for frame in frames:
            h, w = frame.shape[:2]

            # Detect pose keypoints
            pose_result = self._detector(frame)

            # Render skeleton on canvas
            # draw_pose returns (3, H, W) uint8 canvas
            canvas = draw_pose(pose_result, h, w)
            pose_image = canvas.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

            pose_frames.append(pose_image)

        return np.stack(pose_frames, axis=0)

    def release(self):
        """Release detector memory."""
        if self._detector is not None:
            self._detector.release_memory()
            self._detector = None


# =============================================================================
# Video I/O Utilities
# =============================================================================

def load_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    target_fps: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """
    Load video frames from file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        target_fps: Target FPS (will skip frames to match)

    Returns:
        Tuple of (frames array (T,H,W,C) RGB uint8, original fps)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame skip for target FPS
    frame_skip = 1
    if target_fps and target_fps < original_fps:
        frame_skip = int(original_fps / target_fps)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")

    return np.stack(frames, axis=0), int(original_fps)


def save_video_frames(
    frames: np.ndarray,
    output_path: str,
    fps: int = 24,
) -> str:
    """
    Save numpy frames to video file using imageio (ffmpeg backend).

    Args:
        frames: Video frames (T, H, W, C) in RGB uint8 format
        output_path: Output video path (.mp4)
        fps: Frames per second

    Returns:
        Output path
    """
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()
    return output_path


def image_to_video_frames(
    image: Union[np.ndarray, Image.Image],
    num_frames: int,
) -> np.ndarray:
    """
    Repeat a single image to create video frames.

    Args:
        image: Input image (RGB)
        num_frames: Number of frames to create

    Returns:
        Video frames (T, H, W, C) RGB uint8
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    return np.stack([image] * num_frames, axis=0)


def resize_frames(
    frames: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Resize video frames to target dimensions.

    Args:
        frames: (T, H, W, C) RGB uint8
        height: Target height
        width: Target width

    Returns:
        Resized frames (T, height, width, C)
    """
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames

    resized = []
    for frame in frames:
        r = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        resized.append(r)

    return np.stack(resized, axis=0)


def pad_or_trim_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Ensure exact number of frames by padding (repeat last) or trimming.

    Args:
        frames: (T, H, W, C)
        num_frames: Desired frame count

    Returns:
        Adjusted frames (num_frames, H, W, C)
    """
    current = len(frames)

    if current == num_frames:
        return frames
    elif current > num_frames:
        return frames[:num_frames]
    else:
        # Pad by repeating last frame
        last_frame = frames[-1:]
        padding = np.repeat(last_frame, num_frames - current, axis=0)
        return np.concatenate([frames, padding], axis=0)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <video_path> [output_path]")
        print("  Extracts DWPose skeleton video from input.")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "pose_skeleton.mp4"

    # Load video
    print(f"Loading video: {video_path}")
    frames, fps = load_video_frames(video_path, max_frames=97)
    print(f"Loaded {len(frames)} frames at {fps} fps, shape: {frames.shape}")

    # Generate pose skeleton
    extractor = PoseExtractor()
    print("Generating DWPose skeleton...")
    pose_frames = extractor.generate_pose(frames)
    print(f"Pose frames shape: {pose_frames.shape}")

    # Save result
    save_video_frames(pose_frames, output_path, fps=min(fps, 24))
    print(f"Saved to: {output_path}")

    extractor.release()
