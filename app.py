"""
LTX-2 IC-LoRA Demo - Gradio UI (Pose Control Only)
"""
import os
import gradio as gr
import torch
from pathlib import Path
from typing import Optional

from config import (
    GenerationConfig,
    OUTPUTS_DIR,
    UPLOADS_DIR,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_SHARE,
    ensure_directories,
    validate_model_paths,
    get_valid_resolution,
    get_valid_num_frames,
)
from pipeline import LTX2PosePipeline, get_pipeline


# =============================================================================
# Global State
# =============================================================================

_pipeline: Optional[LTX2PosePipeline] = None


def get_or_load_pipeline() -> LTX2PosePipeline:
    """Get or load pipeline."""
    global _pipeline

    if _pipeline is None:
        _pipeline = LTX2PosePipeline()

    return _pipeline


# =============================================================================
# Generation Function
# =============================================================================

def generate_video(
    prompt: str,
    video_file: Optional[str],
    image_file: Optional[str],
    width: int,
    height: int,
    num_frames: int,
    frame_rate: int,
    ic_lora_strength: float,
    conditioning_strength: float,
    seed: int,
    progress=gr.Progress(),
) -> tuple[Optional[str], Optional[str], str]:
    """
    Generate video with pose-controlled IC-LoRA.

    Returns:
        Tuple of (output_video_path, control_preview_path, status_message)
    """
    # Validate inputs
    if not prompt.strip():
        return None, None, "Please enter a prompt."

    if video_file is None and image_file is None:
        return None, None, "Please upload a video or image for conditioning."

    # Validate dimensions
    height, width = get_valid_resolution(height, width)
    num_frames = get_valid_num_frames(num_frames)

    progress(0, desc="Loading pipeline...")

    try:
        # Get pipeline
        pipeline = get_or_load_pipeline()

        # Create config
        config = GenerationConfig(
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            ic_lora_strength=ic_lora_strength,
            conditioning_strength=conditioning_strength,
            seed=seed,
        )

        # Progress callback
        def on_progress(step, total, message):
            progress(step / total, desc=message)

        progress(0.1, desc="Generating video...")

        # Generate
        with torch.no_grad():
            output_path, control_preview = pipeline.generate(
                prompt=prompt,
                video_path=video_file,
                image_path=image_file,
                config=config,
                progress_callback=on_progress,
            )

        progress(1.0, desc="Complete!")

        duration = config.duration
        return (
            output_path,
            control_preview,
            f"Generated {duration:.1f}s video at {width}x{height} ({num_frames} frames)",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"


# =============================================================================
# UI Components
# =============================================================================

def create_ui() -> gr.Blocks:
    """Create Gradio UI."""

    # Check model status
    model_status = validate_model_paths()
    models_ready = all(model_status.values())

    with gr.Blocks(
        title="LTX-2 Pose Control Demo",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # LTX-2 IC-LoRA Pose Control Demo

        Generate videos with **DWPose skeleton control**.
        Upload a video, and the pose skeleton will be extracted automatically
        to guide the generation.

        **How to use:**
        1. Upload a conditioning video or image
        2. Write a prompt describing the desired output
        3. Click Generate!
        """)

        # Model status warning
        if not models_ready:
            missing = [k for k, v in model_status.items() if not v]
            gr.Markdown(f"""
            **Warning: Some models are missing!**

            Missing: {', '.join(missing)}

            Please download the required models to `/models` directory.
            """)

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3,
                )

                with gr.Tabs():
                    with gr.Tab("Video Input"):
                        video_input = gr.Video(
                            label="Conditioning Video",
                            sources=["upload"],
                        )

                    with gr.Tab("Image Input"):
                        image_input = gr.Image(
                            label="Conditioning Image",
                            type="filepath",
                        )

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        width = gr.Slider(
                            minimum=256, maximum=1280, step=32, value=768,
                            label="Width",
                        )
                        height = gr.Slider(
                            minimum=256, maximum=1280, step=32, value=512,
                            label="Height",
                        )

                    with gr.Row():
                        num_frames = gr.Slider(
                            minimum=9, maximum=257, step=8, value=97,
                            label="Frames",
                            info="Number of frames (8k+1)",
                        )
                        frame_rate = gr.Slider(
                            minimum=8, maximum=30, step=1, value=24,
                            label="FPS",
                        )

                    ic_lora_strength = gr.Slider(
                        minimum=0.5, maximum=1.0, step=0.05, value=1.0,
                        label="IC-LoRA Strength",
                        info="Official recommendation: 1.0",
                    )

                    conditioning_strength = gr.Slider(
                        minimum=0.1, maximum=1.0, step=0.05, value=1.0,
                        label="Conditioning Strength",
                        info="How strongly the pose skeleton guides generation",
                    )

                    seed = gr.Number(
                        value=42,
                        label="Seed",
                        precision=0,
                    )

                generate_btn = gr.Button(
                    "Generate Video",
                    variant="primary",
                    size="lg",
                )

            # Right column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### Output")

                output_video = gr.Video(
                    label="Generated Video",
                )

                control_preview = gr.Video(
                    label="Pose Skeleton Preview",
                )

                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

        # Examples
        gr.Markdown("### Example Prompts")
        gr.Examples(
            examples=[
                ["A person dancing gracefully in a ballroom, cinematic lighting"],
                ["An animated character performing martial arts, anime style"],
                ["A robot mimicking human movements, sci-fi environment"],
                ["A dancer in traditional clothing, studio lighting, high quality"],
            ],
            inputs=prompt,
        )

        # Connect events
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                video_input,
                image_input,
                width,
                height,
                num_frames,
                frame_rate,
                ic_lora_strength,
                conditioning_strength,
                seed,
            ],
            outputs=[output_video, control_preview, status],
            show_progress="full",
        )

        # Footer
        gr.Markdown("""
        ---
        **LTX-2 IC-LoRA Pose Control Demo** | Built with [LTX-2](https://github.com/Lightricks/LTX-2) and [Gradio](https://gradio.app)

        Pose extraction: [DWPose](https://github.com/IDEA-Research/DWPose) (ONNX)

        This is a demo. Generated content may not be accurate.
        """)

    return demo


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    # Setup directories
    ensure_directories()

    # Create and launch UI
    demo = create_ui()

    print(f"\nStarting LTX-2 Pose Control Demo")
    print(f"   Server: http://{GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")

    demo.queue()
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
    )


if __name__ == "__main__":
    main()
