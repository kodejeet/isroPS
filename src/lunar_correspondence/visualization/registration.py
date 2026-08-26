"""Registration overlay and alignment comparison visualization module."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lunar_correspondence.preprocessing.normalization import to_grayscale


def plot_registration_overlay(
    image_ref: np.ndarray,
    registered_src: np.ndarray,
    output_path: str | None = None,
) -> np.ndarray:
    """Create a 2-panel registration result figure (Reference vs Registered Warped, plus Color Overlay).

    In the color overlay:
    - Red channel: Registered warped source image
    - Green channel: Reference image
    - Blue channel: Average of source and reference
    Perfect alignment produces crisp monochromatic details; misalignments appear as red/green color fringes.

    Args:
        image_ref: Reference image array.
        registered_src: Warped source image array aligned to reference shape.
        output_path: Optional path to save figure.

    Returns:
        RGB numpy array of composite figure canvas.
    """
    ref_gray = to_grayscale(image_ref)
    warped_gray = to_grayscale(registered_src)

    # Ensure matching shape
    h, w = ref_gray.shape[:2]
    if warped_gray.shape[:2] != (h, w):
        warped_gray = cv2.resize(warped_gray, (w, h))

    # Construct false-color overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = warped_gray  # Red channel = Warped source
    overlay[:, :, 1] = ref_gray  # Green channel = Reference
    overlay[:, :, 2] = (
        (warped_gray.astype(np.uint16) + ref_gray.astype(np.uint16)) // 2
    ).astype(np.uint8)

    # Construct side-by-side composite canvas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ref_gray, cmap="gray")
    axes[0].set_title("Reference Image")
    axes[0].axis("off")

    axes[1].imshow(warped_gray, cmap="gray")
    axes[1].set_title("Warped Source Image")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("False Color Overlay (R=Source, G=Ref)")
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    # Render figure to numpy canvas
    fig.canvas.draw()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba())
        canvas = rgba[:, :, :3].copy()
    except AttributeError:
        # Fallback for older matplotlib
        canvas = overlay.copy()

    plt.close(fig)
    return canvas
