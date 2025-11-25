"""
Image Comparison Helpers

Utilities for comparing rendered images in visual tests.
Used to validate that transpiled shaders produce correct output.

Will be fully implemented in Phase 6 when we have visual validation tests.
"""

import numpy as np
from typing import Tuple, Optional


def compute_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.

    Args:
        image1: First image as numpy array
        image2: Second image as numpy array

    Returns:
        MSE value (0.0 = identical, higher = more different)

    Raises:
        ValueError: If images have different shapes
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Images must have same shape. Got {image1.shape} and {image2.shape}"
        )

    mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
    return float(mse)


def compute_psnr(image1: np.ndarray, image2: np.ndarray, max_value: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        image1: First image as numpy array
        image2: Second image as numpy array
        max_value: Maximum possible pixel value (1.0 for normalized, 255 for uint8)

    Returns:
        PSNR value in dB (higher is better, typically 20-50 dB for lossy compression)
    """
    mse = compute_mse(image1, image2)

    if mse == 0:
        return float('inf')  # Images are identical

    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return float(psnr)


def images_nearly_equal(
    image1: np.ndarray,
    image2: np.ndarray,
    tolerance: float = 1e-5
) -> bool:
    """
    Check if two images are nearly equal within tolerance.

    Useful for comparing floating-point rendered images where exact
    equality is not expected due to numerical precision differences.

    Args:
        image1: First image as numpy array
        image2: Second image as numpy array
        tolerance: Maximum allowed absolute difference per pixel

    Returns:
        True if images are nearly equal, False otherwise
    """
    if image1.shape != image2.shape:
        return False

    max_diff = np.max(np.abs(image1.astype(float) - image2.astype(float)))
    return max_diff <= tolerance


# Placeholder for SSIM (Structural Similarity Index)
# Will be implemented in Phase 6 when needed for visual validation
def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    window_size: int = 11
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM is a perceptual metric that quantifies image quality degradation.
    Range: [-1, 1] where 1 = identical

    Args:
        image1: First image as numpy array
        image2: Second image as numpy array
        window_size: Size of sliding window for comparison

    Returns:
        SSIM value (higher is better, 1.0 = identical)

    Note:
        This is a placeholder. Full implementation will be added in Phase 6.
        For now, returns simple MSE-based approximation.
    """
    # Placeholder implementation using MSE
    # TODO: Implement proper SSIM in Phase 6
    mse = compute_mse(image1, image2)
    if mse == 0:
        return 1.0

    # Approximate SSIM from MSE (not accurate but good enough for placeholder)
    approximate_ssim = 1.0 / (1.0 + mse)
    return approximate_ssim


def save_comparison_image(
    image1: np.ndarray,
    image2: np.ndarray,
    diff_image: np.ndarray,
    output_path: str
) -> None:
    """
    Save side-by-side comparison image showing original, result, and difference.

    Args:
        image1: First image (e.g., reference from Shadertoy)
        image2: Second image (e.g., transpiled OpenCL output)
        diff_image: Difference image
        output_path: Path to save comparison image

    Note:
        This is a placeholder. Will be implemented in Phase 6 using PIL or matplotlib.
    """
    # TODO: Implement in Phase 6 when visual validation is needed
    raise NotImplementedError(
        "Visual comparison saving not yet implemented (Phase 6)"
    )
