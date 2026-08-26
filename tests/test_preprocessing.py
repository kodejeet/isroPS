"""Unit tests for image preprocessing functions."""

import numpy as np

from lunar_correspondence.preprocessing.enhancement import apply_clahe
from lunar_correspondence.preprocessing.normalization import (
    normalize_to_uint8,
    to_grayscale,
)
from lunar_correspondence.preprocessing.pyramid import build_gaussian_pyramid
from lunar_correspondence.preprocessing.tiling import generate_tiles


def test_normalize_to_uint8():
    arr_16 = np.array([[0, 32768], [65535, 10000]], dtype=np.uint16)
    arr_8 = normalize_to_uint8(arr_16)
    assert arr_8.dtype == np.uint8
    assert arr_8.min() == 0
    assert arr_8.max() == 255


def test_to_grayscale():
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    gray = to_grayscale(rgb)
    assert gray.shape == (50, 50)
    assert gray.dtype == np.uint8


def test_apply_clahe():
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    enhanced = apply_clahe(img)
    assert enhanced.shape == (100, 100)


def test_pyramid_and_tiling():
    img = np.zeros((256, 256, 1), dtype=np.uint8)
    pyramid = build_gaussian_pyramid(img, levels=3)
    assert len(pyramid) == 3
    assert pyramid[1].shape[:2] == (128, 128)

    tiles = generate_tiles(img, tile_size=(128, 128), overlap=16)
    assert len(tiles) > 0
