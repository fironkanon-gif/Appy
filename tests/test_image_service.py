# ============================================================
# GOTHIC OCR — IMAGE SERVICE TESTS
# ============================================================
from pathlib import Path
import numpy as np
import pytest

from services.image_service import ImageService


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def service():
    return ImageService()


def make_image(width, height):
    """
    إنشاء صورة RGB وهمية بالحجم المطلوب.
    """

    return np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )


# ============================================================
# RGB VALIDATION
# ============================================================

def test_rgb_image_is_valid(service):

    image = make_image(
        640,
        480,
    )

    result = service._validate_rgb(
        image
    )

    assert result.shape == (
        480,
        640,
        3,
    )

    assert result.dtype == np.uint8


def test_grayscale_image_is_rejected(service):

    image = np.zeros(
        (480, 640),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):

        service._validate_rgb(
            image
        )


def test_rgba_image_is_rejected(service):

    image = np.zeros(
        (480, 640, 4),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):

        service._validate_rgb(
            image
        )


# ============================================================
# PREPARE
# ============================================================

def test_prepare_returns_1024_input(service):

    image = make_image(
        800,
        600,
    )

    prepared, meta = service.prepare(
        image
    )

    assert prepared.shape == (
        1,
        1024,
        1024,
        3,
    )


def test_prepare_returns_float32(service):

    image = make_image(
        800,
        600,
    )

    prepared, meta = service.prepare(
        image
    )

    assert prepared.dtype == np.float32


def test_prepare_normalizes_pixels(service):

    image = np.full(
        (500, 700, 3),
        255,
        dtype=np.uint8,
    )

    prepared, meta = service.prepare(
        image
    )

    assert prepared.max() <= 1.0

    assert prepared.min() >= 0.0


# ============================================================
# METADATA
# ============================================================

def test_prepare_metadata_contains_original_size(
    service,
):

    width = 800
    height = 600

    image = make_image(
        width,
        height,
    )

    prepared, meta = service.prepare(
        image
    )

    assert meta["original_width"] == width

    assert meta["original_height"] == height


def test_prepare_scale_is_positive(service):

    image = make_image(
        1600,
        900,
    )

    prepared, meta = service.prepare(
        image
    )

    assert meta["scale"] > 0


def test_prepare_contains_padding_metadata(
    service,
):

    image = make_image(
        800,
        600,
    )

    prepared, meta = service.prepare(
        image
    )

    assert "pad_x" in meta

    assert "pad_y" in meta


# ============================================================
# TILE POSITIONS
# ============================================================

def test_small_image_has_one_tile(service):

    positions = service._tile_positions(
        800,
        600,
        1024,
        0.15,
    )

    assert positions == [0]


def test_exact_1024_image_has_one_tile(service):

    positions = service._tile_positions(
        1024,
        1024,
        1024,
        0.15,
    )

    assert positions == [0]


def test_large_dimension_has_multiple_tiles(
    service,
):

    positions = service._tile_positions(
        3000,
        2000,
        1024,
        0.15,
    )

    assert len(positions) > 1


def test_tile_positions_are_sorted(service):

    positions = service._tile_positions(
        5000,
        3500,
        1024,
        0.15,
    )

    assert positions == sorted(
        positions
    )


def test_tile_positions_do_not_exceed_image(
    service,
):

    image_width = 3500
    tile_size = 1024

    positions = service._tile_positions(
        image_width,
        2000,
        tile_size,
        0.15,
    )

    for x in positions:

        assert x >= 0

        assert x + tile_size <= image_width


# ============================================================
# TILING — SMALL IMAGE
# ============================================================

def test_iter_tiles_small_image(service):

    image = make_image(
        800,
        600,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert len(tiles) == 1

    tile = tiles[0]

    assert tile["input"].shape == (
        1,
        1024,
        1024,
        3,
    )

    assert tile["offset_x"] == 0

    assert tile["offset_y"] == 0


# ============================================================
# TILING — LARGE IMAGE
# ============================================================

def test_iter_tiles_large_image(service):

    image = make_image(
        3000,
        2500,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert len(tiles) > 1


def test_each_tile_has_correct_input_shape(
    service,
):

    image = make_image(
        3000,
        2500,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    for tile in tiles:

        assert tile["input"].shape == (
            1,
            1024,
            1024,
            3,
        )


# ============================================================
# TILE OFFSETS
# ============================================================

def test_tile_offsets_are_non_negative(
    service,
):

    image = make_image(
        3000,
        2500,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    for tile in tiles:

        assert tile["offset_x"] >= 0

        assert tile["offset_y"] >= 0


def test_tile_offsets_stay_inside_image(
    service,
):

    width = 3000
    height = 2500

    image = make_image(
        width,
        height,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    for tile in tiles:

        x = tile["offset_x"]
        y = tile["offset_y"]

        tile_width = tile["tile_width"]
        tile_height = tile["tile_height"]

        assert x + tile_width <= width

        assert y + tile_height <= height


# ============================================================
# TILE COVERAGE
# ============================================================

def test_tiles_cover_left_edge(service):

    image = make_image(
        3000,
        2500,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert any(
        tile["offset_x"] == 0
        for tile in tiles
    )


def test_tiles_cover_top_edge(service):

    image = make_image(
        3000,
        2500,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert any(
        tile["offset_y"] == 0
        for tile in tiles
    )


def test_tiles_cover_right_edge(service):

    width = 3000

    image = make_image(
        width,
        2500,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert any(
        tile["offset_x"]
        + tile["tile_width"]
        == width
        for tile in tiles
    )


def test_tiles_cover_bottom_edge(service):

    height = 2500

    image = make_image(
        3000,
        height,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert any(
        tile["offset_y"]
        + tile["tile_height"]
        == height
        for tile in tiles
    )


# ============================================================
# OVERLAP
# ============================================================

def test_overlap_creates_shared_regions(
    service,
):

    width = 3000

    positions = service._tile_positions(
        width,
        1024,
        1024,
        0.15,
    )

    assert len(positions) >= 2

    for first, second in zip(
        positions,
        positions[1:],
    ):

        overlap_width = (
            first
            + 1024
            - second
        )

        assert overlap_width > 0


# ============================================================
# RGB CONTENT
# ============================================================

def test_tile_preserves_rgb_channels(service):

    image = np.zeros(
        (1500, 1500, 3),
        dtype=np.uint8,
    )

    image[:, :, 0] = 255

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert len(tiles) > 1

    for tile in tiles:

        prepared = tile["input"]

        assert prepared.shape[-1] == 3

        assert prepared.dtype == np.float32


# ============================================================
# TILE METADATA
# ============================================================

def test_tile_contains_required_metadata(
    service,
):

    image = make_image(
        2000,
        1600,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    required_keys = {
        "input",
        "meta",
        "offset_x",
        "offset_y",
        "tile_width",
        "tile_height",
    }

    for tile in tiles:

        assert required_keys.issubset(
            tile.keys()
        )


def test_tile_metadata_keeps_original_dimensions(
    service,
):

    width = 2000
    height = 1600

    image = make_image(
        width,
        height,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    for tile in tiles:

        meta = tile["meta"]

        assert (
            meta["original_width"]
            == tile["tile_width"]
        )

        assert (
            meta["original_height"]
            == tile["tile_height"]
        )


# ============================================================
# DIFFERENT IMAGE SIZES
# ============================================================

@pytest.mark.parametrize(
    "width,height",
    [
        (100, 100),
        (640, 480),
        (1024, 1024),
        (1500, 1200),
        (2048, 2048),
        (3000, 2000),
        (5000, 3500),
    ],
)
def test_tiling_for_multiple_sizes(
    service,
    width,
    height,
):

    image = make_image(
        width,
        height,
    )

    tiles = list(
        service.iter_tiles(
            image,
            overlap=0.15,
            tile_size=1024,
        )
    )

    assert len(tiles) >= 1

    for tile in tiles:

        assert tile["input"].shape == (
            1,
            1024,
            1024,
            3,
        )

        assert (
            0 <= tile["offset_x"] <= width
        )

        assert (
            0 <= tile["offset_y"] <= height
    )
