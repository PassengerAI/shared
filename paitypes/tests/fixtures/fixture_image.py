from typing import List, Iterator
import pytest
import numpy as np
from paitypes.image import (Image, BGRImage, GrayscaleImage,
                            ndarray_to_bgr_image, ndarray_to_grayscale_image)


@pytest.fixture()
def all_valid_images(grayscale_images: List[GrayscaleImage],
                     bgr_images: List[BGRImage]
                     ) -> Iterator[List[Image]]:
    yield grayscale_images + bgr_images


@pytest.fixture()
def random_grayscale_image() -> Iterator[GrayscaleImage]:
    yield ndarray_to_grayscale_image(
        np.random.randint(0, 255, size=(100, 100), dtype=np.uint8))


@pytest.fixture()
def random_bgr_image() -> Iterator[BGRImage]:
    yield ndarray_to_bgr_image(
        np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8))


@pytest.fixture()
def empty_grayscale_image() -> Iterator[GrayscaleImage]:
    yield ndarray_to_grayscale_image(np.zeros((0, 0), dtype=np.uint8))


@pytest.fixture()
def black_grayscale_image() -> Iterator[GrayscaleImage]:
    yield ndarray_to_grayscale_image(np.zeros((100, 100), dtype=np.uint8))


@pytest.fixture()
def gray_grayscale_image() -> Iterator[GrayscaleImage]:
    yield ndarray_to_grayscale_image(
        np.full((100, 100), 255 // 2, dtype=np.uint8))


@pytest.fixture()
def white_grayscale_image() -> Iterator[GrayscaleImage]:
    yield ndarray_to_grayscale_image(
        np.full((100, 100), 255, dtype=np.uint8))


@pytest.fixture()
def empty_bgr_image() -> Iterator[BGRImage]:
    yield ndarray_to_bgr_image(np.zeros((0, 0, 3), dtype=np.uint8))


@pytest.fixture()
def black_bgr_image() -> Iterator[BGRImage]:
    yield ndarray_to_bgr_image(np.zeros((100, 100, 3), dtype=np.uint8))


@pytest.fixture()
def gray_bgr_image() -> Iterator[BGRImage]:
    yield ndarray_to_bgr_image(
        np.full((100, 100, 3), 255 // 2, dtype=np.uint8))


@pytest.fixture()
def white_bgr_image() -> Iterator[BGRImage]:
    yield ndarray_to_bgr_image(
        np.full((100, 100, 3), 255, dtype=np.uint8))


@pytest.fixture()
def blue_bgr_image() -> Iterator[BGRImage]:
    image = ndarray_to_bgr_image(
        np.zeros((100, 100, 3), dtype=np.uint8))
    image[:, :, 0] = 255
    return image


@pytest.fixture()
def green_bgr_image() -> Iterator[BGRImage]:
    image = ndarray_to_bgr_image(
        np.zeros((100, 100, 3), dtype=np.uint8))
    image[:, :, 1] = 255
    return image


@pytest.fixture()
def red_bgr_image() -> Iterator[BGRImage]:
    image = ndarray_to_bgr_image(
        np.zeros((100, 100, 3), dtype=np.uint8))
    image[:, :, 2] = 255
    return image


@pytest.fixture()
def empty_images(empty_bgr_image: BGRImage,
                 empty_grayscale_image: GrayscaleImage
                 ) -> Iterator[List[Image]]:
    yield [empty_bgr_image, empty_grayscale_image]


@pytest.fixture()
def bgr_images(random_bgr_image: BGRImage,
               black_bgr_image: BGRImage,
               gray_bgr_image: BGRImage,
               white_bgr_image: BGRImage
               ) -> Iterator[List[BGRImage]]:
    yield [random_bgr_image, black_bgr_image, gray_bgr_image,
           white_bgr_image]


@pytest.fixture()
def grayscale_images(random_grayscale_image: GrayscaleImage,
                     black_grayscale_image: GrayscaleImage,
                     gray_grayscale_image: GrayscaleImage,
                     white_grayscale_image: GrayscaleImage
                     ) -> Iterator[List[GrayscaleImage]]:
    yield [random_grayscale_image, black_grayscale_image,
           gray_grayscale_image, white_grayscale_image]


@pytest.fixture()
def random_images(random_grayscale_image: GrayscaleImage,
                  random_bgr_image: BGRImage
                  ) -> Iterator[List[Image]]:
    yield [random_grayscale_image, random_bgr_image]
