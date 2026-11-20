from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from examshield.errors import ExamShieldError
from examshield.pdfio import GrayImage


@dataclass(frozen=True)
class CaptureConditions:
    print_dpi: int = 300
    capture_width: int = 1600
    skew_degrees: float = 2.0
    perspective: float = 0.03
    illumination: float = 0.25
    blur_sigma: float = 0.8
    noise_sigma: float = 4.0
    jpeg_quality: int = 80
    seed: int = 20261120

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def conditions_from(data: dict[str, Any]) -> CaptureConditions:
    try:
        return CaptureConditions(**data)
    except TypeError as error:
        raise ExamShieldError(f"촬영 시뮬레이션 조건이 잘못됐다: {error}") from error


def simulate_capture(page: GrayImage, conditions: CaptureConditions) -> GrayImage:
    rng = np.random.default_rng(conditions.seed)
    scale = conditions.capture_width / page.shape[1]
    height = max(1, round(page.shape[0] * scale))
    resized = cv2.resize(page, (conditions.capture_width, height), interpolation=cv2.INTER_AREA)
    image = _warp(np.asarray(resized, dtype=np.uint8), conditions, rng)
    image = _light(image, conditions.illumination)
    if conditions.blur_sigma > 0:
        image = np.asarray(cv2.GaussianBlur(image, (0, 0), conditions.blur_sigma), dtype=np.uint8)
    if conditions.noise_sigma > 0:
        noise = rng.normal(0.0, conditions.noise_sigma, image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return _jpeg(image, conditions.jpeg_quality)


def _warp(image: GrayImage, conditions: CaptureConditions, rng: np.random.Generator) -> GrayImage:
    height, width = image.shape
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    jitter = rng.uniform(-1.0, 1.0, corners.shape) * conditions.perspective * width
    moved = (corners + jitter).astype(np.float32)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), conditions.skew_degrees, 1.0)
    moved = cv2.transform(moved[None, :, :], rotation)[0].astype(np.float32)
    matrix = cv2.getPerspectiveTransform(corners, moved)
    warped = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(128,)
    )
    return np.asarray(warped, dtype=np.uint8)


def _light(image: GrayImage, strength: float) -> GrayImage:
    if strength <= 0:
        return image
    height, width = image.shape
    ramp = np.linspace(1.0, 1.0 - strength, width, dtype=np.float64)[None, :]
    falloff = np.linspace(1.0, 1.0 - strength / 2, height, dtype=np.float64)[:, None]
    lit = image.astype(np.float64) * ramp * falloff
    return np.clip(lit, 0, 255).astype(np.uint8)


def _jpeg(image: GrayImage, quality: int) -> GrayImage:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ExamShieldError("촬영 시뮬레이션의 JPEG 압축에 실패했다.")
    return np.asarray(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)


def encode_png(image: GrayImage) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ExamShieldError("PNG 인코딩에 실패했다.")
    return bytes(encoded.tobytes())
