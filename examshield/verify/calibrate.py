"""실제로 인쇄해서 찍은 사진을 촬영 시뮬레이션과 견준다(명세 11절 5단계).

같은 쪽을 프로파일의 촬영 조건으로 시뮬레이션한 그림과 실제 사진에서 같은 값을 재서 나란히 놓는다.
인쇄와 촬영은 사람이 한다. 이 모듈은 사진을 받아 재기만 한다.
"""

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from examshield.check.compare import OVERLAY_DRAW, Ocr
from examshield.errors import ExamShieldError
from examshield.pdfio import GrayImage, open_pdf_bytes, page_text, render_gray
from examshield.verify.capture_sim import CaptureConditions, encode_png, simulate_capture
from examshield.verify.metrics import legibility

CALIBRATE_DPI = 200
BACKGROUND_MIN = 250
MIN_INLIERS = 20
MATCH_RATIO = 0.75
MAX_PHOTO_SIDE = 3600
MIN_PHOTO_SIDE = 16
# 글자가 빽빽한 쪽은 특징점이 수만 개다. 적게 자르면 사진과 원본에서 서로 다른 점이 남아 맞지 않는다.
MAX_FEATURES = 40000
MIN_PAGE_WIDTH_PX = 300
BLUR_STEPS = tuple(round(0.2 * step, 1) for step in range(0, 31))


@dataclass(frozen=True)
class Measured:
    """사진 한 장에서 잰 촬영 조건. 길이는 사진의 픽셀 단위다."""

    page_index: int
    inliers: int
    page_width_px: float
    skew_degrees: float
    perspective: float
    light_falloff: float
    blur_sigma: float
    noise_sigma: float
    ocr_legibility: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_index + 1,
            "page_width_px": round(self.page_width_px, 1),
            "skew_degrees": round(self.skew_degrees, 2),
            "perspective": round(self.perspective, 4),
            "light_falloff": round(self.light_falloff, 3),
            "blur_sigma": round(self.blur_sigma, 2),
            "noise_sigma": round(self.noise_sigma, 2),
            "ocr_legibility": None
            if self.ocr_legibility is None
            else round(self.ocr_legibility, 4),
        }


def decode_photo(data: bytes) -> GrayImage:
    try:
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise ExamShieldError("사진을 그림으로 열지 못했다.") from error
    if decoded is None:
        raise ExamShieldError("사진을 그림으로 열지 못했다.")
    photo = np.asarray(decoded, dtype=np.uint8)
    if min(photo.shape) < MIN_PHOTO_SIDE:
        raise ExamShieldError(f"그림이 너무 작다. 짧은 변이 {MIN_PHOTO_SIDE}픽셀 이상이어야 한다.")
    longest = max(photo.shape)
    if longest > MAX_PHOTO_SIDE:
        scale = MAX_PHOTO_SIDE / longest
        size = (max(1, round(photo.shape[1] * scale)), max(1, round(photo.shape[0] * scale)))
        photo = np.asarray(cv2.resize(photo, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    return photo


def open_plain_pdf(pdf: bytes, label: str) -> Any:
    """암호가 없고 쪽이 있는 PDF만 연다."""
    try:
        doc = open_pdf_bytes(pdf)
    except Exception as error:
        raise ExamShieldError(f"{label}을(를) PDF로 열지 못했다.") from error
    if not doc.is_pdf:
        raise ExamShieldError(f"{label}이(가) PDF가 아니다.")
    if doc.needs_pass or doc.is_encrypted:
        raise ExamShieldError(f"{label}에 암호가 걸려 있다. 암호가 걸린 PDF는 처리하지 않는다.")
    if doc.page_count == 0:
        raise ExamShieldError(f"{label}에 쪽이 없다.")
    return doc


Features = tuple[Any, Any]


def features(image: GrayImage) -> Features:
    """그림의 특징점과 그 묘사값. 한 사진을 여러 쪽과 맞춰 볼 때 한 번만 구해 돌려쓴다."""
    sift = getattr(cv2, "SIFT_create")(nfeatures=MAX_FEATURES)  # noqa: B009 (cv2 타입 정보에 없다)
    points, described = sift.detectAndCompute(image, None)
    return points, described


def register(
    photo: GrayImage, page: GrayImage, known: Features | None = None
) -> tuple[int, NDArray[np.float64]] | None:
    """사진의 점을 원본 쪽의 점으로 옮기는 변환(원근 포함)과 맞은 특징점 수.

    사진이 원본 쪽보다 작으면 글자의 특징이 다른 크기에서 잡혀 잘 맞지 않는다. 그래서 그대로
    맞춰 보고, 안 되면 사진을 원본 쪽 크기로 키워서 다시 맞춘다.
    """
    page_features = features(page)
    found = _match(known or features(photo), page_features)
    if found is not None and not _plausible(found[1], page.shape):
        found = None
    scale = max(page.shape) / max(photo.shape)
    if found is None and scale > 1.1:
        size = (round(photo.shape[1] * scale), round(photo.shape[0] * scale))
        larger = np.asarray(cv2.resize(photo, size, interpolation=cv2.INTER_CUBIC), dtype=np.uint8)
        enlarged = _match(features(larger), page_features)
        if enlarged is not None:
            shrink = np.diag([size[0] / photo.shape[1], size[1] / photo.shape[0], 1.0])
            found = (enlarged[0], enlarged[1] @ shrink)
            if not _plausible(found[1], page.shape):
                found = None
    return found


def _match(photo: Features, page: Features) -> tuple[int, NDArray[np.float64]] | None:
    photo_points, photo_features = photo
    page_points, page_features = page
    if photo_features is None or page_features is None or len(photo_points) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(photo_features, page_features, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
    ]
    if len(good) < MIN_INLIERS:
        return None
    source = np.array([photo_points[match.queryIdx].pt for match in good], dtype=np.float32)
    target = np.array([page_points[match.trainIdx].pt for match in good], dtype=np.float32)
    # 글자가 빽빽한 쪽은 비슷한 획이 많아 잘못 짝지어진 점이 대부분일 수 있다. 넉넉히 시도한다.
    matrix, inliers = cv2.findHomography(
        source, target, cv2.USAC_MAGSAC, 3.0, maxIters=20000, confidence=0.999
    )
    count = int(inliers.sum()) if inliers is not None else 0
    if matrix is None or count < MIN_INLIERS:
        return None
    return count, np.asarray(matrix, dtype=np.float64)


def _plausible(matrix: NDArray[np.float64], shape: tuple[int, ...]) -> bool:
    """쪽을 한 점으로 찌그러뜨리는 변환을 거른다. 되풀이되는 무늬끼리 잘못 맞으면 이런 변환이 나온다."""
    height, width = shape[0], shape[1]
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        return False
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    seen = cv2.perspectiveTransform(corners[None, :, :], np.linalg.inv(matrix))[0]
    if not np.all(np.isfinite(seen)) or not cv2.isContourConvex(seen.astype(np.float32)):
        return False
    return float(np.linalg.norm(seen[1] - seen[0])) >= MIN_PAGE_WIDTH_PX


def without_overlay(doc: Any) -> Any:
    """쪽 위에 덧씌운 무늬 그림을 뺀 문서. 사진을 맞추는 기준으로 쓴다.

    점 무늬는 같은 모양이 되풀이돼서, 남겨 두면 엉뚱한 자리끼리 맞는다.
    """
    for page in doc:
        contents = page.get_contents()
        if contents and OVERLAY_DRAW.fullmatch(doc.xref_stream(contents[-1])):
            kept = " ".join(f"{xref} 0 R" for xref in contents[:-1])
            doc.xref_set_key(page.xref, "Contents", f"[{kept}]")
    return doc


def measure(
    photo: GrayImage,
    pages: list[GrayImage],
    texts: list[str],
    ocr: Ocr | None = None,
    page_index: int | None = None,
) -> Measured:
    """사진이 어느 쪽인지 찾고(page_index를 주면 그 쪽으로) 촬영 조건을 잰다."""
    candidates = range(len(pages)) if page_index is None else [page_index]
    best: tuple[int, int, NDArray[np.float64]] | None = None
    known = features(photo)
    for index in candidates:
        if not 0 <= index < len(pages):
            raise ExamShieldError(f"문서에 {index + 1}쪽이 없다.")
        found = register(photo, pages[index], known)
        if found is not None and (best is None or found[0] > best[0]):
            best = (found[0], index, found[1])
    if best is None:
        raise ExamShieldError(
            "사진을 문서의 어느 쪽에도 맞추지 못했다. 쪽 전체가 또렷하게 나온 사진인지 확인한다."
        )
    inliers, index, matrix = best
    page = pages[index]
    height, width = page.shape
    back = np.linalg.inv(matrix)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    seen = cv2.perspectiveTransform(corners[None, :, :], back)[0]
    top, bottom = seen[1] - seen[0], seen[2] - seen[3]
    page_width = float((np.linalg.norm(top) + np.linalg.norm(bottom)) / 2)
    skew = math.degrees(math.atan2(float(top[1]), float(top[0])))
    # 이동, 회전, 확대·축소만으로 설명되지 않고 남는 모서리의 어긋남이 원근 왜곡이다.
    plain, _ = cv2.estimateAffinePartial2D(
        corners.astype(np.float32), seen.astype(np.float32), method=cv2.LMEDS
    )
    ideal = cv2.transform(corners[None, :, :], plain)[0] if plain is not None else seen
    perspective = float(np.max(np.linalg.norm(seen - ideal, axis=1)) / page_width)

    flat = np.asarray(
        cv2.warpPerspective(photo, matrix, (width, height), flags=cv2.INTER_LINEAR),
        dtype=np.uint8,
    )
    covered = (
        np.asarray(
            cv2.warpPerspective(np.full(photo.shape, 255, dtype=np.uint8), matrix, (width, height))
        )
        == 255
    )
    window = max(15, width // 40) | 1
    blank = (
        cv2.erode(
            ((page >= BACKGROUND_MIN) & covered).astype(np.uint8),
            np.ones((window, window), dtype=np.uint8),
        )
        > 0
    )
    if int(blank.sum()) < 1000:
        raise ExamShieldError("사진에 빈 바탕이 너무 적어 조명과 잡음을 재지 못했다.")
    smooth = np.asarray(cv2.blur(flat.astype(np.float64), (window, window)), dtype=np.float64)
    low, high = np.percentile(smooth[blank], [2, 98])
    falloff = float(1.0 - low / high) if high > 0 else 0.0

    scale = page_width / width
    return Measured(
        page_index=index,
        inliers=inliers,
        page_width_px=page_width,
        skew_degrees=skew,
        perspective=perspective,
        light_falloff=falloff,
        blur_sigma=_blur(flat, page, covered, smooth) * scale,
        noise_sigma=_noise(photo, back, blank),
        ocr_legibility=legibility(ocr(encode_png(flat)), texts[index]) if ocr else None,
    )


def _blur(
    flat: GrayImage, page: GrayImage, covered: NDArray[np.bool_], light: NDArray[np.float64]
) -> float:
    """원본 쪽을 여러 세기로 흐려 보고 사진과 가장 비슷한 세기를 고른다. 원본 쪽의 픽셀 단위."""
    near_ink = cv2.dilate((page < 128).astype(np.uint8), np.ones((9, 9), dtype=np.uint8)) > 0
    area = near_ink & covered
    if int(area.sum()) < 500:
        return 0.0
    # 조명을 나눠 없애고 0(잉크)~1(종이)로 맞춘 뒤 비교한다.
    seen = np.clip(flat.astype(np.float64) / np.maximum(light * 1.02, 1.0), 0.0, 1.0)[area]
    clean = page.astype(np.float64) / 255.0
    best, error = 0.0, math.inf
    for sigma in BLUR_STEPS:
        blurred = clean if sigma == 0 else cv2.GaussianBlur(clean, (0, 0), sigma)
        model = blurred[area]
        # 사진의 잉크는 완전히 검지 않다. 밝기 범위를 맞추는 일차식을 먼저 구한다.
        gain, offset = np.polyfit(model, seen, 1)
        miss = float(np.mean((gain * model + offset - seen) ** 2))
        if miss < error:
            best, error = sigma, miss
    return best


def _noise(photo: GrayImage, back: NDArray[np.float64], blank: NDArray[np.bool_]) -> float:
    """빈 바탕에서 이웃 픽셀과의 차이로 잡음을 잰다. 사진을 펴지 않고 원래 픽셀에서 잰다."""
    height, width = photo.shape
    area = (
        cv2.warpPerspective(
            blank.astype(np.uint8) * 255, back, (width, height), flags=cv2.INTER_NEAREST
        )
        == 255
    )
    picture = photo.astype(np.float64)
    residual = picture - cv2.GaussianBlur(picture, (0, 0), 2.0)
    values = residual[cv2.erode(area.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)) > 0]
    if values.size < 500:
        return 0.0
    # 중앙값 기준 편차는 점 무늬나 먼지처럼 드문 값에 끌려가지 않는다.
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def reference_pages(pdf: bytes) -> tuple[list[GrayImage], list[str]]:
    doc = open_plain_pdf(pdf, "인쇄한 문서")
    texts = [page_text(page) for page in doc]
    return [render_gray(page, CALIBRATE_DPI) for page in without_overlay(doc)], texts


def simulated(
    pdf: bytes, page_index: int, conditions: CaptureConditions, ocr: Ocr | None = None
) -> Measured:
    """같은 쪽을 시뮬레이션으로 찍은 그림에서 같은 값을 잰다."""
    pages, texts = reference_pages(pdf)
    printed = render_gray(open_plain_pdf(pdf, "인쇄한 문서")[page_index], conditions.print_dpi)
    return measure(simulate_capture(printed, conditions), pages, texts, ocr, page_index)


def suggest(real: Measured, conditions: CaptureConditions) -> dict[str, Any]:
    """실제 사진에 맞춘 촬영 조건. jpeg_quality와 seed는 사진에서 알 수 없어 그대로 둔다."""
    # 시뮬레이션의 조명은 가로로 s, 세로로 s/2만큼 어두워진다: 1 - f = (1 - s)(1 - s/2)
    falloff = min(max(real.light_falloff, 0.0), 0.95)
    light = (3 - math.sqrt(1 + 8 * (1 - falloff))) / 2
    return {
        "print_dpi": conditions.print_dpi,
        "capture_width": round(real.page_width_px),
        "skew_degrees": round(abs(real.skew_degrees), 1),
        "perspective": round(real.perspective, 3),
        "illumination": round(light, 2),
        "blur_sigma": round(real.blur_sigma, 1),
        "noise_sigma": round(real.noise_sigma, 1),
        "jpeg_quality": conditions.jpeg_quality,
        "seed": conditions.seed,
    }


ROWS = (
    ("쪽 너비(px)", "page_width_px", "{:.0f}"),
    ("기울기(도)", "skew_degrees", "{:.1f}"),
    ("원근 왜곡", "perspective", "{:.3f}"),
    ("조명 차이", "light_falloff", "{:.2f}"),
    ("흐림(px)", "blur_sigma", "{:.1f}"),
    ("잡음", "noise_sigma", "{:.1f}"),
    ("OCR 판독률", "ocr_legibility", "{:.1%}"),
)


def format_comparison(real: Measured, model: Measured) -> str:
    lines = [f"쪽           {real.page_index + 1} (맞은 특징점 {real.inliers}개)"]
    lines.append(f"{'항목':<10}  {'실제 사진':>10}  {'시뮬레이션':>10}")
    for label, name, pattern in ROWS:
        left, right = getattr(real, name), getattr(model, name)
        cells = ["-" if value is None else pattern.format(value) for value in (left, right)]
        lines.append(f"{label:<10}  {cells[0]:>10}  {cells[1]:>10}")
    return "\n".join(lines)
