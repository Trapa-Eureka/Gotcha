from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from examshield.errors import ExamShieldError
from examshield.pdfio import render_gray
from examshield.protect.copies import DOTS, MapRow
from examshield.protect.pipeline import sha256_hex
from examshield.protect.techniques import dots, watermark
from examshield.verify.calibrate import (
    Features,
    decode_photo,
    features,
    open_plain_pdf,
    register,
)

TRACE_DPI = 150
# 점 무늬는 점이 작아서 더 촘촘하게 그려 놓고 읽는다.
DOTS_DPI = 200
DOTS_TILE = 8
MIN_TILE_CELLS = 8
PDF_SUFFIX = ".pdf"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
BACKGROUND_MIN = 250
MIN_CELLS = 300
MIN_SCORE = 6.0
MIN_MARGIN = 3.0
MIN_INLIERS = 20
MATCH_RATIO = 0.75
# 쪽 순서를 정할 때 쓰는 줄인 그림의 긴 변과 특징점 수.
COARSE_SIDE = 700
COARSE_FEATURES = 1500
# 워터마크가 낼 수 있는 것보다 훨씬 크게 달라진 칸은 가려진 곳(도구 막대, 낙서)이라 버린다.
MAX_CELL_CHANGE = 2 * watermark.MAX_STRENGTH + 4

Gray = NDArray[np.uint8]


@dataclass(frozen=True)
class Evidence:
    """한 쪽에서 읽어 낸 무늬 칸. values는 원본보다 어두워진 정도(평균을 뺀 값)다."""

    page_index: int
    rows: int
    cols: int
    cells: NDArray[np.intp]
    values: NDArray[np.float64]


@dataclass(frozen=True)
class TraceResult:
    found: bool
    method: str
    examinee_id: str | None
    score: float | None
    runner_up: float | None
    cells: int
    pages: tuple[int, ...]
    reason: str = ""
    suspects: tuple[str, ...] = ()


def load_reference(pdf: bytes, dpi: int = TRACE_DPI) -> list[Gray]:
    return [render_gray(page, dpi) for page in open_plain_pdf(pdf, "원본")]


class _Pages:
    """원본의 쪽을 필요할 때 그린다. 그림 한 장을 추적할 때는 대개 한두 쪽만 크게 그리면 된다."""

    def __init__(self, pdf: bytes, dpi: int) -> None:
        self._doc = open_plain_pdf(pdf, "원본")
        self._dpi = dpi

    def __len__(self) -> int:
        return int(self._doc.page_count)

    def full(self, index: int) -> Gray:
        return render_gray(self._doc[index], self._dpi)

    def small(self, index: int) -> Gray:
        """쪽 순서를 정할 때 쓰는 작은 그림. 긴 변이 COARSE_SIDE 화소쯤 되게 그린다."""
        rect = self._doc[index].rect
        dpi = max(18, min(self._dpi, round(COARSE_SIDE * 72.0 / max(rect.width, rect.height))))
        return render_gray(self._doc[index], dpi)


def page_sizes(pdf: bytes) -> list[tuple[float, float]]:
    return [(page.rect.width, page.rect.height) for page in open_plain_pdf(pdf, "원본")]


def trace(
    leaked: bytes,
    suffix: str,
    original: bytes,
    rows: list[MapRow],
    seed: bytes,
) -> TraceResult:
    keys = _keys(rows, seed)
    digest = sha256_hex(leaked)
    for row in rows:
        if row.output_sha256 == digest:
            return TraceResult(True, "파일 해시 일치", row.examinee_id, None, None, 0, ())
    sizes = page_sizes(original)
    cell_pt = rows[0].cell_pt
    suffix = suffix.lower()
    ids = [row.examinee_id for row in rows]
    if rows[0].kind == DOTS:
        return _trace_dots(leaked, suffix, original, sizes, cell_pt, ids, keys)
    if suffix == PDF_SUFFIX:
        reference = load_reference(original)
        return _decide(_from_pdf(leaked, reference, sizes, cell_pt), ids, keys, "PDF 렌더링")
    if suffix not in IMAGE_SUFFIXES:
        raise ExamShieldError(f"trace가 받는 형식이 아니다: {suffix} (PDF 또는 그림 파일)")
    pages = _Pages(original, TRACE_DPI)
    return _best_page(_from_image(leaked, pages, sizes, cell_pt), ids, keys, "이미지 정합")


def _best_page(
    candidates: Iterator[Evidence], ids: list[str], keys: list[bytes], method: str
) -> TraceResult:
    """잘 맞는 쪽부터 하나씩 읽어, 수험번호가 뚜렷하게 나오면 거기서 멈춘다.

    내용이 같은 쪽이 여럿이면 어느 쪽인지 그림만으로는 모른다. 그래서 맞춰지는 쪽을 차례로 재 본다.
    엉뚱한 쪽의 무늬로는 기준을 넘는 점수가 나오지 않으므로, 뚜렷한 답이 나오면 나머지는 재지 않는다.
    끝까지 뚜렷한 답이 없으면 그중 점수가 가장 높은 것을 돌려준다.
    """
    results = []
    for found in candidates:
        result = _decide([found], ids, keys, method)
        if result.found:
            return result
        results.append(result)
    if not results:
        return _decide([], ids, keys, method)
    return max(results, key=lambda result: result.score or 0.0)


def _keys(rows: list[MapRow], seed: bytes) -> list[bytes]:
    keys = [watermark.derive_key(seed, row.examinee_id) for row in rows]
    if any(watermark.code_of(key) != row.code for key, row in zip(keys, rows, strict=True)):
        raise ExamShieldError("시드 파일이 이 매핑표를 만들 때 쓴 것과 다르다.")
    if len({(row.cell_pt, row.kind) for row in rows}) != 1:
        raise ExamShieldError("매핑표에 무늬 종류나 칸 크기가 다른 행이 섞여 있다.")
    if rows[0].kind == DOTS:
        return [dots.examinee_key(key) for key in keys]
    return keys


def _trace_dots(
    leaked: bytes,
    suffix: str,
    original: bytes,
    sizes: list[tuple[float, float]],
    cell_pt: int,
    ids: list[str],
    keys: list[bytes],
) -> TraceResult:
    """점 무늬가 든 복사본. PDF, 화면 캡처, 인쇄물을 찍은 사진을 다룬다."""
    if suffix == PDF_SUFFIX:
        reference = load_reference(original, DOTS_DPI)
        doc = open_plain_pdf(leaked, "유출본")
        if doc.page_count != len(reference):
            raise ExamShieldError(
                f"유출본의 쪽수({doc.page_count})가 원본({len(reference)})과 다르다. "
                "쪽을 그림으로 저장해서 한 쪽씩 넣는다."
            )
        evidence = []
        for index, page in enumerate(doc):
            image = render_gray(page, DOTS_DPI)
            if image.shape != reference[index].shape:
                raise ExamShieldError(f"{index + 1}쪽의 크기가 원본과 다르다.")
            found = _read_dots(
                image,
                np.ones(image.shape, dtype=bool),
                reference[index],
                sizes[index],
                cell_pt,
                index,
            )
            if found is not None:
                evidence.append(found)
        return _decide(evidence, ids, keys, "PDF 렌더링")
    if suffix not in IMAGE_SUFFIXES:
        raise ExamShieldError(f"trace가 받는 형식이 아니다: {suffix} (PDF 또는 그림 파일)")
    shot = decode_photo(leaked)
    known = features(shot)
    pages = _Pages(original, DOTS_DPI)

    def read() -> Iterator[Evidence]:
        # 사진에 찍힌 쪽일 가능성이 높은 쪽부터 정밀하게 맞춰 본다. 맞춰 보는 일이 가장 오래 걸린다.
        for index in _likely_pages(shot, pages):
            page = pages.full(index)
            fit = register(shot, page, known)
            if fit is None:
                continue
            matrix = fit[1]
            height, width = page.shape
            warped = cv2.warpPerspective(shot, matrix, (width, height), flags=cv2.INTER_LINEAR)
            covered = cv2.warpPerspective(
                np.full(shot.shape, 255, dtype=np.uint8), matrix, (width, height)
            )
            evidence = _read_dots(
                np.asarray(warped, dtype=np.uint8),
                np.asarray(covered) == 255,
                page,
                sizes[index],
                cell_pt,
                index,
            )
            if evidence is not None:
                yield evidence

    return _best_page(read(), ids, keys, "사진 정합")


def _read_dots(
    image: Gray,
    valid: NDArray[np.bool_],
    reference: Gray,
    size_pt: tuple[float, float],
    cell_pt: int,
    page_index: int,
) -> Evidence | None:
    """빈 바탕 칸마다 칸 가운데가 둘레보다 얼마나 어두운지 잰다.

    둘레와 견주므로 조명이 고르지 않아도 된다. 정합이 한두 픽셀 어긋나는 것은 구역마다
    칸 가운데를 조금씩 옮겨 보며 점이 가장 뚜렷한 자리를 골라 바로잡는다. 점은 칸 가운데에만
    있으므로 이 자리는 수험번호를 몰라도 정해진다.
    """
    rows, cols = watermark.grid_shape(size_pt[0], size_pt[1], cell_pt)
    step = cell_pt * DOTS_DPI / 72.0
    window = round(step) | 1
    radius = max(1, round(DOTS_DPI / 72.0 / dots.PX_PER_PT))
    picture = image.astype(np.float64)
    dark = cv2.blur(picture, (window, window)) - cv2.blur(picture, (2 * radius + 1, 2 * radius + 1))
    usable = (reference >= BACKGROUND_MIN) & valid
    blank = cv2.erode(usable.astype(np.uint8), np.ones((window, window), dtype=np.uint8)) > 0
    height, width = reference.shape
    reach = int(step // 2)
    offsets = np.arange(-reach, reach + 1, dtype=np.intp)
    shifts_y, shifts_x = (grid.ravel() for grid in np.meshgrid(offsets, offsets, indexing="ij"))
    cells: list[NDArray[np.intp]] = []
    values: list[NDArray[np.float64]] = []
    for top in range(0, rows, DOTS_TILE):
        for left in range(0, cols, DOTS_TILE):
            grid_rows, grid_cols = np.meshgrid(
                np.arange(top, min(top + DOTS_TILE, rows)),
                np.arange(left, min(left + DOTS_TILE, cols)),
                indexing="ij",
            )
            ys = np.round((grid_rows + 0.5) * step).astype(np.intp).ravel()
            xs = np.round((grid_cols + 0.5) * step).astype(np.intp).ravel()
            numbers = (grid_rows * cols + grid_cols).astype(np.intp).ravel()
            inside = (ys >= reach) & (xs >= reach) & (ys + reach < height) & (xs + reach < width)
            ys, xs, numbers = ys[inside], xs[inside], numbers[inside]
            keep = blank[ys, xs]
            ys, xs, numbers = ys[keep], xs[keep], numbers[keep]
            if len(numbers) < MIN_TILE_CELLS:
                continue
            # 옮겨 볼 자리마다 한 줄. 점이 가장 뚜렷한(값이 가장 고르지 않은) 자리를 고른다.
            samples = dark[ys[None, :] + shifts_y[:, None], xs[None, :] + shifts_x[:, None]]
            best = samples[int(np.argmax(samples.var(axis=1)))]
            cells.append(numbers)
            values.append(best - best.mean())
    if sum(len(entry) for entry in cells) < MIN_CELLS // 4:
        return None
    measured = np.concatenate(values)
    # 낙서나 가려진 곳처럼 점보다 훨씬 크게 튀는 칸이 점수를 끌고 가지 않게 한다.
    limit = 4.0 * float(np.median(np.abs(measured))) + 1e-9
    return Evidence(
        page_index=page_index,
        rows=rows,
        cols=cols,
        cells=np.concatenate(cells),
        values=np.clip(measured, -limit, limit),
    )


def _from_pdf(
    leaked: bytes, reference: list[Gray], sizes: list[tuple[float, float]], cell_pt: int
) -> list[Evidence]:
    doc = open_plain_pdf(leaked, "유출본")
    if doc.page_count != len(reference):
        raise ExamShieldError(
            f"유출본의 쪽수({doc.page_count})가 원본({len(reference)})과 다르다. "
            "쪽을 그림으로 저장해서 한 쪽씩 넣는다."
        )
    evidence = []
    for index, page in enumerate(doc):
        image = render_gray(page, TRACE_DPI)
        if image.shape != reference[index].shape:
            raise ExamShieldError(f"{index + 1}쪽의 크기가 원본과 다르다.")
        valid = np.ones(image.shape, dtype=bool)
        found = _read_cells(image, valid, reference[index], sizes[index], cell_pt, index)
        if found is not None:
            evidence.append(found)
    return evidence


def _from_image(
    leaked: bytes, pages: _Pages, sizes: list[tuple[float, float]], cell_pt: int
) -> Iterator[Evidence]:
    """캡처가 맞춰지는 쪽마다 읽어 낸 칸. 캡처에 찍힌 쪽일 가능성이 높은 쪽부터 내놓는다."""
    shot = decode_photo(leaked)
    known = _features(shot)
    for index in _likely_pages(shot, pages):
        page = pages.full(index)
        found = _register(known, page)
        if found is None:
            continue
        transform = found[1]
        height, width = page.shape
        warped = cv2.warpAffine(shot, transform, (width, height), flags=cv2.INTER_LINEAR)
        covered = cv2.warpAffine(
            np.full(shot.shape, 255, dtype=np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        found_cells = _read_cells(
            np.asarray(warped, dtype=np.uint8),
            np.asarray(covered) == 255,
            page,
            sizes[index],
            cell_pt,
            index,
        )
        if found_cells is not None:
            yield found_cells


def _small(image: Gray) -> Gray:
    scale = COARSE_SIDE / max(image.shape)
    if scale >= 1.0:
        return image
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def _likely_pages(shot: Gray, pages: _Pages) -> list[int]:
    """모든 쪽 번호를, 그림에 찍힌 쪽일 가능성이 높은 순서로 돌려준다.

    줄인 그림끼리 특징점을 견줘 순서만 정한다. 쪽을 빼지는 않는다. 순서가 틀려도 결과는 같고
    시간만 더 든다. 쪽마다 정밀하게 맞춰 보는 것보다 훨씬 빠르다.
    """
    if len(pages) <= 1:
        return list(range(len(pages)))
    sift = getattr(cv2, "SIFT_create")(nfeatures=COARSE_FEATURES)  # noqa: B009
    shot_points, shot_described = sift.detectAndCompute(_small(shot), None)
    scores = []
    for index in range(len(pages)):
        score = 0
        page_points, page_described = sift.detectAndCompute(pages.small(index), None)
        usable = shot_described is not None and page_described is not None
        if usable and len(shot_points) >= 2 and len(page_points) >= 2:
            pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(shot_described, page_described, k=2)
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
            ]
            score = len(good)
            if len(good) >= 4:
                source = np.array([shot_points[m.queryIdx].pt for m in good], dtype=np.float32)
                target = np.array([page_points[m.trainIdx].pt for m in good], dtype=np.float32)
                _, inliers = cv2.estimateAffinePartial2D(
                    source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                # 자리까지 맞는 짝을 훨씬 무겁게 친다.
                score += 100 * (int(inliers.sum()) if inliers is not None else 0)
        scores.append((-score, index))
    return [index for _, index in sorted(scores)]


def _features(image: Gray) -> Features:
    sift = getattr(cv2, "SIFT_create")(nfeatures=6000)  # noqa: B009 (cv2 타입 정보에 없다)
    points, described = sift.detectAndCompute(image, None)
    return points, described


def _register(shot: Features, page: Gray) -> tuple[int, Any] | None:
    """화면 캡처를 원본 쪽에 맞춘다. 이동, 확대·축소, 회전만 다룬다. 원근 왜곡(사진)은 다루지 않는다.

    캡처의 특징점은 한 번만 구해서 쪽마다 돌려쓴다.
    """
    shot_points, shot_features = shot
    page_points, page_features = _features(page)
    if shot_features is None or page_features is None or len(shot_points) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(shot_features, page_features, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
    ]
    if len(good) < MIN_INLIERS:
        return None
    source = np.array([shot_points[match.queryIdx].pt for match in good], dtype=np.float32)
    target = np.array([page_points[match.trainIdx].pt for match in good], dtype=np.float32)
    transform, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    count = int(inliers.sum()) if inliers is not None else 0
    if transform is None or count < MIN_INLIERS:
        return None
    return count, transform


def _read_cells(
    image: Gray,
    valid: NDArray[np.bool_],
    reference: Gray,
    size_pt: tuple[float, float],
    cell_pt: int,
    page_index: int,
) -> Evidence | None:
    rows, cols = watermark.grid_shape(size_pt[0], size_pt[1], cell_pt)
    scale = TRACE_DPI / 72.0
    step = cell_pt * scale
    inset = step * 0.25
    darker = reference.astype(np.float64) - image.astype(np.float64)
    cells: list[int] = []
    values: list[float] = []
    height, width = reference.shape
    for row in range(rows):
        top, bottom = round(row * step + inset), round((row + 1) * step - inset)
        if bottom > height or bottom <= top:
            continue
        for col in range(cols):
            left, right = round(col * step + inset), round((col + 1) * step - inset)
            if right > width or right <= left:
                continue
            # 글자가 없는 빈 바탕 칸만 쓴다. 글자 가장자리는 캡처마다 달라서 잡음이 크다.
            if int(np.min(reference[top:bottom, left:right])) < BACKGROUND_MIN:
                continue
            if not valid[top:bottom, left:right].all():
                continue
            change = float(darker[top:bottom, left:right].mean())
            if abs(change) > MAX_CELL_CHANGE:
                continue
            cells.append(row * cols + col)
            values.append(change)
    if len(cells) < MIN_CELLS // 4:
        return None
    measured = np.array(values, dtype=np.float64)
    return Evidence(
        page_index=page_index,
        rows=rows,
        cols=cols,
        cells=np.array(cells, dtype=np.intp),
        values=measured - measured.mean(),
    )


def _decide(
    evidence: list[Evidence], ids: list[str], keys: list[bytes], method: str
) -> TraceResult:
    total = sum(len(entry.cells) for entry in evidence)
    pages = tuple(entry.page_index + 1 for entry in evidence)
    if total < MIN_CELLS:
        return TraceResult(
            False,
            method,
            None,
            None,
            None,
            total,
            pages,
            "원본과 맞춰 볼 수 있는 빈 바탕이 너무 적다.",
        )
    values = np.concatenate([entry.values for entry in evidence])
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return TraceResult(
            False,
            method,
            None,
            None,
            None,
            total,
            pages,
            "원본과 차이가 없다. 워터마크가 없는 문서다.",
        )
    scores = np.empty(len(keys), dtype=np.float64)
    for position, key in enumerate(keys):
        signs = np.concatenate(
            [
                watermark.pattern(key, entry.page_index, entry.rows, entry.cols).ravel()[
                    entry.cells
                ]
                for entry in evidence
            ]
        )
        # 무늬와 무관한 문서라면 이 값은 평균 0, 표준편차 1에 가깝다.
        scores[position] = float(np.where(signs, 1.0, -1.0) @ values) / norm
    order = np.argsort(scores)[::-1]
    best = float(scores[order[0]])
    second = float(scores[order[1]]) if len(order) > 1 else None
    suspects = tuple(ids[int(position)] for position in order if scores[position] >= MIN_SCORE)
    clear = len(suspects) == 1 and (second is None or best - second >= MIN_MARGIN)
    if clear:
        reason = ""
    elif len(suspects) > 1:
        reason = (
            "기준을 넘는 수험번호가 여럿이다: "
            + ", ".join(suspects[:10])
            + ". 여러 복사본을 겹쳐 만든 유출본일 수 있다."
        )
    else:
        reason = "어느 수험번호의 무늬와도 뚜렷하게 맞지 않는다."
    return TraceResult(
        found=clear,
        method=method,
        examinee_id=ids[int(order[0])] if clear else None,
        score=round(best, 2),
        runner_up=round(second, 2) if second is not None else None,
        cells=total,
        pages=pages,
        reason=reason,
        suspects=suspects,
    )


def format_trace(result: TraceResult) -> str:
    lines = [f"방법         {result.method}"]
    if result.score is not None:
        second = "-" if result.runner_up is None else f"{result.runner_up:.2f}"
        lines.append(
            f"일치 점수    {result.score:.2f} (2위 {second}, 기준 {MIN_SCORE:.0f} 이상이고 2위와 "
            f"{MIN_MARGIN:.0f} 이상 차이)"
        )
        lines.append(f"비교한 칸    {result.cells}칸, 쪽 {', '.join(map(str, result.pages))}")
    if result.found:
        lines.append(f"수험번호     {result.examinee_id}")
    else:
        lines.append(f"결과         찾지 못했다. {result.reason}")
    return "\n".join(lines)


def suffix_of(path: Path) -> str:
    return path.suffix.lower()
