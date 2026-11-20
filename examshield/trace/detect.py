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
    for row in rows:
        if row.output_sha256 == sha256_hex(leaked):
            return TraceResult(True, "파일 해시 일치", row.examinee_id, None, None, 0, ())
    sizes = page_sizes(original)
    cell_pt = rows[0].cell_pt
    suffix = suffix.lower()
    ids = [row.examinee_id for row in rows]
    if rows[0].kind == DOTS:
        return _trace_dots(leaked, suffix, original, sizes, cell_pt, ids, keys)
    reference = load_reference(original)
    if suffix == PDF_SUFFIX:
        return _decide(_from_pdf(leaked, reference, sizes, cell_pt), ids, keys, "PDF 렌더링")
    if suffix not in IMAGE_SUFFIXES:
        raise ExamShieldError(f"trace가 받는 형식이 아니다: {suffix} (PDF 또는 그림 파일)")
    # 내용이 같은 쪽이 여럿이면 어느 쪽인지 그림만으로는 모른다. 맞춰지는 쪽을 모두 재 보고 가장 뚜렷한 것을 쓴다.
    results = [
        _decide([found], ids, keys, "이미지 정합")
        for found in _from_image(leaked, reference, sizes, cell_pt)
    ]
    if not results:
        return _decide([], ids, keys, "이미지 정합")
    return max(results, key=lambda result: (result.found, result.score or 0.0))


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
    reference = load_reference(original, DOTS_DPI)
    if suffix == PDF_SUFFIX:
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
    results = []
    known = features(shot)
    for index, page in enumerate(reference):
        matrix = _register_photo(shot, page, known)
        if matrix is None:
            continue
        height, width = page.shape
        warped = cv2.warpPerspective(shot, matrix, (width, height), flags=cv2.INTER_LINEAR)
        covered = cv2.warpPerspective(
            np.full(shot.shape, 255, dtype=np.uint8), matrix, (width, height)
        )
        found = _read_dots(
            np.asarray(warped, dtype=np.uint8),
            np.asarray(covered) == 255,
            page,
            sizes[index],
            cell_pt,
            index,
        )
        if found is not None:
            results.append(_decide([found], ids, keys, "사진 정합"))
    if not results:
        return _decide([], ids, keys, "사진 정합")
    return max(results, key=lambda result: (result.found, result.score or 0.0))


def _register_photo(shot: Gray, page: Gray, known: Features) -> Any | None:
    """사진을 원본 쪽에 맞춘다. 기울기와 원근 왜곡까지 바로잡는다. 종이가 휜 것은 못 잡는다."""
    found = register(shot, page, known)
    return None if found is None else found[1]


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
            best: NDArray[np.float64] | None = None
            spread = -1.0
            for shift_y in range(-reach, reach + 1):
                for shift_x in range(-reach, reach + 1):
                    sample = dark[ys + shift_y, xs + shift_x]
                    variance = float(sample.var())
                    if variance > spread:
                        best, spread = sample, variance
            if best is not None:
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
    leaked: bytes, reference: list[Gray], sizes: list[tuple[float, float]], cell_pt: int
) -> list[Evidence]:
    shot = decode_photo(leaked)
    evidence = []
    for index, page in enumerate(reference):
        matched = _register(shot, page)
        if matched is None:
            continue
        height, width = page.shape
        warped = cv2.warpAffine(shot, matched[1], (width, height), flags=cv2.INTER_LINEAR)
        covered = cv2.warpAffine(
            np.full(shot.shape, 255, dtype=np.uint8),
            matched[1],
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        found = _read_cells(
            np.asarray(warped, dtype=np.uint8),
            np.asarray(covered) == 255,
            page,
            sizes[index],
            cell_pt,
            index,
        )
        if found is not None:
            evidence.append(found)
    return evidence


def _register(shot: Gray, page: Gray) -> tuple[int, Any] | None:
    """화면 캡처를 원본 쪽에 맞춘다. 이동, 확대·축소, 회전만 다룬다. 원근 왜곡(사진)은 다루지 않는다."""
    sift = getattr(cv2, "SIFT_create")(nfeatures=6000)  # noqa: B009 (cv2 타입 정보에 없다)
    shot_points, shot_features = sift.detectAndCompute(shot, None)
    page_points, page_features = sift.detectAndCompute(page, None)
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
