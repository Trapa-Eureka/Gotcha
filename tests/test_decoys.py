"""답안과 미끼의 일치 분석. 정직한 응시자를 지목하지 않는 것이 가장 중요하다."""

import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from examshield.cli import app
from examshield.errors import ExamShieldError
from examshield.profiles import load_profile
from examshield.protect.pipeline import KeyEntry, protect_bytes
from examshield.trace.decoys import (
    MIN_COHORT,
    Response,
    analyse,
    format_analysis,
    read_decoys,
    read_responses,
    tail,
)

PAGES = 12
CHOICES = 4
ANSWERS = tuple((page * 7) % CHOICES + 1 for page in range(PAGES))
# 정답표를 써서 만든 복사본처럼, 미끼는 늘 정답이 아닌 번호다.
DECOYS = tuple(answer % CHOICES + 1 for answer in ANSWERS)
TABLE = {(page, None): decoy for page, decoy in enumerate(DECOYS, start=1)}
KEY = tuple(KeyEntry(page, None, answer) for page, answer in enumerate(ANSWERS, start=1))
runner = CliRunner()


def honest(name: str, rng: random.Random, skill: float) -> list[Response]:
    rows = []
    for page in range(1, PAGES + 1):
        right = ANSWERS[page - 1]
        wrong = [number for number in range(1, CHOICES + 1) if number != right]
        rows.append(Response(name, page, right if rng.random() < skill else rng.choice(wrong)))
    return rows


def cheater(name: str, follow: int = PAGES) -> list[Response]:
    return [
        Response(name, page, DECOYS[page - 1] if page <= follow else ANSWERS[page - 1])
        for page in range(1, PAGES + 1)
    ]


def cohort(seed: int, size: int = 60) -> list[Response]:
    rng = random.Random(seed)
    rows: list[Response] = []
    for index in range(size):
        rows += honest(f"H{index:03d}", rng, rng.uniform(0.2, 0.95))
    return rows


def test_tail_is_the_exact_binomial_tail() -> None:
    assert tail([0.5, 0.5], 0) == pytest.approx(1.0)
    assert tail([0.5, 0.5], 2) == pytest.approx(0.25)
    assert tail([1 / 3] * 12, 12) == pytest.approx((1 / 3) ** 12)
    assert tail([0.2, 0.5], 1) == pytest.approx(1 - 0.8 * 0.5)
    assert tail([], 0) == 1.0 and tail([0.3], 2) == 0.0


def test_someone_who_follows_every_decoy_is_flagged_and_honest_people_are_not() -> None:
    rows = cohort(1) + cheater("CHEAT")

    result = analyse(rows, TABLE, key_used=True)
    flagged = [found.examinee_id for found in result.findings if found.flagged]
    top = result.findings[0]

    assert flagged == ["CHEAT"]
    assert (top.examinee_id, top.items, top.matches) == ("CHEAT", PAGES, PAGES)
    assert top.chance < 1e-5
    assert "자기 복사본의 미끼와 강하게 일치: 1명 (CHEAT)" in format_analysis(result)
    assert "베낀 응시자도 함께 일치한다" in format_analysis(result)


@pytest.mark.parametrize("seed", range(40))
def test_no_honest_examinee_is_ever_flagged(seed: int) -> None:
    result = analyse(cohort(seed, size=80), TABLE, key_used=True)

    assert not [found for found in result.findings if found.flagged]


def test_weak_students_who_guess_are_not_flagged() -> None:
    rng = random.Random(5)
    rows: list[Response] = []
    for index in range(200):
        rows += honest(f"G{index:03d}", rng, 0.0)

    assert not [found for found in analyse(rows, TABLE, key_used=True).findings if found.flagged]


def test_a_popular_wrong_option_does_not_get_people_flagged() -> None:
    """미끼 번호가 원래 많이 고르는 오답이면, 그 번호를 골랐다는 것은 근거가 약하다."""
    rng = random.Random(9)
    rows: list[Response] = []
    for index in range(60):
        name = f"P{index:03d}"
        rows += [
            Response(name, page, DECOYS[page - 1] if rng.random() < 0.8 else ANSWERS[page - 1])
            for page in range(1, PAGES + 1)
        ]

    result = analyse(rows, TABLE, key_used=True)

    assert "다른 응시자들" in result.base
    assert not [found for found in result.findings if found.flagged]


def test_partial_followers_rank_by_how_much_they_followed() -> None:
    rows = cohort(3) + cheater("ALL") + cheater("MOST", follow=9) + cheater("SOME", follow=4)

    result = analyse(rows, TABLE, key_used=True)
    order = [found.examinee_id for found in result.findings[:3]]
    flagged = {found.examinee_id for found in result.findings if found.flagged}

    assert order[:2] == ["ALL", "MOST"]
    assert "ALL" in flagged and "SOME" not in flagged


def test_a_small_group_uses_the_fixed_bound_and_few_answers_are_never_flagged() -> None:
    rows = cheater("A") + cheater("B", follow=3)[:3]

    result = analyse(rows, TABLE, key_used=True)
    found = {entry.examinee_id: entry for entry in result.findings}

    assert len(found) < MIN_COHORT and "33%" in result.base
    assert found["A"].flagged and found["A"].chance == pytest.approx((1 / 3) ** 12)
    assert found["B"].matches == 3 and not found["B"].flagged
    assert any("5개보다 적은" in note for note in result.notes)


def test_items_where_the_decoy_equals_the_answer_are_left_out() -> None:
    table = dict(TABLE)
    table[(1, None)], table[(2, None)] = ANSWERS[0], ANSWERS[1]
    rows = [Response("TOP", page, ANSWERS[page - 1]) for page in range(1, PAGES + 1)]

    with_key = analyse(rows, table, key_used=False, answer_key=KEY)
    without = analyse(rows, table, key_used=False)

    assert with_key.dropped == 2
    assert with_key.findings[0].items == 10 and with_key.findings[0].matches == 0
    assert without.findings[0].matches == 2
    assert any("정답표" in note for note in without.notes) and not with_key.notes


def test_answers_on_pages_without_a_decoy_are_set_aside() -> None:
    result = analyse([Response("A", 99, 1), *cheater("A")], TABLE, key_used=True)

    assert result.unmatched == 1 and result.findings[0].items == PAGES


def test_per_question_decoys_are_matched_by_question_number() -> None:
    table = {(1, 11): 2, (1, 12): 4, (2, 13): 1}
    rows = [
        Response("A", 1, 2, 11),
        Response("A", 1, 4, 12),
        Response("A", 2, 1, 13),
        Response("B", 1, 4, 11),
        Response("B", 1, 2, 12),
        Response("B", 2, 3, 13),
    ]

    found = {entry.examinee_id: entry for entry in analyse(rows, table, key_used=True).findings}

    assert (found["A"].items, found["A"].matches) == (3, 3)
    assert (found["B"].items, found["B"].matches) == (3, 0)


def personal_tables(
    names: list[str], seed: int = 0
) -> dict[str, dict[tuple[int, int | None], int]]:
    """수험번호마다 다른 미끼. 정답은 늘 피한다."""
    rng = random.Random(seed)
    return {
        name: {
            (page, None): rng.choice([n for n in range(1, CHOICES + 1) if n != ANSWERS[page - 1]])
            for page in range(1, PAGES + 1)
        }
        for name in names
    }


def test_copying_from_a_cheater_is_told_apart_from_cheating() -> None:
    rows = cohort(7, size=40)
    names = sorted({entry.examinee_id for entry in rows}) + ["CHEAT", "COPIER"]
    tables = personal_tables(names)
    cheat = [Response("CHEAT", page, tables["CHEAT"][(page, None)]) for page in range(1, PAGES + 1)]
    copier = [Response("COPIER", entry.page, entry.answer) for entry in cheat]

    result = analyse(rows + cheat + copier, tables, key_used=True)
    flagged = [found.examinee_id for found in result.findings if found.flagged]
    pairs = [(found.examinee_id, found.source_id) for found in result.borrowed]
    text = format_analysis(result)

    assert result.per_examinee
    assert flagged == ["CHEAT"]
    assert pairs == [("COPIER", "CHEAT")]
    assert "COPIER의 답이 CHEAT의 미끼와 12문항 중 12문항 일치" in text
    assert "수험번호마다 다르다" in text


@pytest.mark.parametrize("seed", range(15))
def test_honest_groups_show_no_one_with_personal_decoys(seed: int) -> None:
    rows = cohort(seed, size=50)
    tables = personal_tables(sorted({entry.examinee_id for entry in rows}), seed)

    result = analyse(rows, tables, key_used=True)

    assert not [found for found in result.findings if found.flagged]
    assert result.borrowed == ()


def test_examinees_missing_from_the_mapping_table_are_reported() -> None:
    tables = personal_tables(["A"])

    result = analyse(cheater("A") + cheater("STRANGER"), tables, key_used=True)

    assert [found.examinee_id for found in result.findings] == ["A"]
    assert any("STRANGER" in note for note in result.notes)


def test_bad_inputs_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ExamShieldError, match="선택지"):
        analyse(cheater("A"), TABLE, key_used=True, choices=1)
    for name, text in {
        "empty.csv": "",
        "short.csv": "A1,3\n",
        "id.csv": "한글번호,1,2\n",
        "page.csv": "A1,0,2\n",
        "mixed.csv": "A1,1,2\nA1,1,7,2\n",
        "question.csv": "A1,1,x,2\n",
        "wide.csv": "A1,1,2,3,4\n",
    }.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ExamShieldError):
            read_responses(path)
    for name, text in {"nojob.json": "{", "nodecoy.json": '{"decoy_answers": null}'}.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ExamShieldError, match="미끼 번호"):
            read_decoys(path)


def test_answers_with_question_numbers_are_read(tmp_path: Path) -> None:
    path = tmp_path / "answers.csv"
    path.write_text("수험번호,쪽,문항,답\nA1,1,11,2\nA1,1,12,\nA2,2,13,4\n", encoding="utf-8")

    assert read_responses(path) == [Response("A1", 1, 2, 11), Response("A2", 2, 4, 13)]


def test_old_job_records_without_decoy_items_still_read(tmp_path: Path) -> None:
    job = tmp_path / "old.job.json"
    job.write_text(json.dumps({"decoy_answers": [3, 1, 2]}), encoding="utf-8")

    assert read_decoys(job) == ({(1, None): 3, (2, None): 1, (3, None): 2}, False)


def test_blank_and_unreadable_answers_count_as_no_response(tmp_path: Path) -> None:
    path = tmp_path / "answers.csv"
    path.write_text("수험번호,쪽,답\nA1,1,2\nA1,2,\nA1,3,x\nA1,4,0\nA2,1,4\n", encoding="utf-8")

    rows = read_responses(path)

    assert rows == [Response("A1", 1, 2), Response("A2", 1, 4)]


def test_cli_reads_the_job_record_of_a_real_copy(
    sample_file: Path, sample_pdf: bytes, tmp_path: Path
) -> None:
    key = tmp_path / "key.csv"
    key.write_text("1,3\n2,1\n3,2\n", encoding="utf-8")
    out = tmp_path / "out"
    made = runner.invoke(
        app,
        ["protect", str(sample_file), "--out", str(out), "--profile", "a-remap-decoy",
         "--answer-key", str(key)],
    )  # fmt: skip
    decoys = json.loads((out / "exam.job.json").read_text(encoding="utf-8"))["decoy_answers"]
    answers = tmp_path / "answers.csv"
    lines = ["수험번호,쪽,답"]
    # 세 쪽짜리 문서라 쪽마다 문항이 넷 있다고 치고 열두 답을 적는다. 여섯 문항으로는 지목하지 않는다.
    for page, decoy in enumerate(decoys, start=1):
        lines += [f"COPY,{page},{decoy}"] * 4
        lines += [f"FAIR,{page},{(3, 1, 2)[page - 1]}"] * 4
    answers.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["decoys", str(answers), "--job", str(out / "exam.job.json")])
    plain = protect_bytes(sample_pdf, load_profile("a-glyph-remap"))
    nodecoy = tmp_path / "nodecoy.job.json"
    nodecoy.write_text(json.dumps(plain.record), encoding="utf-8")
    refused = runner.invoke(app, ["decoys", str(answers), "--job", str(nodecoy)])
    missing = runner.invoke(app, ["decoys", str(tmp_path / "x.csv"), "--job", str(nodecoy)])

    assert made.exit_code == 0, made.output
    assert result.exit_code == 0, result.output
    assert "COPY" in result.stdout and "← 미끼와 강하게 일치" in result.stdout
    assert "자기 복사본의 미끼와 강하게 일치: 1명 (COPY)" in result.stdout
    assert "증명이 아니다" in result.stdout
    assert refused.exit_code == 2 and "미끼 번호를 읽지 못했다" in refused.stderr
    assert missing.exit_code == 2


def test_cli_tells_the_cheater_from_the_copier_with_per_examinee_copies(
    sample_file: Path, tmp_path: Path
) -> None:
    seed = tmp_path / "exam.seed"
    seed.write_bytes(b"0123456789abcdef-decoy-seed")
    ids = tmp_path / "ids.csv"
    names = [f"E{number:02d}" for number in range(1, 9)]
    ids.write_text("\n".join(names) + "\n", encoding="utf-8")
    key = tmp_path / "key.csv"
    # 세 쪽에 문항이 네 개씩, 문항 번호와 정답을 적는다.
    right = {question: (question % 4) + 1 for question in range(1, 13)}
    key.write_text(
        "쪽,문항,정답\n" + "".join(f"{(q - 1) // 4 + 1},{q},{a}\n" for q, a in right.items()),
        encoding="utf-8",
    )
    out, mapping = tmp_path / "out", tmp_path / "map.csv"
    made = runner.invoke(
        app,
        [
            "protect", str(sample_file), "--out", str(out), "--profile", "a-full",
            "--copies", str(ids), "--seed-file", str(seed), "--map-file", str(mapping),
            "--answer-key", str(key),
        ],
    )  # fmt: skip
    assert made.exit_code == 0, made.output
    records = {
        name: json.loads((out / f"exam-{name}.job.json").read_text(encoding="utf-8"))
        for name in names
    }
    decoys = {
        name: {entry["question"]: entry["answer"] for entry in record["decoy_items"]}
        for name, record in records.items()
    }
    lines = ["수험번호,쪽,문항,답"]
    for name in names:
        for question, answer in right.items():
            page = (question - 1) // 4 + 1
            given = decoys["E03"][question] if name in ("E03", "E07") else answer
            lines.append(f"{name},{page},{question},{given}")
    answers = tmp_path / "answers.csv"
    answers.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = runner.invoke(
        app, ["decoys", str(answers), "--jobs-dir", str(out), "--map-file", str(mapping)]
    )
    both = runner.invoke(
        app,
        ["decoys", str(answers), "--job", str(out / "exam-E01.job.json"), "--jobs-dir", str(out),
         "--map-file", str(mapping)],
    )  # fmt: skip
    half = runner.invoke(app, ["decoys", str(answers), "--jobs-dir", str(out)])

    assert len({tuple(sorted(table.items())) for table in decoys.values()}) > 1
    assert all(len(table) == 12 for table in decoys.values())
    assert all(table[q] != right[q] for table in decoys.values() for q in right)
    assert all(record["decoys_per_examinee"] for record in records.values())
    assert result.exit_code == 0, result.output
    assert "자기 복사본의 미끼와 강하게 일치: 1명 (E03)" in result.stdout
    assert "E07의 답이 E03의 미끼와 12문항 중 12문항 일치" in result.stdout
    assert both.exit_code == 2 and half.exit_code == 2
