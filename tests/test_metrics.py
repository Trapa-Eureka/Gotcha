import pytest
from hypothesis import given
from hypothesis import strategies as st

from examshield.verify.metrics import legibility, mean, parse_answer, strip_answer_line


@given(st.text(min_size=1).filter(lambda text: text.strip() != ""))
def test_identical_text_is_fully_legible(text: str) -> None:
    assert legibility(text, text) == 1.0


@given(st.text(), st.text())
def test_legibility_stays_between_zero_and_one(left: str, right: str) -> None:
    assert 0.0 <= legibility(left, right) <= 1.0


def test_legibility_ignores_whitespace_and_width_differences() -> None:
    assert legibility("１.  물이\n끓는  온도", "1. 물이 끓는 온도") == 1.0


def test_unrelated_text_scores_low() -> None:
    assert legibility("qwerty zxcvb", "물이 끓는 온도는 몇 도인가") < 0.2


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("1. 문제\n정답: 3", 3),
        ("정답은 ②번이다.", 2),
        ("정답: 1\n다시 생각하면\n정답: 4", 4),
        ("정답 : 9", None),
        ("모르겠다", None),
        ("", None),
    ],
)
def test_parse_answer(reply: str, expected: int | None) -> None:
    assert parse_answer(reply, 4) == expected


@given(st.integers(min_value=1, max_value=4), st.text(alphabet="가나다 abc\n", max_size=40))
def test_parse_answer_reads_the_last_answer_line(number: int, prefix: str) -> None:
    assert parse_answer(f"{prefix}\n정답: {number}", 4) == number


def test_strip_answer_line_removes_only_the_answer_line() -> None:
    assert strip_answer_line("1. 문제\n① 보기\n정답: 2") == "1. 문제\n① 보기"


def test_mean_of_nothing_is_none() -> None:
    assert mean([]) is None
    assert mean([0.5, 1.0]) == 0.75
