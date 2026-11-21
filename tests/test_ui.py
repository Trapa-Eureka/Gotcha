"""로컬 웹 화면. 실제 HTTP로 붙어 본다. 127.0.0.1 밖으로는 나가지 않는다."""

import base64
import http.client
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.cli import app
from examshield.pdfio import open_pdf_bytes, save_deterministic
from examshield.protect import batch
from examshield.ui.server import HOST, TOKEN_HEADER, Service, make_server, safe_name
from examshield.verify.capture_sim import encode_png


class Client:
    def __init__(self, address: str) -> None:
        self.port = int(address.split(":")[2].split("/")[0])
        self.token = address.split("?t=")[1]

    def send(
        self,
        method: str,
        path: str,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        raw: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        body = raw if raw is not None else (json.dumps(payload).encode() if payload else None)
        connection = http.client.HTTPConnection(HOST, self.port, timeout=60)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, path: str, payload: Any, **extra: str) -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json", TOKEN_HEADER: self.token, **extra}
        status, _, body = self.send("POST", path, payload, headers)
        answer = json.loads(body)
        if status != 202:
            return status, answer
        # 오래 걸리는 일은 작업 번호만 돌아온다. 끝날 때까지 물어서 예전처럼 결과를 돌려준다.
        final = self.wait(answer["job"])
        if final["state"] == "failed":
            return 400, {"error": final["error"]}
        return 200, final["result"]

    def job(self, job_id: str = "") -> tuple[int, dict[str, Any]]:
        query = f"/api/job?t={self.token}" + (f"&id={job_id}" if job_id else "")
        status, _, body = self.send("GET", query)
        return status, json.loads(body)

    def wait(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status, state = self.job(job_id)
            assert status == 200, state
            if state["state"] != "running":
                return state
            time.sleep(0.05)
        raise AssertionError("작업이 끝나지 않았다.")

    def start(self, path: str, payload: Any) -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json", TOKEN_HEADER: self.token}
        status, _, body = self.send("POST", path, payload, headers)
        return status, json.loads(body)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    server, address = make_server(
        tmp_path / "work", reports=tmp_path / "reports", records=tmp_path / "records"
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Client(address)
    server.shutdown()
    server.server_close()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_the_page_is_served_only_with_the_token(client: Client) -> None:
    status, headers, body = client.send("GET", f"/?t={client.token}")
    without, _, _ = client.send("GET", "/")
    wrong, _, _ = client.send("GET", "/?t=guess")

    assert status == 200 and "복사본 만들기" in body.decode("utf-8")
    assert client.token in body.decode("utf-8") and "__TOKEN__" not in body.decode("utf-8")
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Frame-Options"] == "DENY" and headers["Cache-Control"] == "no-store"
    assert without == wrong == 403


def test_the_page_loads_nothing_from_outside() -> None:
    page = (Path(__file__).resolve().parent.parent / "examshield/ui/page.html").read_text("utf-8")

    assert (
        "http://" not in page and "https://" not in page and "//" not in page.split("<script>")[0]
    )
    assert "innerHTML" not in page


def test_profiles_are_listed_with_their_verification_state(client: Client) -> None:
    status, _, body = client.send("GET", f"/api/profiles?t={client.token}")
    listed = {entry["name"]: entry for entry in json.loads(body)["profiles"]}

    assert status == 200
    assert {"none", "a-remap-decoy", "a-full", "b-dots"} <= set(listed)
    assert listed["a-remap-decoy"]["verification"] == "검증 안 됨"
    assert any("검증한 기록이 없다" in warning for warning in listed["a-remap-decoy"]["warnings"])
    assert listed["a-full"]["needs_copies"] and listed["b-dots"]["per_examinee"]
    assert not listed["b-dots"]["needs_copies"]


def test_protect_and_check_through_the_browser_api(client: Client, sample_pdf: bytes) -> None:
    status, made = client.post(
        "/api/protect", {"name": "시험지.pdf", "file": b64(sample_pdf), "profile": "a-remap-decoy"}
    )
    copy = base64.b64decode(made["pdf"])
    checked = client.post(
        "/api/check",
        {"name": "시험지.pdf", "file": b64(sample_pdf), "copy": made["pdf"], "job": made["job"]},
    )
    without_job = client.post(
        "/api/check", {"name": "시험지.pdf", "file": b64(sample_pdf), "copy": made["pdf"]}
    )
    changed = open_pdf_bytes(copy)
    changed[0].insert_text((100, 500), "3", fontsize=9)
    tampered = client.post(
        "/api/check",
        {"name": "x.pdf", "file": b64(sample_pdf), "copy": b64(save_deterministic(changed))},
    )

    assert status == 200 and made["passed"]
    assert made["file_name"] == "시험지.a-remap-decoy.pdf"
    assert copy.startswith(b"%PDF") and copy != sample_pdf
    assert json.loads(made["job"])["profile"]["name"] == "a-remap-decoy"
    assert checked == (200, {"passed": True, "summary": checked[1]["summary"]})
    assert not without_job[1]["passed"] and not tampered[1]["passed"]


def test_nothing_is_left_on_disk_for_a_pdf(
    client: Client, sample_pdf: bytes, tmp_path: Path
) -> None:
    status, made = client.post(
        "/api/protect", {"name": "a.pdf", "file": b64(sample_pdf), "profile": "b-dots"}
    )
    checked = client.post(
        "/api/check",
        {"name": "a.pdf", "file": b64(sample_pdf), "copy": made["pdf"], "job": made["job"]},
    )

    assert status == 200 and made["passed"] and checked[1]["passed"]
    assert not (tmp_path / "work").exists()


def test_hangul_documents_go_through_the_converter_and_carry_the_notice(
    client: Client, sample_pdf: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned = tmp_path / "canned.pdf"
    canned.write_bytes(sample_pdf)
    tools = tmp_path / "tools"
    tools.mkdir()
    fake = tools / "soffice"
    fake.write_text(
        "#!/bin/bash\n"
        'while [ $# -gt 0 ]; do [ "$1" = "--outdir" ] && out="$2"; shift; done\n'
        f'cp "{canned}" "$out/input.pdf"\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")

    status, made = client.post(
        "/api/protect", {"name": "시험지.hwp", "file": b64(b"hwp bytes"), "profile": "a-decoy"}
    )

    assert status == 200 and made["passed"], made
    assert any("한글 문서" in note for note in made["notes"])
    assert list((tmp_path / "work").iterdir()) == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"name": "a.exe", "file": "AAAA", "profile": "none"}, "지원하지 않는 파일 형식"),
        ({"name": "a.pdf", "file": "AAAA", "profile": "no-such"}, "찾지 못했다"),
        ({"name": "a.pdf", "file": "AAAA", "profile": "../../x"}, "찾지 못했다"),
        ({"name": "a.pdf", "file": "AAAA", "profile": "none"}, "PDF를 열지 못했다"),
        ({"name": "a.pdf", "file": "###", "profile": "none"}, "읽지 못했다"),
        ({"name": "a.pdf", "profile": "none"}, "없다"),
        ({"name": "", "file": "AAAA", "profile": "none"}, "파일 이름"),
        ({"name": "a.pdf", "file": "AAAA", "profile": ["none"]}, "프로파일"),
        ({"name": "a.docx", "file": "AAAA", "profile": "none"}, "LibreOffice"),
    ],
)
def test_bad_requests_get_a_plain_message(
    payload: dict[str, Any],
    message: str,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    status, answer = client.post("/api/protect", payload)

    assert status == 400
    assert message in answer["error"]


def test_watermark_profiles_are_sent_to_the_copies_section(
    client: Client, sample_pdf: bytes
) -> None:
    status, answer = client.post(
        "/api/protect",
        {"name": "exam.pdf", "file": base64.b64encode(sample_pdf).decode(), "profile": "a-full"},
    )

    assert status == 400 and "수험번호별 복사본" in answer["error"]


def test_requests_from_elsewhere_are_refused(client: Client) -> None:
    payload = {"name": "a.pdf", "file": "AAAA", "profile": "none"}
    body = json.dumps(payload).encode()
    good = {"Content-Type": "application/json", TOKEN_HEADER: client.token}

    cases = {
        "no token": {"Content-Type": "application/json"},
        "wrong token": {**good, TOKEN_HEADER: "guess"},
        "other origin": {**good, "Origin": "https://evil.example"},
        "rebound host": {**good, "Host": "evil.example"},
        "form content type": {**good, "Content-Type": "text/plain"},
    }
    statuses = {
        name: client.send("POST", "/api/protect", raw=body, headers=headers)[0]
        for name, headers in cases.items()
    }
    same_origin = client.send(
        "POST", "/api/protect", raw=body, headers={**good, "Origin": f"http://{HOST}:{client.port}"}
    )[0]

    assert statuses == {
        "no token": 403,
        "wrong token": 403,
        "other origin": 403,
        "rebound host": 403,
        "form content type": 415,
    }
    assert same_origin == 400  # 막히지 않고 파일 검사까지 간다.


def test_odd_bodies_and_paths_are_refused(client: Client) -> None:
    good = {"Content-Type": "application/json", TOKEN_HEADER: client.token}

    assert client.send("POST", "/api/protect", raw=b"{", headers=good)[0] == 400
    assert client.send("POST", "/api/protect", raw=b"[1]", headers=good)[0] == 400
    assert client.send("POST", "/api/protect", raw=b"[" * 200_000, headers=good)[0] == 400
    assert client.send("POST", "/api/nothing", raw=b"{}", headers=good)[0] == 404
    assert client.send("GET", f"/etc/passwd?t={client.token}")[0] == 404
    assert client.send("GET", f"/../page.html?t={client.token}")[0] == 404
    assert client.send("PUT", "/api/protect", raw=b"{}", headers=good)[0] == 501
    oversized = {**good, "Content-Length": str(400 * 1024 * 1024)}
    assert client.send("POST", "/api/protect", raw=b"{}", headers=oversized)[0] == 413


def test_the_server_listens_on_loopback_only(tmp_path: Path) -> None:
    server, address = make_server(tmp_path)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert address.startswith("http://127.0.0.1:")
        assert len(address.split("?t=")[1]) >= 32
        assert make_server(tmp_path)[1].split("?t=")[1] != address.split("?t=")[1]
    finally:
        server.server_close()


def test_download_names_cannot_carry_paths_or_markup() -> None:
    assert safe_name("../../etc/passwd") == "_.._etc_passwd"
    assert "/" not in safe_name("a/b\\c") and "\\" not in safe_name("a/b\\c")
    assert safe_name('a"><script>') == "a_script_"
    assert safe_name("2026 시험지-1") == "2026 시험지-1"


def test_ui_command_reports_a_port_it_cannot_use(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr("examshield.cli.make_server", refuse)

    result = CliRunner().invoke(app, ["ui", "--port", "48799", "--no-browser"])

    assert result.exit_code == 2
    assert "띄우지 못했다" in result.stderr


def test_a_silent_connection_does_not_block_other_requests(client: Client) -> None:
    idle = socket.create_connection((HOST, client.port), timeout=5)
    half = socket.create_connection((HOST, client.port), timeout=5)
    half.sendall(b"POST /api/protect HTTP/1.1\r\nHost: x\r\n")
    try:
        status, _, _ = client.send("GET", f"/api/profiles?t={client.token}")
    finally:
        idle.close()
        half.close()

    assert status == 200


def test_broken_files_get_an_answer_instead_of_a_dropped_connection(
    client: Client, sample_pdf: bytes
) -> None:
    png = b64(encode_png(np.zeros((40, 40), dtype=np.uint8)))
    locked = b64(
        bytes(
            open_pdf_bytes(sample_pdf).tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="p")
        )
    )
    cut = b64(sample_pdf[: len(sample_pdf) // 2])
    good = b64(sample_pdf)

    answers = [
        client.post("/api/protect", {"name": "t.pdf", "file": png, "profile": "a-decoy"}),
        client.post("/api/protect", {"name": "t.pdf", "file": cut, "profile": "a-decoy"}),
        client.post("/api/protect", {"name": "t.pdf", "file": locked, "profile": "none"}),
        client.post("/api/check", {"name": "t.pdf", "file": png, "copy": png}),
        client.post("/api/check", {"name": "t.pdf", "file": locked, "copy": locked}),
        client.post("/api/check", {"name": "t.pdf", "file": good, "copy": good, "job": "{"}),
        client.post(
            "/api/check", {"name": "t.pdf", "file": good, "copy": good, "job": "[" * 100_000}
        ),
        client.post("/api/check", {"name": "t.pdf", "file": good, "copy": good, "job": "\ud800"}),
    ]

    for status, answer in answers:
        assert status in (400, 500) and answer["error"], (status, answer)
        assert "/" not in answer["error"] and "Traceback" not in answer["error"]
    assert client.send("GET", f"/api/profiles?t={client.token}")[0] == 200


def test_an_unexpected_failure_is_answered_without_details(
    client: Client, sample_pdf: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object) -> None:
        raise RuntimeError("비밀 문항 내용")

    monkeypatch.setattr(Service, "protect", boom)

    status, answer = client.post(
        "/api/protect", {"name": "a.pdf", "file": b64(sample_pdf), "profile": "none"}
    )

    assert status == 500
    assert "비밀" not in answer["error"]


def test_localhost_is_accepted_as_a_host_name(client: Client) -> None:
    named = client.send(
        "GET", f"/api/profiles?t={client.token}", headers={"Host": f"localhost:{client.port}"}
    )[0]
    shouted = client.send(
        "GET", f"/api/profiles?t={client.token}", headers={"Host": f"LOCALHOST:{client.port}"}
    )[0]

    assert named == 200 and shouted == 403


def test_the_page_drops_the_token_from_the_address_bar() -> None:
    page = (Path(__file__).resolve().parent.parent / "examshield/ui/page.html").read_text("utf-8")

    assert 'history.replaceState(null, "", "/")' in page


SEED = b"0123456789abcdef-institution-seed"


@pytest.fixture
def desk(tmp_path: Path) -> Iterator[tuple[Client, Path]]:
    """웹 화면을 띄운 폴더가 정해진 서버. 시드 파일이 secrets/에 있다."""
    base = tmp_path / "desk"
    (base / "secrets").mkdir(parents=True)
    (base / "secrets" / "exam.seed").write_bytes(SEED)
    server, address = make_server(
        tmp_path / "work",
        reports=tmp_path / "reports",
        records=tmp_path / "records",
        base_dir=base,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield Client(address), base
    server.shutdown()
    server.server_close()


def copies_request(pdf: bytes, **changed: Any) -> dict[str, Any]:
    return {
        "name": "exam.pdf",
        "file": base64.b64encode(pdf).decode(),
        "profile": "a-full",
        "ids": "수험번호\nA1\nA2\nA3\n",
        "seed_path": "secrets/exam.seed",
        "map_path": "secrets/exam.map.csv",
        "out_dir": "out/copies",
        **changed,
    }


def test_copies_are_written_to_the_folder_and_a_capture_is_traced(
    desk: tuple[Client, Path], sample_pdf: bytes
) -> None:
    client, base = desk

    status, made = client.post("/api/copies", copies_request(sample_pdf))
    written = sorted(path.name for path in (base / "out" / "copies").glob("*.pdf"))
    table = (base / "secrets" / "exam.map.csv").read_text(encoding="utf-8")

    assert status == 200 and made["passed"], made
    assert written == ["exam-A1.pdf", "exam-A2.pdf", "exam-A3.pdf"]
    assert "A2" in table and "복사본 3개" in made["summary"]
    # 복사본도, 시드도, 매핑표의 내용도 브라우저로 돌아가지 않는다.
    sent_back = json.dumps(made)
    assert "pdf" not in made and "0123456789abcdef" not in sent_back
    assert table.splitlines()[1].split(",")[1] not in sent_back

    copy = open_pdf_bytes((base / "out" / "copies" / "exam-A2.pdf").read_bytes())
    pixmap = copy[1].get_pixmap(dpi=130, colorspace=pymupdf.csGRAY)
    shot = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.stride)
    capture = encode_png(np.ascontiguousarray(shot[:, : pixmap.width]))
    trace_request = {
        "name": "exam.pdf",
        "file": base64.b64encode(sample_pdf).decode(),
        "leaked": [{"name": "capture.png", "file": base64.b64encode(capture).decode()}],
        "seed_path": "secrets/exam.seed",
        "map_path": "secrets/exam.map.csv",
    }
    status, traced = client.post("/api/trace", trace_request)

    assert status == 200 and traced["found"], traced
    assert "A2" in traced["summary"]

    again = client.post("/api/copies", copies_request(sample_pdf))
    resumed = client.post("/api/copies", copies_request(sample_pdf, resume=True))
    assert again[0] == 400 and "이미 같은 이름" in again[1]["error"]
    assert resumed[0] == 200 and "그대로 둔 것 3개" in resumed[1]["summary"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed_path", "/etc/passwd"),
        ("seed_path", "../outside.seed"),
        ("map_path", "../../map.csv"),
        ("out_dir", ".."),
        ("out_dir", "."),
        ("out_dir", "out/../../elsewhere"),
        ("map_path", "out/copies/map.csv"),
        ("seed_path", "secrets/none.seed"),
        ("seed_path", ""),
    ],
)
def test_paths_outside_the_folder_or_inside_the_output_are_refused(
    desk: tuple[Client, Path], sample_pdf: bytes, tmp_path: Path, field: str, value: str
) -> None:
    client, base = desk
    (tmp_path / "outside.seed").write_bytes(b"x" * 32)

    status, answer = client.post("/api/copies", copies_request(sample_pdf, **{field: value}))

    assert status == 400 and "error" in answer
    assert not (base / "out").exists() and not (tmp_path / "elsewhere").exists()
    assert not list(tmp_path.glob("*.csv"))


def test_a_link_out_of_the_folder_is_not_followed(
    desk: tuple[Client, Path], sample_pdf: bytes, tmp_path: Path
) -> None:
    client, base = desk
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)

    status, answer = client.post("/api/copies", copies_request(sample_pdf, out_dir="link/copies"))

    assert status == 400 and "밖" in answer["error"]
    assert list(outside.iterdir()) == []


def test_copies_need_a_per_examinee_profile_and_a_clean_list(
    desk: tuple[Client, Path], sample_pdf: bytes
) -> None:
    client, _ = desk

    plain = client.post("/api/copies", copies_request(sample_pdf, profile="a-remap-decoy"))
    bad_id = client.post("/api/copies", copies_request(sample_pdf, ids="A1\n../A2\n"))
    twice = client.post("/api/copies", copies_request(sample_pdf, ids="A1\na1\n"))
    none = client.post("/api/copies", copies_request(sample_pdf, ids="\n\n"))

    assert [status for status, _ in (plain, bad_id, twice, none)] == [400, 400, 400, 400]
    assert "수험번호" in bad_id[1]["error"] and "겹친다" in twice[1]["error"]


def test_trace_refuses_missing_tables_and_too_many_or_no_files(
    desk: tuple[Client, Path], sample_pdf: bytes
) -> None:
    client, _ = desk
    base_request = {
        "name": "exam.pdf",
        "file": base64.b64encode(sample_pdf).decode(),
        "seed_path": "secrets/exam.seed",
        "map_path": "secrets/none.csv",
    }
    one = {"name": "a.png", "file": base64.b64encode(b"x").decode()}

    missing = client.post("/api/trace", {**base_request, "leaked": [one]})
    empty = client.post("/api/trace", {**base_request, "leaked": []})
    crowd = client.post("/api/trace", {**base_request, "leaked": [one] * 51})
    odd = client.post("/api/trace", {**base_request, "leaked": ["a.png"]})

    assert missing[0] == 400 and "매핑표" in missing[1]["error"]
    assert empty[0] == 400 and crowd[0] == 400 and odd[0] == 400


def test_a_long_job_is_started_then_asked_about_then_read(
    desk: tuple[Client, Path], sample_pdf: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """복사본 만들기는 바로 작업 번호를 돌려주고, 진행과 결과는 따로 물어 본다."""
    client, base = desk
    gate = threading.Event()
    reached = threading.Event()
    real = batch.write_outputs

    def held(protected: Any, out: Path, stem: str) -> Path:
        if stem.endswith("A2"):
            reached.set()
            assert gate.wait(60)
        return real(protected, out, stem)

    monkeypatch.setattr(batch, "write_outputs", held)

    status, started = client.start("/api/copies", copies_request(sample_pdf))
    assert reached.wait(60)
    _, middle = client.job(started["job"])
    busy = client.start("/api/copies", copies_request(sample_pdf, out_dir="out/second"))
    latest = client.job()[1]
    gate.set()
    final = client.wait(started["job"])
    kept = client.job()[1]

    assert status == 202 and set(started) == {"job"}
    assert middle["state"] == "running" and (middle["done"], middle["total"]) == (1, 3)
    assert middle["result"] is None and middle["kind"] == "copies"
    assert busy[0] == 409 and "진행 중" in busy[1]["error"]
    assert not (base / "out" / "second").exists()
    assert latest["id"] == started["job"]
    assert final["state"] == "done" and final["result"]["passed"]
    assert "복사본 3개" in final["result"]["summary"]
    # 결과는 다음 작업을 시작할 때까지 남는다. 화면을 새로 열어도 볼 수 있다.
    assert kept == final


def test_a_job_that_fails_says_why_and_frees_the_desk(
    desk: tuple[Client, Path], sample_pdf: bytes
) -> None:
    client, _ = desk
    client.post("/api/copies", copies_request(sample_pdf))

    status, started = client.start("/api/copies", copies_request(sample_pdf))
    failed = client.wait(started["job"])
    again = client.post("/api/copies", copies_request(sample_pdf, resume=True))

    assert status == 202
    assert failed["state"] == "failed" and "이미 같은 이름" in failed["error"]
    assert failed["result"] is None
    assert again[0] == 200 and again[1]["passed"]


def test_jobs_are_answered_only_with_the_token_and_the_right_number(
    desk: tuple[Client, Path], sample_pdf: bytes
) -> None:
    client, _ = desk

    idle = client.job()
    refused, _, _ = client.send("GET", "/api/job")
    client.post("/api/copies", copies_request(sample_pdf))
    wrong = client.job("no-such-job")
    gone, _, _ = client.send("GET", f"/api/progress?t={client.token}")

    assert idle == (200, {"state": "none"})
    assert refused == 403
    assert wrong[0] == 404
    assert gone == 404


def test_a_job_result_never_carries_the_seed_or_the_table(
    desk: tuple[Client, Path], sample_pdf: bytes
) -> None:
    client, base = desk
    client.post("/api/copies", copies_request(sample_pdf))

    _, _, body = client.send("GET", f"/api/job?t={client.token}")
    codes = [
        line.split(",")[1]
        for line in (base / "secrets" / "exam.map.csv").read_text().splitlines()[1:]
    ]

    assert SEED not in body and base64.b64encode(SEED) not in body
    assert all(code.encode() not in body for code in codes)
    assert b"%PDF" not in body and base64.b64encode(sample_pdf)[:40] not in body


def test_the_page_offers_copies_and_trace_and_tells_where_paths_start(
    desk: tuple[Client, Path],
) -> None:
    client, base = desk

    _, _, page = client.send("GET", f"/?t={client.token}")
    _, _, listed = client.send("GET", f"/api/profiles?t={client.token}")
    text = page.decode("utf-8")

    assert "수험번호별 복사본" in text and "유출본 추적" in text
    assert 'id="copies-seed" type="text"' in text and 'type="file" id="copies-seed"' not in text
    assert json.loads(listed)["base_dir"] == str(base.resolve())


def test_low_vision_numbers_get_copies_marked_only_in_the_margins(
    desk: tuple[Client, Path], sample_pdf: bytes
) -> None:
    client, base = desk

    status, made = client.post("/api/copies", copies_request(sample_pdf, low_vision="A2\n"))
    table = (base / "secrets" / "exam.map.csv").read_text(encoding="utf-8").splitlines()
    unknown = client.post(
        "/api/copies", copies_request(sample_pdf, low_vision="Z9\n", out_dir="out/other")
    )
    _, _, page = client.send("GET", f"/?t={client.token}")

    assert status == 200 and made["passed"], made
    assert "저시력 수험생용 복사본 1개" in made["summary"]
    assert [line.rsplit(",", 1)[1] for line in table[1:]] == ["0", "18", "0"]
    assert unknown[0] == 400 and "Z9" in unknown[1]["error"]
    assert 'id="copies-low"' in page.decode("utf-8")
