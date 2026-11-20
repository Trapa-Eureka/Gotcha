"""로컬 웹 화면. 실제 HTTP로 붙어 본다. 127.0.0.1 밖으로는 나가지 않는다."""

import base64
import http.client
import json
import os
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from typer.testing import CliRunner

from examshield.cli import app
from examshield.pdfio import open_pdf_bytes, save_deterministic
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
        return status, json.loads(body)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    server, address = make_server(tmp_path / "work", reports=tmp_path / "reports")
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


def test_watermark_profiles_are_sent_to_the_command_line(client: Client, sample_pdf: bytes) -> None:
    status, answer = client.post(
        "/api/protect", {"name": "a.pdf", "file": b64(sample_pdf), "profile": "a-full"}
    )

    assert status == 400 and "--copies" in answer["error"]


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
    def refuse(*_: object) -> None:
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
