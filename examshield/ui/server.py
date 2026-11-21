"""이 PC에서만 열리는 웹 화면.

127.0.0.1에만 붙고, 시작할 때 만든 토큰이 있어야 쓸 수 있다. 올린 파일은 이 프로세스의 메모리에서
처리해 결과를 브라우저로 돌려준다. 디스크에는 Word·한글 변환용 임시 파일만 잠깐 쓴다.

오래 걸리는 일(수험번호별 복사본, 추적)은 요청 하나로 끝까지 기다리지 않는다. 시작 요청은 작업
번호만 돌려주고, 화면이 그 번호로 진행과 결과를 물어 본다. 브라우저가 몇 분짜리 응답을 기다리다
끊어도 작업은 끝까지 가고, 결과는 다음 작업을 시작할 때까지 남아 있다.

수험번호별 복사본과 추적은 다르다. 시드 파일과 매핑표는 브라우저를 거치지 않는다. 화면에는 경로만
적고, 서버가 이 PC의 그 파일을 직접 읽고 쓴다. 복사본도 내려받기 대신 출력 폴더에 쓴다. 경로는
웹 화면을 띄운 폴더 안으로만 받는다.
"""

import base64
import binascii
import hmac
import json
import re
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from examshield import __version__
from examshield.benchmark import load_items
from examshield.check.compare import Expectation, compare, format_result
from examshield.convert import (
    HWP_NOTICE,
    SUPPORTED_SUFFIXES,
    needs_review,
    read_document,
    work_folder,
)
from examshield.errors import ExamShieldError
from examshield.ocr import find_tesseract
from examshield.profiles import Profile, list_profile_names, load_profile
from examshield.protect.batch import CopiesFailed, low_vision_lines, make_copies
from examshield.protect.copies import keyed_technique, parse_examinee_ids, read_map
from examshield.protect.pipeline import KEYED, expectation_from_text, protect_and_check
from examshield.trace.detect import MAX_LEAKED_FILES, format_trace, trace_many
from examshield.verify.drift import DEFAULT_REPORT_DIR, RECORD_DIR, Current, assess, find_latest

HOST = "127.0.0.1"
PAGE = Path(__file__).resolve().parent / "page.html"
TOKEN_HEADER = "X-Examshield-Token"
MAX_BODY_BYTES = 100 * 1024 * 1024
MAX_IDS_CHARS = 200_000
# 아무것도 보내지 않는 연결이 자리를 잡고 있지 못하게 한다.
IDLE_TIMEOUT_SECONDS = 30
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
}
_NAME = re.compile(r"[^\w.\-가-힣 ]+")


class Refused(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


RUNNING, DONE, FAILED = "running", "done", "failed"


@dataclass
class Job:
    """오래 걸리는 일 하나. 화면이 번호로 진행과 결과를 물어 본다.

    결과에는 화면에 보여 줄 요약만 든다. 올린 파일, 시드, 매핑표의 내용은 들지 않는다.
    """

    id: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    kind: str = ""
    state: str = RUNNING
    label: str = ""
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "result": self.result,
            "error": self.error,
        }


class Service:
    """화면이 부르는 기능. HTTP와 떨어져 있어 따로 테스트할 수 있다."""

    def __init__(
        self,
        work_dir: Path,
        reports: Path | None = None,
        records: Path | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.reports = reports or DEFAULT_REPORT_DIR
        self.records = records or RECORD_DIR
        # 화면에 적은 경로는 이 폴더 안으로만 받는다.
        self.base_dir = (base_dir or Path.cwd()).resolve()
        # 가장 최근에 시작한 오래 걸리는 일. 다음 것을 시작하면 바뀐다.
        self.job: Job | None = None
        # 문서 처리는 한 번에 하나만 한다. 연결은 여러 개를 받아도 메모리와 CPU는 하나 몫만 쓴다.
        self.busy = threading.Lock()

    def _warnings(self, profile: Profile) -> tuple[str, list[str]]:
        try:
            now = datetime.now().astimezone()
            items = load_items()
            found, source, changed = find_latest(profile, [self.records, self.reports], now)
            result = assess(
                profile,
                found,
                now,
                Current(__version__, items.version, items.sha256),
                changed=changed,
                source=source,
            )
        except Exception:  # 경고를 못 만들었다고 화면을 멈추지 않는다.
            return "알 수 없음", []
        return result.label, list(result.warnings)

    def profiles(self) -> list[dict[str, Any]]:
        listed = []
        for name in list_profile_names():
            profile = load_profile(name)
            label, warnings = self._warnings(profile)
            listed.append(
                {
                    "name": profile.name,
                    "scenario": profile.scenario,
                    "status": profile.status,
                    "description": profile.description,
                    "techniques": [entry.id for entry in profile.techniques],
                    "needs_copies": any(entry.id == "watermark" for entry in profile.techniques),
                    "per_examinee": any(entry.id in KEYED for entry in profile.techniques),
                    "verification": label,
                    "warnings": warnings,
                }
            )
        return listed

    def _document(self, name: str, data: bytes) -> tuple[bytes, Path]:
        """올린 파일을 PDF 바이트로 바꾼다. PDF가 아니면 작업 폴더에 잠깐 써서 변환한다."""
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ExamShieldError(
                f"지원하지 않는 파일 형식이다: {suffix or '확장자 없음'} "
                "(PDF, .docx, .hwp, .hwpx만 받는다)"
            )
        label = Path(f"input{suffix}")
        if suffix == ".pdf":
            return data, label
        with work_folder(self.work_dir) as folder:
            source = folder / f"upload{suffix}"
            source.write_bytes(data)
            return read_document(source, folder), label

    def protect(self, name: str, data: bytes, profile_name: str) -> dict[str, Any]:
        profile = load_profile(profile_name)
        if any(entry.id == "watermark" for entry in profile.techniques):
            raise ExamShieldError(
                f"프로파일 '{profile.name}'은 수험번호별 복사본용이다. 아래의 "
                "'수험번호별 복사본'에서 만든다."
            )
        original, label = self._document(name, data)
        engine = find_tesseract()
        protected, result = protect_and_check(
            original, profile, None, engine.read if engine else None
        )
        _label, warnings = self._warnings(profile)
        notes = list(result.notes)
        if needs_review(label):
            notes.append(HWP_NOTICE)
        stem = safe_name(Path(name).stem) or "document"
        return {
            "passed": result.passed,
            "summary": format_result(result),
            "warnings": warnings,
            "notes": notes,
            "similarity": protected.record["text_layer_similarity"],
            "file_name": f"{stem}.{profile.name}.pdf",
            "job_name": f"{stem}.{profile.name}.job.json",
            # 검사에 실패한 복사본은 내주지 않는다.
            "pdf": base64.b64encode(protected.pdf).decode("ascii") if result.passed else None,
            "job": json.dumps(protected.record, ensure_ascii=False, indent=2, sort_keys=True)
            if result.passed
            else None,
        }

    def check(self, name: str, original: bytes, copy: bytes, job: str | None) -> dict[str, Any]:
        document, _label = self._document(name, original)
        expect = expectation_from_text(job, copy, "작업 기록") if job else Expectation()
        engine = find_tesseract()
        result = compare(document, copy, expect, engine.read if engine else None)
        return {"passed": result.passed, "summary": format_result(result)}

    def local_path(self, text: str, label: str) -> Path:
        """화면에 적은 경로를 이 PC의 경로로 바꾼다. 웹 화면을 띄운 폴더 밖은 받지 않는다."""
        if not text.strip() or "\x00" in text:
            raise ExamShieldError(f"{label} 경로가 비어 있다.")
        given = Path(text.strip())
        if given.is_absolute() or given.drive:
            raise ExamShieldError(
                f"{label} 경로는 웹 화면을 띄운 폴더({self.base_dir}) 기준으로 적는다. "
                "절대 경로는 받지 않는다."
            )
        resolved = (self.base_dir / given).resolve()
        if resolved == self.base_dir or not resolved.is_relative_to(self.base_dir):
            raise ExamShieldError(f"{label} 경로가 웹 화면을 띄운 폴더 밖이다: {text}")
        return resolved

    def copies(
        self,
        name: str,
        data: bytes,
        profile_name: str,
        ids_text: str,
        seed_path: str,
        map_path: str,
        out_dir: str,
        resume: bool,
        low_text: str = "",
    ) -> dict[str, Any]:
        return self.plan_copies(
            name, data, profile_name, ids_text, seed_path, map_path, out_dir, resume, low_text
        )(Job())

    def plan_copies(
        self,
        name: str,
        data: bytes,
        profile_name: str,
        ids_text: str,
        seed_path: str,
        map_path: str,
        out_dir: str,
        resume: bool,
        low_text: str = "",
    ) -> Callable[[Job], dict[str, Any]]:
        """적어 준 값을 바로 확인하고, 복사본을 만드는 일을 돌려준다. 잘못 적은 것은 여기서 걸린다."""
        profile = load_profile(profile_name)
        keyed_technique(profile)
        ids = parse_examinee_ids(ids_text, "수험번호 목록")
        low = parse_examinee_ids(low_text, "저시력 수험번호 목록") if low_text.strip() else []
        seed_file = self.local_path(seed_path, "시드 파일")
        table = self.local_path(map_path, "매핑표")
        out = self.local_path(out_dir, "출력 폴더")
        stem = safe_name(Path(name).stem) or "document"
        engine = find_tesseract()
        label = Path(name)

        def work(job: Job) -> dict[str, Any]:
            def started(kept: int, total: int) -> None:
                job.label, job.done, job.total = "복사본", kept, total

            def stepped() -> None:
                job.done += 1

            try:
                batch = make_copies(
                    stem=stem,
                    load=lambda: self._document(name, data)[0],
                    out=out,
                    profile=profile,
                    seed_file=seed_file,
                    ids=ids,
                    table=table,
                    resume=resume,
                    ocr=engine.read if engine else None,
                    low_vision=low,
                    on_start=started,
                    on_step=stepped,
                )
            except CopiesFailed as error:
                return {"passed": False, "summary": f"{error}\n{error.detail}", "notes": []}
            _label, warnings = self._warnings(profile)
            notes = [HWP_NOTICE] if needs_review(label) else []
            lines = [
                f"복사본 {len(batch.rows)}개, 모두 검사 통과",
                f"이번에 만든 것 {batch.made}개, 그대로 둔 것 {batch.kept}개",
                f"수험번호: {ids[0]} ~ {ids[-1]} ({len(ids)}개)",
                *low_vision_lines(batch),
                f"출력 폴더: {out.relative_to(self.base_dir)}",
                f"매핑표: {table.relative_to(self.base_dir)}",
                "매핑표와 시드 파일은 유출본을 추적할 때 필요하다. 배포하지 말고 따로 보관한다.",
            ]
            return {
                "passed": True,
                "summary": "\n".join(lines),
                "warnings": warnings,
                "notes": notes,
            }

        return work

    def trace(
        self,
        name: str,
        original: bytes,
        leaked: list[tuple[str, bytes]],
        seed_path: str,
        map_path: str,
    ) -> dict[str, Any]:
        return self.plan_trace(name, original, leaked, seed_path, map_path)(Job())

    def plan_trace(
        self,
        name: str,
        original: bytes,
        leaked: list[tuple[str, bytes]],
        seed_path: str,
        map_path: str,
    ) -> Callable[[Job], dict[str, Any]]:
        seed_file = self.local_path(seed_path, "시드 파일")
        table = self.local_path(map_path, "매핑표")
        for label, path in (("시드 파일", seed_file), ("매핑표", table)):
            if not path.is_file():
                raise ExamShieldError(f"{label}을 찾지 못했다: {path.relative_to(self.base_dir)}")
        if not 1 <= len(leaked) <= MAX_LEAKED_FILES:
            raise ExamShieldError(f"유출본은 1개에서 {MAX_LEAKED_FILES}개까지 받는다.")

        def work(job: Job) -> dict[str, Any]:
            job.label = "추적"
            document, _label = self._document(name, original)
            result = trace_many(
                [
                    (safe_name(entry) or "유출본", data, Path(entry).suffix)
                    for entry, data in leaked
                ],
                document,
                read_map(table),
                seed_file.read_bytes(),
            )
            return {"found": result.found, "summary": format_trace(result)}

        return work

    def start(self, kind: str, work: Callable[[Job], dict[str, Any]]) -> Job:
        """오래 걸리는 일을 따로 돌리고 바로 돌아온다. 한 번에 하나만 돌린다."""
        if not self.busy.acquire(blocking=False):
            raise Refused(HTTPStatus.CONFLICT, "다른 작업이 진행 중이다. 끝난 뒤에 다시 한다.")
        job = Job(kind=kind)
        self.job = job

        def run() -> None:
            try:
                job.result = work(job)
                job.state = DONE
            except ExamShieldError as error:
                job.error, job.state = str(error), FAILED
            except Exception:  # 내용은 내보내지 않는다.
                job.error = "파일을 처리하지 못했다. 깨졌거나 지원하지 않는 형식일 수 있다."
                job.state = FAILED
            finally:
                self.busy.release()

        threading.Thread(target=run, daemon=True, name=f"examshield-{kind}").start()
        return job

    def job_view(self, wanted: str) -> dict[str, Any]:
        """작업의 진행과 결과. 번호를 주지 않으면 가장 최근 것을 돌려준다(화면을 새로 열었을 때)."""
        job = self.job
        if job is None or (wanted and not hmac.compare_digest(wanted.encode(), job.id.encode())):
            if wanted:
                raise Refused(
                    HTTPStatus.NOT_FOUND, "그런 작업이 없다. 다른 작업을 시작했을 수 있다."
                )
            return {"state": "none"}
        return job.view()


def safe_name(text: str) -> str:
    return _NAME.sub("_", text).strip(" .")[:80]


def _decode(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise Refused(HTTPStatus.BAD_REQUEST, f"{label}이(가) 없다.")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise Refused(HTTPStatus.BAD_REQUEST, f"{label}을(를) 읽지 못했다.") from error


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 300:
        raise Refused(HTTPStatus.BAD_REQUEST, f"{label}이(가) 없거나 너무 길다.")
    return value


def make_handler(service: Service, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "examshield-ui"
        sys_version = ""
        timeout = IDLE_TIMEOUT_SECONDS

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """요청 기록을 남기지 않는다. 파일 이름이 로그에 들어가지 않게 하려는 것이다."""

        def _allowed_hosts(self) -> set[str]:
            port = self.connection.getsockname()[1]
            return {f"{HOST}:{port}", f"localhost:{port}"}

        def _guard(self) -> None:
            # 다른 사이트가 이 PC의 주소로 이름을 돌려 붙는 것(DNS rebinding)을 막는다.
            if self.headers.get("Host", "") not in self._allowed_hosts():
                raise Refused(HTTPStatus.FORBIDDEN, "이 주소로는 쓸 수 없다.")
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {
                f"http://{host}" for host in self._allowed_hosts()
            }:
                raise Refused(HTTPStatus.FORBIDDEN, "다른 사이트에서 온 요청이다.")

        def _send(self, status: HTTPStatus, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _fail(self, error: Refused) -> None:
            self.close_connection = True
            self._json(error.status, {"error": error.message})

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._guard()
                url = urlsplit(self.path)
                given = parse_qs(url.query).get("t", [""])[0]
                if not hmac.compare_digest(given.encode(), token.encode()):
                    raise Refused(
                        HTTPStatus.FORBIDDEN,
                        "토큰이 없다. examshield ui가 보여준 주소로 연다.",
                    )
                if url.path == "/":
                    page = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", token)
                    self._send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
                elif url.path == "/api/profiles":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "profiles": service.profiles(),
                            "version": __version__,
                            "base_dir": str(service.base_dir),
                        },
                    )
                elif url.path == "/api/job":
                    wanted = parse_qs(url.query).get("id", [""])[0]
                    self._json(HTTPStatus.OK, service.job_view(wanted))
                else:
                    raise Refused(HTTPStatus.NOT_FOUND, "없는 주소다.")
            except Refused as error:
                self._fail(error)
            except ExamShieldError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception:  # 어떤 요청도 응답 없이 끊기지 않게 한다. 내용은 내보내지 않는다.
                self._unexpected()

        def _unexpected(self) -> None:
            self.close_connection = True
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "파일을 처리하지 못했다. 깨졌거나 지원하지 않는 형식일 수 있다."},
            )

        def _body(self) -> dict[str, Any]:
            if not hmac.compare_digest(self.headers.get(TOKEN_HEADER, "").encode(), token.encode()):
                raise Refused(HTTPStatus.FORBIDDEN, "토큰이 맞지 않는다.")
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise Refused(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "JSON만 받는다.")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as error:
                raise Refused(HTTPStatus.LENGTH_REQUIRED, "본문 길이가 없다.") from error
            if not 0 < length <= MAX_BODY_BYTES:
                raise Refused(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "파일이 너무 크다.")
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, RecursionError) as error:
                raise Refused(HTTPStatus.BAD_REQUEST, "본문이 JSON이 아니다.") from error
            if not isinstance(payload, dict):
                raise Refused(HTTPStatus.BAD_REQUEST, "본문이 JSON 객체가 아니다.")
            return payload

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._guard()
                path = urlsplit(self.path).path
                if path not in ("/api/protect", "/api/check", "/api/copies", "/api/trace"):
                    raise Refused(HTTPStatus.NOT_FOUND, "없는 주소다.")
                payload = self._body()
                name = _text(payload.get("name"), "파일 이름")
                document = _decode(payload.get("file"), "파일")
                if path == "/api/protect":
                    profile = _text(payload.get("profile"), "프로파일")
                    with service.busy:
                        answer = service.protect(name, document, profile)
                elif path == "/api/copies":
                    ids_text = payload.get("ids")
                    if not isinstance(ids_text, str) or len(ids_text) > MAX_IDS_CHARS:
                        raise Refused(HTTPStatus.BAD_REQUEST, "수험번호 목록이 없거나 너무 길다.")
                    low_text = payload.get("low_vision", "")
                    if not isinstance(low_text, str) or len(low_text) > MAX_IDS_CHARS:
                        raise Refused(HTTPStatus.BAD_REQUEST, "저시력 수험번호 목록이 너무 길다.")
                    planned = service.plan_copies(
                        name,
                        document,
                        _text(payload.get("profile"), "프로파일"),
                        ids_text,
                        _text(payload.get("seed_path"), "시드 파일 경로"),
                        _text(payload.get("map_path"), "매핑표 경로"),
                        _text(payload.get("out_dir"), "출력 폴더"),
                        payload.get("resume") is True,
                        low_text,
                    )
                    self._json(HTTPStatus.ACCEPTED, {"job": service.start("copies", planned).id})
                    return
                elif path == "/api/trace":
                    listed = payload.get("leaked")
                    if not isinstance(listed, list) or not listed:
                        raise Refused(HTTPStatus.BAD_REQUEST, "유출본이 없다.")
                    leaked = []
                    for entry in listed:
                        if not isinstance(entry, dict):
                            raise Refused(HTTPStatus.BAD_REQUEST, "유출본을 읽지 못했다.")
                        leaked.append(
                            (
                                _text(entry.get("name"), "유출본 이름"),
                                _decode(entry.get("file"), "유출본"),
                            )
                        )
                    planned = service.plan_trace(
                        name,
                        document,
                        leaked,
                        _text(payload.get("seed_path"), "시드 파일 경로"),
                        _text(payload.get("map_path"), "매핑표 경로"),
                    )
                    self._json(HTTPStatus.ACCEPTED, {"job": service.start("trace", planned).id})
                    return
                else:
                    job = payload.get("job")
                    if job is not None and not isinstance(job, str):
                        raise Refused(HTTPStatus.BAD_REQUEST, "작업 기록을 읽지 못했다.")
                    copy = _decode(payload.get("copy"), "복사본")
                    with service.busy:
                        answer = service.check(name, document, copy, job)
                self._json(HTTPStatus.OK, answer)
            except Refused as error:
                self._fail(error)
            except ExamShieldError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except TimeoutError:
                self.close_connection = True
            except Exception:  # 어떤 요청도 응답 없이 끊기지 않게 한다. 내용은 내보내지 않는다.
                self._unexpected()

    return Handler


def make_server(
    work_dir: Path,
    port: int = 0,
    reports: Path | None = None,
    records: Path | None = None,
    base_dir: Path | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    """서버와 화면 주소를 돌려준다. port가 0이면 빈 포트를 고른다."""
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(
        (HOST, port), make_handler(Service(work_dir, reports, records, base_dir), token)
    )
    server.daemon_threads = True
    return server, f"http://{HOST}:{server.server_port}/?t={token}"
