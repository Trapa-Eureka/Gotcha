import ipaddress
import os

from examshield.errors import NetworkRefusedError

OFFLINE_ENV = "EXAMSHIELD_OFFLINE"
_FALSE_VALUES = {"", "0", "false", "no", "off"}


def offline_enabled() -> bool:
    # 모르는 값은 켜진 것으로 본다. 오타 때문에 꺼진 채로 도는 일을 막는다.
    return os.environ.get(OFFLINE_ENV, "").strip().lower() not in _FALSE_VALUES


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def require_network(host: str, purpose: str) -> None:
    if offline_enabled() and not is_loopback(host):
        raise NetworkRefusedError(
            f"오프라인 설정({OFFLINE_ENV})이 켜져 있어 {host} 접속을 거부했다. 용도: {purpose}"
        )
