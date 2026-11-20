import os
from dataclasses import dataclass

from examshield.errors import ExamShieldError
from examshield.providers import Provider

SERVICE = "examshield"
SOURCE_KEYCHAIN = "OS 키체인"
VISIBLE_TAIL = 4
MIN_LENGTH_TO_SHOW_TAIL = 12


@dataclass(frozen=True)
class FoundKey:
    value: str
    source: str

    @property
    def masked(self) -> str:
        return mask(self.value)


def mask(value: str) -> str:
    if len(value) < MIN_LENGTH_TO_SHOW_TAIL:
        return "****"
    return "****" + value[-VISIBLE_TAIL:]


def usable(value: str) -> bool:
    return value.isascii() and value.isprintable() and " " not in value


def find_key(provider: Provider) -> FoundKey | None:
    """우선순위: 환경변수, 그다음 OS 키체인."""
    for name in provider.key_env:
        value = os.environ.get(name, "").strip()
        if value:
            return FoundKey(value=value, source=f"환경변수 {name}")
    stored = _keychain_get(provider.id)
    return FoundKey(value=stored, source=SOURCE_KEYCHAIN) if stored else None


def store_key(provider: Provider, value: str) -> None:
    value = value.strip()
    if not value:
        raise ExamShieldError("빈 키는 저장하지 않는다.")
    if not usable(value):
        raise ExamShieldError("키에 한글이나 공백 같은 글자가 섞여 있다. 붙여넣은 값을 확인한다.")
    import keyring
    import keyring.errors

    try:
        keyring.set_password(SERVICE, provider.id, value)
    except keyring.errors.KeyringError as error:
        raise _no_keychain() from error


def delete_key(provider: Provider) -> bool:
    import keyring
    import keyring.errors

    try:
        if keyring.get_password(SERVICE, provider.id) is None:
            return False
        keyring.delete_password(SERVICE, provider.id)
    except keyring.errors.KeyringError as error:
        raise _no_keychain() from error
    return True


def _keychain_get(provider_id: str) -> str | None:
    import keyring
    import keyring.errors

    try:
        value = keyring.get_password(SERVICE, provider_id)
    except keyring.errors.KeyringError:
        return None
    return value or None


def _no_keychain() -> ExamShieldError:
    return ExamShieldError(
        "OS 키체인을 쓸 수 없다. 키를 평문 파일에 쓰지 않으므로, 대신 환경변수로 키를 준다."
    )


def setup_help(provider: Provider) -> str:
    return (
        f"{provider.company}: 환경변수 {provider.key_env[0]}를 주거나 "
        f"`examshield auth add {provider.id}`로 OS 키체인에 저장한다."
    )
