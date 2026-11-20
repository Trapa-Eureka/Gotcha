from dataclasses import dataclass
from urllib.parse import urlsplit

from examshield.errors import ExamShieldError


@dataclass(frozen=True)
class Provider:
    id: str
    company: str
    key_env: tuple[str, ...]
    model_env: str
    default_model: str
    base_url: str
    package: str

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or ""


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        id="openai",
        company="OpenAI",
        key_env=("OPENAI_API_KEY",),
        model_env="EXAMSHIELD_OPENAI_MODEL",
        default_model="gpt-6.2-sol",
        base_url="https://api.openai.com/v1",
        package="openai",
    ),
    Provider(
        id="anthropic",
        company="Anthropic",
        key_env=("ANTHROPIC_API_KEY",),
        model_env="EXAMSHIELD_ANTHROPIC_MODEL",
        default_model="claude-opus-5-5",
        base_url="https://api.anthropic.com",
        package="anthropic",
    ),
    Provider(
        id="google",
        company="Google",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        model_env="EXAMSHIELD_GEMINI_MODEL",
        default_model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com",
        package="google-genai",
    ),
)


def provider_ids() -> list[str]:
    return [provider.id for provider in PROVIDERS]


def get_provider(provider_id: str) -> Provider:
    for provider in PROVIDERS:
        if provider.id == provider_id:
            return provider
    raise ExamShieldError(f"모르는 제공사다: {provider_id}. 가능한 값: {', '.join(provider_ids())}")
