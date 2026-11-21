from typing import Any

from examshield.auth.keys import FoundKey
from examshield.errors import ModelAbortError, ModelUnavailableError
from examshield.providers import get_provider
from examshield.verify.models.base import FILE, IMAGE, ModelReply, PageInput, token_counts
from examshield.verify.models.cloud import CloudModel, abort_reason, prompt_for, refused, reply_from

FATAL_CODES = (401, 403, 404)
BLOCKED_REASONS = ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII")


class GoogleModel(CloudModel):
    def __init__(self, key: FoundKey, model: str | None = None, base_url: str | None = None):
        from google import genai
        from google.genai import errors, types

        super().__init__(get_provider("google"), key, model, base_url)
        self._errors = errors
        self._types = types
        self._client = genai.Client(
            api_key=key.value,
            vertexai=False,
            http_options=types.HttpOptions(base_url=self.base_url),
        )

    def run(self, page: PageInput, channel: str) -> ModelReply:
        self.before_call()
        contents: list[Any] = []
        if channel == IMAGE:
            contents.append(self._types.Part.from_bytes(data=page.image_png, mime_type="image/png"))
        elif channel == FILE:
            contents.append(self._types.Part.from_bytes(data=page.pdf, mime_type="application/pdf"))
        contents.append(prompt_for(page, channel))
        try:
            response = self._client.models.generate_content(model=self.model, contents=contents)
        except self._errors.ClientError as error:
            # Gemini는 틀린 키를 400으로 알려 준다.
            if error.code in FATAL_CODES or (error.code == 400 and "API key" in str(error)):
                raise ModelAbortError(
                    abort_reason("Google", 401 if error.code == 400 else error.code)
                ) from error
            if error.code == 429:
                raise ModelUnavailableError("Google 호출 한도에 걸렸다.") from error
            raise ModelUnavailableError(f"Google 오류 {error.code}") from error
        except self._errors.APIError as error:
            raise ModelUnavailableError(f"Google 오류 {error.code}") from error
        except Exception as error:
            # 이 SDK는 연결 실패를 HTTP 라이브러리의 예외 그대로 올린다.
            raise ModelUnavailableError("Google에 연결하지 못했다.") from error
        self.saw_version(response.model_version)
        usage = response.usage_metadata
        # 생각하는 데 쓴 토큰도 출력으로 청구된다.
        tokens = (
            token_counts(
                usage.prompt_token_count or 0,
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
            )
            if usage is not None
            else None
        )
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason:
            return refused(tokens)
        text = response.text or ""
        if not text and any(
            str(getattr(candidate.finish_reason, "name", candidate.finish_reason))
            in BLOCKED_REASONS
            for candidate in response.candidates or []
        ):
            return refused(tokens)
        return reply_from(text, page, tokens)
