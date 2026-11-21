from typing import Any

from examshield.auth.keys import FoundKey
from examshield.errors import ModelAbortError, ModelUnavailableError
from examshield.providers import get_provider
from examshield.verify.models.base import FILE, IMAGE, ModelReply, PageInput, token_counts
from examshield.verify.models.cloud import (
    CloudModel,
    b64,
    prompt_for,
    refused,
    reply_from,
)
from examshield.verify.models.cloud import (
    abort_reason as _abort_reason,
)

MAX_TOKENS = 16000
EFFORT = "high"


class AnthropicModel(CloudModel):
    def __init__(self, key: FoundKey, model: str | None = None, base_url: str | None = None):
        import anthropic

        super().__init__(get_provider("anthropic"), key, model, base_url)
        self._sdk = anthropic
        # 재시도를 끈다. 재시도하면 문서를 다시 보내게 되고, 확인받은 호출 수를 넘는다.
        self._client = anthropic.Anthropic(api_key=key.value, base_url=self.base_url, max_retries=0)

    def run(self, page: PageInput, channel: str) -> ModelReply:
        self.before_call()
        content: list[dict[str, Any]] = []
        if channel == IMAGE:
            source = {"type": "base64", "media_type": "image/png", "data": b64(page.image_png)}
            content.append({"type": "image", "source": source})
        elif channel == FILE:
            source = {"type": "base64", "media_type": "application/pdf", "data": b64(page.pdf)}
            content.append({"type": "document", "source": source})
        content.append({"type": "text", "text": prompt_for(page, channel)})
        sdk = self._sdk
        try:
            # 거부 시 다른 모델로 넘기는 fallbacks는 일부러 쓰지 않는다.
            # 리포트에 적힌 모델과 실제로 답한 모델이 달라지면 측정이 아니다.
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": content}],
            )
        except (sdk.AuthenticationError, sdk.PermissionDeniedError, sdk.NotFoundError) as error:
            raise ModelAbortError(_abort_reason("Anthropic", error.status_code)) from error
        except sdk.RateLimitError as error:
            raise ModelUnavailableError("Anthropic 호출 한도에 걸렸다.") from error
        except sdk.APIStatusError as error:
            raise ModelUnavailableError(f"Anthropic 오류 {error.status_code}") from error
        except sdk.APIConnectionError as error:
            raise ModelUnavailableError("Anthropic에 연결하지 못했다.") from error
        self.saw_version(response.model)
        usage = response.usage
        tokens = token_counts(usage.input_tokens, usage.output_tokens)
        if response.stop_reason == "refusal":
            return refused(tokens)
        text = "".join(block.text for block in response.content if block.type == "text")
        return reply_from(text, page, tokens)
