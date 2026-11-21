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


class OpenAIModel(CloudModel):
    def __init__(self, key: FoundKey, model: str | None = None, base_url: str | None = None):
        import openai

        super().__init__(get_provider("openai"), key, model, base_url)
        self._sdk = openai
        # 재시도를 끈다. 재시도하면 문서를 다시 보내게 되고, 확인받은 호출 수를 넘는다.
        self._client = openai.OpenAI(api_key=key.value, base_url=self.base_url, max_retries=0)

    def run(self, page: PageInput, channel: str) -> ModelReply:
        self.before_call()
        content: list[dict[str, Any]] = []
        if channel == IMAGE:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64(page.image_png)}",
                    "detail": "high",
                }
            )
        elif channel == FILE:
            content.append(
                {
                    "type": "input_file",
                    "filename": "page.pdf",
                    "file_data": f"data:application/pdf;base64,{b64(page.pdf)}",
                }
            )
        content.append({"type": "input_text", "text": prompt_for(page, channel)})
        sdk = self._sdk
        try:
            response = self._client.responses.create(
                model=self.model, input=[{"role": "user", "content": content}]
            )
        except (sdk.AuthenticationError, sdk.PermissionDeniedError, sdk.NotFoundError) as error:
            raise ModelAbortError(_abort_reason("OpenAI", error.status_code)) from error
        except sdk.RateLimitError as error:
            raise ModelUnavailableError("OpenAI 호출 한도에 걸렸다.") from error
        except sdk.APIStatusError as error:
            raise ModelUnavailableError(f"OpenAI 오류 {error.status_code}") from error
        except sdk.APIConnectionError as error:
            raise ModelUnavailableError("OpenAI에 연결하지 못했다.") from error
        self.saw_version(response.model)
        usage = response.usage
        tokens = (
            token_counts(usage.input_tokens, usage.output_tokens) if usage is not None else None
        )
        text = response.output_text
        if not text and _refused(response):
            return refused(tokens)
        return reply_from(text, page, tokens)


def _refused(response: Any) -> bool:
    for output in response.output:
        if output.type == "message" and any(part.type == "refusal" for part in output.content):
            return True
    return False
