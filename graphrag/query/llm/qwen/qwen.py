import asyncio
import logging
from http import HTTPStatus
from typing import Any

import dashscope
from tenacity import (
    Retrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseLLMCallback, BaseLLM
from graphrag.query.progress import StatusReporter, ConsoleStatusReporter

log = logging.getLogger(__name__)

class DashscopeGenerationLLM(BaseLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = (Exception,),
        reporter: StatusReporter = ConsoleStatusReporter(),
    ):
        self.api_key = api_key
        self.model = model or dashscope.Generation.Models.qwen_turbo
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.retry_error_types = retry_error_types
        self._reporter = reporter

    def generate(
        self,
        messages: str | list[str],
        streaming: bool = False,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    return self._generate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(
                message="Error at generate()", details={self.__class__.__name__: str(e)}
            )
            return ""
        else:
            return ""

    async def agenerate(
        self,
        messages: str | list[str],
        streaming: bool = False,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    return await asyncio.to_thread(
                        self._generate,
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(f"Error at agenerate(): {e}")
            return ""
        else:
            return ""

    def _generate(
            self,
            messages: str | list[str],
            streaming: bool = False,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        if isinstance(messages, list):
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                stream=streaming,
                incremental_output=streaming,
                timeout=self.request_timeout,
                result_format='message',
                **kwargs,
            )
        else:
            response = dashscope.Generation.call(
                model=self.model,
                prompt=messages,
                api_key=self.api_key,
                stream=streaming,
                incremental_output=streaming,
                timeout=self.request_timeout,
                **kwargs,
            )

        # if response.status_code != HTTPStatus.OK:
        #     raise Exception(f"Error {response.code}: {response.message}")

        if streaming:
            full_response = ""
            for chunk in response:
                if chunk.status_code != HTTPStatus.OK:
                    raise Exception(f"Error {chunk.code}: {chunk.message}")

                decoded_chunk = chunk.output.choices[0]['message']['content']
                full_response += decoded_chunk
                if callbacks:
                    for callback in callbacks:
                        callback.on_llm_new_token(decoded_chunk)
            return full_response
        else:
            if isinstance(messages, list):
                return response.output["choices"][0]["message"]["content"]
            else:
                return response.output["text"]