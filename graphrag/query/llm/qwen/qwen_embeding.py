import asyncio
import logging
from typing import Any

import dashscope
from tenacity import (
    Retrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.progress import StatusReporter, ConsoleStatusReporter

log = logging.getLogger(__name__)


class DashscopeEmbedding(BaseTextEmbedding):

    def __init__(
            self,
            api_key: str | None = None,
            model: str = dashscope.TextEmbedding.Models.text_embedding_v1,
            max_retries: int = 10,
            retry_error_types: tuple[type[BaseException]] = (Exception,),
            reporter: StatusReporter = ConsoleStatusReporter(),
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_error_types = retry_error_types
        self._reporter = reporter

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            embedding = self._embed_with_retry(text, **kwargs)
            return embedding
        except Exception as e:
            self._reporter.error(
                message="Error embedding text",
                details={self.__class__.__name__: str(e)},
            )
            return []

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            embedding = await asyncio.to_thread(self._embed_with_retry, text, **kwargs)
            return embedding
        except Exception as e:
            self._reporter.error(
                message="Error embedding text asynchronously",
                details={self.__class__.__name__: str(e)},
            )
            return []

    def _embed_with_retry(self, text: str, **kwargs: Any) -> list[float]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    response = dashscope.TextEmbedding.call(
                        model=self.model,
                        input=text,
                        api_key=self.api_key,
                        **kwargs,
                    )
                    if response.status_code == 200:
                        embedding = response.output["embeddings"][0]["embedding"]
                        return embedding
                    else:
                        raise Exception(f"Error {response.code}: {response.message}")
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return []
