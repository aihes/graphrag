# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import asyncio
import json
import logging
from http import HTTPStatus
from typing import Unpack, List, Dict

import dashscope
import regex as re

from graphrag.config import LLMType
from graphrag.llm import LLMOutput
from graphrag.llm.base import BaseLLM
from graphrag.llm.base.base_llm import TIn, TOut
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)

log = logging.getLogger(__name__)


class QwenCompletionLLM(
    BaseLLM[
        CompletionInput,
        CompletionOutput,
    ]
):
    def __init__(self, llm_config: dict = None):
        log.info(f"llm_config: {llm_config}")
        self.llm_config = llm_config or {}
        self.api_key = self.llm_config.get("api_key", "")
        self.model = self.llm_config.get("model", dashscope.Generation.Models.qwen_turbo)
        # self.chat_mode = self.llm_config.get("chat_mode", False)
        self.llm_type = llm_config.get("type", LLMType.StaticResponse)
        self.chat_mode = (llm_config.get("type", LLMType.StaticResponse) == LLMType.QwenChat)

    async def _execute_llm(
            self,
            input: CompletionInput,
            **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput:
        log.info(f"input: {input}")
        log.info(f"kwargs: {kwargs}")

        variables = kwargs.get("variables", {})

        # 使用字符串替换功能替换占位符
        formatted_input = replace_placeholders(input, variables)

        if self.chat_mode:
            history = kwargs.get("history", [])
            messages = [
                *history,
                {"role": "user", "content": formatted_input},
            ]
            response = self.call_with_messages(messages)
        else:
            response = self.call_with_prompt(formatted_input)

        if response.status_code == HTTPStatus.OK:
            if self.chat_mode:
                return response.output["choices"][0]["message"]["content"]
            else:
                return response.output["text"]
        else:
            raise Exception(f"Error {response.code}: {response.message}")

    def call_with_prompt(self, query: str):
        print("call_with_prompt {}".format(query))
        response = dashscope.Generation.call(
            model=self.model,
            prompt=query,
            api_key=self.api_key
        )
        return response

    def call_with_messages(self, messages: list[dict[str, str]]):
        print("call_with_messages {}".format(messages))
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            result_format='message',
        )
        return response

    # 主函数
    async def _invoke_json(self, input: TIn, **kwargs) -> LLMOutput[TOut]:
        try:
            output = await self._execute_llm(input, **kwargs)
        except Exception as e:
            print(f"Error executing LLM: {e}")
            return LLMOutput[TOut](output=None, json=None)

        # 解析output的内容
        extracted_jsons = extract_json_strings(output)

        if len(extracted_jsons) > 0:
            json_data = extracted_jsons[0]
        else:
            json_data = None

        try:
            output_str = json.dumps(json_data)
        except (TypeError, ValueError) as e:
            print(f"Error serializing JSON: {e}")
            output_str = None

        return LLMOutput[TOut](
            output=output_str,
            json=json_data
        )


def replace_placeholders(input_str, variables):
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        input_str = input_str.replace(placeholder, value)
    return input_str


def preprocess_input(input_str):
    # 预处理输入字符串，移除或转义特殊字符
    return input_str.replace('<', '&lt;').replace('>', '&gt;')


def extract_json_strings(input_string: str) -> List[Dict]:
    # 正则表达式模式，用于匹配 JSON 对象
    json_pattern = re.compile(r'(\{(?:[^{}]|(?R))*\})')

    # 查找所有匹配的 JSON 子字符串
    matches = json_pattern.findall(input_string)

    json_objects = []
    for match in matches:
        try:
            # 尝试解析 JSON 子字符串
            json_object = json.loads(match)
            json_objects.append(json_object)
        except json.JSONDecodeError:
            # 如果解析失败，忽略此子字符串
            log.warning(f"Invalid JSON string: {match}")
            pass

    return json_objects


if __name__ == '__main__':
    llm_config = {
        "api_key": "",
        "model": dashscope.Generation.Models.qwen_turbo,
        "type": LLMType.QwenChat,
    }

    llm = QwenCompletionLLM(llm_config)


    async def main():
        output = await llm("请介绍一下通义千问")
        print(output)


    asyncio.run(main())

    llm_config = {
        "api_key": "",
        "model": dashscope.Generation.Models.qwen_turbo,
        "type": LLMType.Qwen,
    }
    llm = QwenCompletionLLM(llm_config)

    asyncio.run(main())
