'''Implements LLM clients and related utilities'''

import logging
from typing import Any, Optional
import openai
import anthropic
import huggingface_hub
import google.generativeai as genai


class LLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        pass


class OpenAILLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = openai.OpenAI()

    def invoke(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
        return response.choices[0].message.content

class AnthropicLLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = anthropic.Anthropic()

    def invoke(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.messages.create(model=self.model_name, messages=messages, **kwargs)
        return response.choices[0].message.content

class GeminiLLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = genai.GenerativeAI()

    def invoke(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.generate_content(model=self.model_name, messages=messages, **kwargs)
        return response.text

class HuggingFaceLLMClient(LLMClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = huggingface_hub.InferenceClient()

    def invoke(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
        return response.choices[0].message.content