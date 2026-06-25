import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "mimo-v2.5-free"
DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        rpm: int = 50,
        openrouter_key: Optional[str] = None,
    ):
        if openrouter_key:
            self.api_key = openrouter_key
            self.base_url = base_url or OPENROUTER_BASE_URL
            self.model = model or os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
        else:
            self.api_key = api_key or os.getenv("OPENCODE_ZEN_API_KEY", "no-key")
            self.base_url = base_url or os.getenv("OPENCODE_ZEN_BASE_URL", DEFAULT_BASE_URL)
            self.model = model or os.getenv("OPENCODE_ZEN_MODEL", DEFAULT_MODEL)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval = 60.0 / rpm
        self._last_call = 0.0

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
            max_retries=2,
        )

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def generate(self, prompt: str, strategy: str = "zero_shot") -> str:
        max_tokens = self.max_tokens
        if strategy == "cot":
            max_tokens = max(max_tokens, 768)

        for attempt in range(3):
            self._rate_limit()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                self._last_call = time.time()
                content = resp.choices[0].message.content or ""
                if content.strip():
                    return content
            except Exception as e:
                print(f"[llm_client] Attempt {attempt+1}/{3} failed for {self.model}: {type(e).__name__}: {e}", flush=True)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        print(f"[llm_client] All 3 attempts failed for {self.model}, returning empty", flush=True)
        return ""
