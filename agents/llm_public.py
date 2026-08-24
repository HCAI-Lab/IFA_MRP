import json
import os
import re
from typing import Any

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM_public:
    def __init__(self, **kwargs: Any):
        self.MODEL_ID = kwargs.get("model_id")
        self.TEMPERATURE = float(kwargs.get("temperature", 0.0))
        self.DO_SAMPLE = bool(
            kwargs.get("do_sample", self.TEMPERATURE > 0)
        )
        self.MAX_TOKENS = int(kwargs.get("max_tokens", 1024))
        self.TOP_P = float(kwargs.get("top_p", 0.95))
        self.TOP_K = int(kwargs.get("top_k", 20))
        self.HF_TOKEN = (
            kwargs.get("hf_token") or os.environ.get("HF_TOKEN")
        )
        self.DATA_TYPE = kwargs.get("dtype", "auto")
        self.DEVICE = kwargs.get("device", "auto")

        if not self.MODEL_ID:
            raise ValueError("model_id is required.")
        if not self.HF_TOKEN:
            raise ValueError("Set HF_TOKEN or pass hf_token.")
        if self.DO_SAMPLE and self.TEMPERATURE <= 0:
            raise ValueError(
                "temperature must be > 0 when do_sample=True."
            )

        self.tokenizer = None
        self.model = None
        self.N_PARAMS = None
        self.load_llm()

    def load_llm(self) -> None:
        print(f"Loading model: {self.MODEL_ID}...")
        login(token=self.HF_TOKEN)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN,
            device_map=self.DEVICE,
            torch_dtype=self._get_dtype(self.DATA_TYPE),
            trust_remote_code=True,
        )
        self.model.eval()

        config = self.model.generation_config
        config.do_sample = self.DO_SAMPLE
        config.pad_token_id = self.tokenizer.pad_token_id

        if self.DO_SAMPLE:
            config.temperature = self.TEMPERATURE
            config.top_p = self.TOP_P
            config.top_k = self.TOP_K
        else:
            # Neutralize Qwen's saved sampling settings.
            config.temperature = 1.0
            config.top_p = 1.0
            config.top_k = 50

        total_params = sum(p.numel() for p in self.model.parameters())
        self.N_PARAMS = self._format_params(total_params)
        print(f"Model loaded successfully. Parameters: {self.N_PARAMS}")

    @staticmethod
    def _get_dtype(dtype):
        if not isinstance(dtype, str):
            return dtype

        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }

        key = dtype.lower()
        if key not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return dtype_map[key]

    @staticmethod
    def _format_params(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        return str(n)

    def _generation_kwargs(
        self,
        max_tokens: int | None = None,
    ) -> dict:
        kwargs = {
            "max_new_tokens": max_tokens or self.MAX_TOKENS,
            "do_sample": self.DO_SAMPLE,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.DO_SAMPLE:
            kwargs.update({
                "temperature": self.TEMPERATURE,
                "top_p": self.TOP_P,
                "top_k": self.TOP_K,
            })

        return kwargs

    def _prepare_inputs(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if tools is not None:
            template_kwargs["tools"] = tools

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

        return self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

    def generate(
        self,
        prompt: str | list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> str:
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )

        inputs = self._prepare_inputs(messages)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_tokens),
            )

        output_ids = generated[
            0,
            inputs.input_ids.shape[1]:,
        ]

        return self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

    def generate_tool_call(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_name: str,
        max_tokens: int | None = None,
    ) -> dict:
        inputs = self._prepare_inputs(messages, tools)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_tokens),
            )

        output_ids = generated[
            0,
            inputs.input_ids.shape[1]:,
        ]

        raw_output = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
        ).strip()

        return self._parse_tool_call(raw_output, tool_name)

    def _parse_tool_call(
        self,
        raw_output: str,
        expected_tool_name: str,
    ) -> dict:
        raw_output = self._clean_tokens(raw_output)

        # Qwen parameter-style output.
        function_match = re.search(
            r"<function=([^>]+)>\s*(.*?)\s*</function>",
            raw_output,
            flags=re.DOTALL,
        )

        if function_match:
            tool_name, body = function_match.groups()
            tool_name = tool_name.strip()

            if tool_name != expected_tool_name:
                raise ValueError(
                    f"Expected {expected_tool_name}, got {tool_name}."
                )

            return {
                key.strip(): self._parse_value(value)
                for key, value in re.findall(
                    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
                    body,
                    flags=re.DOTALL,
                )
            }

        # JSON inside tool-call tags.
        for pattern in (
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            (
                r"<\|tool_call_start\|>\s*(.*?)"
                r"\s*<\|tool_call_end\|>"
            ),
        ):
            match = re.search(pattern, raw_output, flags=re.DOTALL)
            if match:
                return self._parse_json_tool_call(
                    match.group(1),
                    expected_tool_name,
                )

        # Direct JSON fallback.
        candidate = self._extract_json(raw_output)
        if candidate is None:
            raise ValueError(
                f"No tool call found.\nRaw output:\n{raw_output}"
            )

        return self._parse_json_tool_call(
            candidate,
            expected_tool_name,
        )

    @staticmethod
    def _clean_tokens(text: str) -> str:
        for token in (
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<|end_of_text|>",
        ):
            text = text.replace(token, "")

        return text.strip()

    @classmethod
    def _parse_value(cls, value: str):
        value = cls._clean_tokens(value)

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        if value.lower() in {"none", "null"}:
            return None
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)

        return value

    @staticmethod
    def _extract_json(text: str) -> str | None:
        decoder = json.JSONDecoder()

        for index, character in enumerate(text):
            if character != "{":
                continue

            try:
                _, end = decoder.raw_decode(text[index:])
                return text[index:index + end]
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _parse_json_tool_call(
        text: str,
        expected_tool_name: str,
    ) -> dict:
        parsed = json.loads(text.strip())

        if "name" not in parsed or "arguments" not in parsed:
            return parsed

        if parsed["name"] != expected_tool_name:
            raise ValueError(
                f"Expected {expected_tool_name}, "
                f"got {parsed['name']}."
            )

        arguments = parsed["arguments"]

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return arguments