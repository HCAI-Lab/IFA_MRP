import argparse
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, OpenAIError


class LLM_gpt:
    """
    Client wrapper for OpenAI GPT models.

    Supports:
    - text generation
    - image inputs
    - function/tool calling
    """

    def __init__(self, **kwargs: Any):
        self.MODEL_ID: str = kwargs.get("model_id", "gpt-5-mini")
        self.TEMPERATURE: float = float(kwargs.get("temperature", 0.0))
        self.MAX_TOKENS: int = int(kwargs.get("max_tokens", 1024))
        self.TOP_P: float = float(kwargs.get("top_p", 1.0))
        self.REQUEST_TIMEOUT: float = float(kwargs.get("timeout", 120.0))
        self.MAX_RETRIES: int = int(kwargs.get("max_retries", 10))
        self.RETRY_INITIAL_DELAY: float = float(kwargs.get("retry_initial_delay", 10.0))

        self.API_KEY: Optional[str] = (
            kwargs.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
        )

        self.BASE_URL: Optional[str] = (
            kwargs.get("base_url")
            or os.environ.get("OPENAI_BASE_URL")
        )

        if not self.API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set the environment variable or pass api_key."
            )

        client_kwargs = {
            "api_key": self.API_KEY,
        }

        if self.BASE_URL:
            client_kwargs["base_url"] = self.BASE_URL

        self.client = OpenAI(**client_kwargs)

        print(f"LLM_gpt initialized with model: {self.MODEL_ID}")


    def _prepare_messages(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        if isinstance(prompt, str):

            if image_path:
                content = [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    self._encode_image_content(image_path),
                ]

                messages.append({
                    "role": "user",
                    "content": content,
                })

            else:
                messages.append({
                    "role": "user",
                    "content": prompt,
                })

        elif isinstance(prompt, list):
            messages.extend(prompt)

        else:
            raise TypeError(
                "prompt must be a string or a list of message dictionaries."
            )

        return messages


    @staticmethod
    def _encode_image_content(
        image_path: str,
    ) -> Dict[str, Any]:

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image path not found: {image_path}"
            )

        ext = os.path.splitext(image_path)[1].lower().strip(".")

        mime_type = (
            "image/jpeg"
            if ext in ("jpg", "jpeg")
            else f"image/{ext}"
        )

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:{mime_type};base64,"
                    f"{encoded_string}"
                )
            },
        }


    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:

        messages = self._prepare_messages(
            prompt,
            system_prompt,
            image_path,
        )

        request_kwargs: Dict[str, Any] = {
            "model": self.MODEL_ID,
            "messages": messages,
            "max_completion_tokens": (
                max_tokens
                if max_tokens is not None
                else self.MAX_TOKENS
            ),
            "timeout": self.REQUEST_TIMEOUT,
        }

        # Avoid non-default sampling parameters for GPT-5-family models.
        if not self.MODEL_ID.startswith("gpt-5"):

            request_kwargs["temperature"] = (
                temperature
                if temperature is not None
                else self.TEMPERATURE
            )

            request_kwargs["top_p"] = self.TOP_P

        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    **request_kwargs
                )

                choice = response.choices[0]
                content = choice.message.content

                if content and content.strip():
                    return content.strip()

                finish_reason = getattr(choice, "finish_reason", None)
                last_error = RuntimeError(
                    f"OpenAI API returned an empty message content"
                    f"{f' (finish_reason={finish_reason})' if finish_reason else ''}."
                )
                print(f"{last_error} Retry {attempt}/{self.MAX_RETRIES}.")

            except OpenAIError as e:
                last_error = e
                print(f"OpenAI API Error: {e}. Retry {attempt}/{self.MAX_RETRIES}.")

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_INITIAL_DELAY)

        if last_error:
            raise last_error

        return ""


    def generate_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        tool_choice = (
            {
                "type": "function",
                "function": {
                    "name": tool_name
                },
            }
            if tool_name
            else "auto"
        )

        request_kwargs: Dict[str, Any] = {
            "model": self.MODEL_ID,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_completion_tokens": (
                max_tokens
                if max_tokens is not None
                else self.MAX_TOKENS
            ),

            # This method returns one tool call,
            # so prevent parallel calls.
            "parallel_tool_calls": False,
        }

        # Avoid non-default sampling parameters for GPT-5-family models.
        if not self.MODEL_ID.startswith("gpt-5"):

            request_kwargs["temperature"] = self.TEMPERATURE
            request_kwargs["top_p"] = self.TOP_P

        try:
            response = self.client.chat.completions.create(
                **request_kwargs
            )

            message = response.choices[0].message

            if message.tool_calls:

                tool_call = message.tool_calls[0]

                args_str = tool_call.function.arguments

                try:
                    return json.loads(args_str)

                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Tool returned invalid JSON: {args_str}"
                    ) from e

            raw_content = message.content or ""

            try:
                return json.loads(raw_content)

            except json.JSONDecodeError:
                return {
                    "response": raw_content
                }

        except OpenAIError as e:
            print(
                f"OpenAI API Error during tool call: {e}"
            )
            raise


def main():

    parser = argparse.ArgumentParser(
        description="Test GPT API call using LLM_gpt class."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="GPT model ID (default: gpt-5-mini)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Hello! Give me a brief task plan "
            "to make coffee."
        ),
        help="Prompt text",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature. "
            "Ignored for GPT-5-family models."
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max output tokens",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file for vision input",
    )

    args = parser.parse_args()

    llm = LLM_gpt(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(
        f"\n--- Sending Request to {args.model} ---"
    )

    response = llm.generate(
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        image_path=args.image,
    )

    print("\n--- Response ---")
    print(response)


if __name__ == "__main__":
    main()
