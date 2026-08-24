import json
import os
from typing import Any

from .instructions import ICL_INSTRUCTION


def build_icl_messages(icl_examples: dict[str, Any], instruction: str = ICL_INSTRUCTION) -> list[dict[str, str]]:
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "Let's go through one example task."},
        {"role": "assistant", "content": "OK"},
    ]
    messages.append({"role": "user", "content": "The example task begins."})
    messages.append({"role": "assistant", "content": "OK"})
    for i, msg in enumerate(icl_examples):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "assistant", "content": "OK"})
    messages.append({"role": "user", "content": "You successfully finished the example task. Now, let's begin the main task."})
    messages.append({"role": "assistant", "content": "OK"})
    return messages


def icl_messages(icl_example_path: str, instruction: str = ICL_INSTRUCTION) -> list[dict[str, str]]:
    with open(icl_example_path, "r") as f:
        icl_examples = json.load(f)

    return build_icl_messages(
        instruction=instruction,
        icl_examples=icl_examples,
    )

def findnplace_icl_messages(icl_examples: dict[str, Any], instruction: str = ICL_INSTRUCTION) -> list[dict[str, str]]:
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "Let's go through one example task."},
        {"role": "assistant", "content": "OK"},
    ]
    messages.append({"role": "user", "content": "The example task begins."})
    messages.append({"role": "assistant", "content": "OK"})
    for i, msg in enumerate(icl_examples):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "assistant", "content": "OK"})
    messages.append({"role": "user", "content": "You successfully finished the example task. Now, let's begin the main task."})
    messages.append({"role": "assistant", "content": "OK"})
    return messages


def findnplace_icl_messages(
    icl_example_path: str = "prompt/hfes_icl_example_find_bdi.json",
    instruction: str = ICL_INSTRUCTION,
    fallback_icl_example_path: str = "prompt/hfes_icl_example_bdi.json",
) -> list[dict[str, str]]:
    example_path = icl_example_path
    if not os.path.exists(example_path):
        example_path = fallback_icl_example_path

    with open(example_path, "r", encoding="utf-8") as f:
        icl_examples = json.load(f)

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "Let's go through one example task."},
        {"role": "assistant", "content": "OK"},
    ]
    for i, msg in enumerate(icl_examples):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "assistant", "content": "OK"})
    messages.append({"role": "user", "content": "You successfully finished the example task. Now, let's begin the main task."})
    messages.append({"role": "assistant", "content": "OK"})
    return messages
