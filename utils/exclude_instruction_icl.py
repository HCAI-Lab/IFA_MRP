#!/usr/bin/env python3
"""Write a chat transcript without instruction and ICL example messages.

This utility is intentionally standalone so the existing runner/prompt code does
not need to change. It reads a JSON transcript, removes scaffold messages, and
writes a filtered copy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("chat.json")
DEFAULT_OUTPUT = Path("chat_no_instruction_icl.json")

ICL_END_MARKER = "You successfully finished the example task. Now, let's begin the main task."
ICL_START_MARKERS = (
    "Let's go through one example task.",
    "The example task begins.",
)
CONTEXT_NOTICE_MARKERS = (
    "All the previous interaction transcripts are hidden.",
    "subtask-relevant memories",
)


def content_of(message: dict[str, Any]) -> str:
    return str(message.get("content", ""))


def is_ok_ack(message: dict[str, Any]) -> bool:
    content = content_of(message).strip().lower().rstrip(".")
    return message.get("role") == "assistant" and content == "ok"


def is_instruction_message(message: dict[str, Any]) -> bool:
    content = content_of(message)
    return (
        message.get("role") == "user"
        and content.startswith("Interact with a household to solve a task.")
        and "AVAILABLE ACTIONS:" in content
    )


def is_icl_start(message: dict[str, Any]) -> bool:
    content = content_of(message)
    return message.get("role") == "user" and any(marker in content for marker in ICL_START_MARKERS)


def is_icl_end(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and ICL_END_MARKER in content_of(message)


def is_context_notice(message: dict[str, Any]) -> bool:
    content = content_of(message)
    return message.get("role") == "user" and all(marker in content for marker in CONTEXT_NOTICE_MARKERS)


def is_memory_prompt(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and content_of(message).lstrip().startswith("* Subtask relevant memories:")


def skip_optional_ok(messages: list[dict[str, Any]], index: int, stats: Counter[str], reason: str) -> int:
    if index < len(messages) and is_ok_ack(messages[index]):
        stats[reason] += 1
        return index + 1
    return index


def skip_instruction_and_icl_block(
    messages: list[dict[str, Any]],
    start_index: int,
    stats: Counter[str],
) -> int:
    i = start_index
    while i < len(messages):
        stats["instruction_or_icl_messages"] += 1
        if is_icl_end(messages[i]):
            i += 1
            return skip_optional_ok(messages, i, stats, "instruction_or_icl_messages")
        i += 1

    return i


def filter_messages(
    messages: list[dict[str, Any]],
    *,
    drop_system: bool = True,
    drop_context_notice: bool = True,
    drop_memory_prompts: bool = True,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    cleaned: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    i = 0

    while i < len(messages):
        message = messages[i]

        if drop_system and message.get("role") == "system":
            stats["system_messages"] += 1
            i += 1
            continue

        if is_instruction_message(message) or is_icl_start(message):
            stats["instruction_or_icl_blocks"] += 1
            i = skip_instruction_and_icl_block(messages, i, stats)
            continue

        if is_icl_end(message):
            stats["instruction_or_icl_messages"] += 1
            i = skip_optional_ok(messages, i + 1, stats, "instruction_or_icl_messages")
            continue

        if drop_context_notice and is_context_notice(message):
            stats["context_notice_messages"] += 1
            i = skip_optional_ok(messages, i + 1, stats, "context_notice_messages")
            continue

        if drop_memory_prompts and is_memory_prompt(message):
            stats["memory_prompt_messages"] += 1
            i += 1
            continue

        cleaned.append(message)
        i += 1

    stats["input_messages"] = len(messages)
    stats["output_messages"] = len(cleaned)
    stats["removed_messages"] = len(messages) - len(cleaned)
    return cleaned, stats


def extract_messages(data: Any, key: str | None) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(data, list):
        messages = data
        used_key = None
    elif isinstance(data, dict):
        candidate_keys = [key] if key else ["messages", "all_messages", "stacked_messages"]
        used_key = next((candidate for candidate in candidate_keys if candidate in data), None)
        if used_key is None:
            raise KeyError(
                "Input JSON object does not contain a message list. "
                "Pass --messages-key with the correct key."
            )
        messages = data[used_key]
    else:
        raise TypeError("Input JSON must be a message list or an object containing a message list.")

    if not isinstance(messages, list):
        raise TypeError("The selected message value must be a list.")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"Message {index} is not a JSON object.")

    return messages, used_key


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exclude instruction, ICL example, and related scaffold messages from a chat JSON file.",
    )
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help="Input chat JSON file.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Filtered output JSON file.")
    parser.add_argument(
        "--messages-key",
        help="Message-list key to use when the input JSON is an object instead of a top-level list.",
    )
    parser.add_argument("--keep-system", action="store_true", help="Keep system separator messages.")
    parser.add_argument("--keep-context-notice", action="store_true", help="Keep find-and-place context notices.")
    parser.add_argument("--keep-memory-prompts", action="store_true", help="Keep subtask memory prompt messages.")
    parser.add_argument("--dry-run", action="store_true", help="Print removal stats without writing output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    messages, messages_key = extract_messages(data, args.messages_key)
    cleaned, stats = filter_messages(
        messages,
        drop_system=not args.keep_system,
        drop_context_notice=not args.keep_context_notice,
        drop_memory_prompts=not args.keep_memory_prompts,
    )

    output_data = cleaned
    if messages_key is not None:
        output_data = deepcopy(data)
        output_data[messages_key] = cleaned

    if not args.dry_run:
        write_json(args.output, output_data)

    print(f"input messages: {stats['input_messages']}")
    print(f"output messages: {stats['output_messages']}")
    print(f"removed messages: {stats['removed_messages']}")
    print(f"instruction/icl blocks removed: {stats['instruction_or_icl_blocks']}")
    print(f"instruction/icl messages removed: {stats['instruction_or_icl_messages']}")
    if stats["context_notice_messages"]:
        print(f"context notice messages removed: {stats['context_notice_messages']}")
    if stats["memory_prompt_messages"]:
        print(f"memory prompt messages removed: {stats['memory_prompt_messages']}")
    if stats["system_messages"]:
        print(f"system messages removed: {stats['system_messages']}")
    if not args.dry_run:
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
