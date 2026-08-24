import time
from typing import Any, Dict, List

from termcolor import colored

from env.utils import parse_action


def generate_valid_action(
    dm_agent: Any,
    active_messages: List[Dict[str, Any]],
    step: int,
    format_retry_count: int,
    format_retry_delay: float,
) -> tuple[str, Any]:
    """Generate an LLM response and retry until it contains a parseable action."""
    llm_output = dm_agent.generate(active_messages)
    print(colored(f"Assistant {step}:", "red"), llm_output)

    try:
        return llm_output, parse_action(llm_output)
    except ValueError as err:
        initial_error = err

    for retry_idx in range(1, format_retry_count + 1):
        print(colored(
            f"⚠️ [Format Warning]: Intention/Action parsing failed. "
            f"Waiting {format_retry_delay:g}s before retrying same context "
            f"({retry_idx}/{format_retry_count})...",
            "yellow",
        ))
        if llm_output.strip():
            print("*" * 100)
            print(llm_output)
            print("*" * 100)
        time.sleep(format_retry_delay)
        llm_output = dm_agent.generate(active_messages)
        print(colored(f"Assistant {step} (Retry {retry_idx}):", "red"), llm_output)
        try:
            return llm_output, parse_action(llm_output)
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse a valid Intention/Action after {format_retry_count} retries."
    ) from initial_error
