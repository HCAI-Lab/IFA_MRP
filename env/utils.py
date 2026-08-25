import re

def parse_action(llm_output: str) -> str:
    """ Parse the executable action from the LLM's raw output (matching Intention: or Action:). """
    if not isinstance(llm_output, str):
        raise ValueError("LLM output is not a valid string.")

    # Match Intention: or Action: (with optional markdown bold) case-insensitively
    match = re.search(r"\*{0,2}(?:Intention|Action)\*{0,2}\s*:\s*(.+)", llm_output, re.IGNORECASE)
    if match:
        action = match.group(1).strip()
        # Take the first line if multi-line output follows
        action = action.split("\n")[0].strip()
        # Remove trailing/leading markdown backticks or asterisks
        action = re.sub(r"^[`*]+|[`*]+$", "", action).strip()
        if action:
            return action

    # Fallback: search for action pattern lines (go to, take, put, open, close, toggle)
    action_match = re.search(r"\b(go to|take|put|open|close|toggle)\b.+", llm_output, re.IGNORECASE)
    if action_match:
        action = action_match.group(0).strip().split("\n")[0].strip()
        action = re.sub(r"^[`*]+|[`*]+$", "", action).strip()
        if action:
            return action

    raise ValueError(f"No valid Intention/Action found in LLM output:\n{llm_output}")