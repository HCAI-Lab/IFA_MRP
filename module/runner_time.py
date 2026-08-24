from typing import Any, Dict, Optional


def get_action_observation_time(
    metadata: Optional[Dict[str, Any]],
    time_board: Optional[Any],
    fallback_step: int,
) -> float:
    if metadata:
        for event in metadata.get("timing_events", []):
            if event.get("kind") == "action":
                return float(event.get("time_end", metadata.get("time_end", fallback_step)))
        return float(metadata.get("time_end", fallback_step))
    if time_board is not None:
        return float(time_board.current_time)
    return float(fallback_step)


def get_current_virtual_time(
    metadata: Optional[Dict[str, Any]],
    time_board: Optional[Any],
    fallback_step: int,
) -> float:
    if metadata:
        return float(metadata.get("current_time", metadata.get("time_end", fallback_step)))
    if time_board is not None:
        return float(time_board.current_time)
    return float(fallback_step)
