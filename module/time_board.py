import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class AgentTimeBoard:
    """Virtual timeline for action execution, dwell, and task-break logging."""

    def __init__(self, timing_data_path: str = "data/user_data.json"):
        self.timing_data_path = timing_data_path
        self.data = self._load_timing_data(timing_data_path)
        self.current_time = 0.0
        self.timeline: List[Dict[str, Any]] = []

    @staticmethod
    def _load_timing_data(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Timing data file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _round_time(value: float) -> float:
        return round(float(value), 2)

    @staticmethod
    def _normalize_name(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return re.sub(r"\s+", " ", str(value).strip().lower())

    @staticmethod
    def format_time(seconds: float) -> str:
        return f"{seconds:.2f}s"

    def format_step_calculation(self, timing_metadata: Optional[Dict[str, Any]]) -> str:
        if not timing_metadata:
            return "Current virtual time: 0.00s"

        events = timing_metadata.get("timing_events", [])
        step_label = events[0].get("timestep") if events else "?"
        lines = [
            f"Current virtual time after step {step_label}: {self.format_time(timing_metadata.get('current_time', 0.0))}",
            "Time calculation:",
        ]

        for event in events:
            begin = self.format_time(event.get("time_begin", 0.0))
            end = self.format_time(event.get("time_end", 0.0))
            duration = self.format_time(event.get("duration", 0.0))
            source = event.get("source", "unknown_source")
            kind = event.get("kind", "event")

            if kind == "action":
                task_name = event.get("task_name") or "unknown_task"
                action_name = event.get("action_name") or "unknown_action"
                action_text = event.get("action") or ""
                detail = f"action {action_name} ({source}: {task_name}/{action_text}, +{duration})"
            elif kind == "dwell":
                task_name = event.get("task_name") or "unknown_task"
                location = event.get("location") or "unknown_location"
                detail = f"dwell at {location} ({source}: {task_name}/{location}, +{duration})"
            elif kind == "break":
                task_before = event.get("from_task") or "unknown_task"
                task_after = event.get("to_task") or "unknown_task"
                detail = f"break {task_before}_to_{task_after} ({source}, +{duration})"
            else:
                detail = f"{kind} ({source}, +{duration})"

            lines.append(f"  {begin} -> {end}: {detail}")

        return "\n".join(lines)

    def _advance(
        self,
        event: Dict[str, Any],
        duration: float,
    ) -> Dict[str, Any]:
        begin = self.current_time
        end = begin + duration
        self.current_time = end
        event.update({
            "time_begin": self._round_time(begin),
            "time_end": self._round_time(end),
            "duration": self._round_time(duration),
        })
        self.timeline.append(event)
        return event

    def _lookup_action_duration(self, task_name: Optional[str], action_name: Optional[str]) -> Tuple[float, str]:
        if not task_name or not action_name:
            return 0.0, "missing_action"

        stats = (
            self.data
            .get("action_execution_time", {})
            .get(task_name, {})
            .get(action_name)
        )
        if not stats or stats.get("Avg") is None:
            return 0.0, "missing_action_stat"

        return float(stats["Avg"]), "action_execution_time.Avg"

    def _lookup_dwell_duration(self, task_name: Optional[str], location_name: Optional[str]) -> Tuple[float, str]:
        normalized_location = self._normalize_name(location_name)
        if not task_name or not normalized_location:
            return 0.0, "missing_location"

        stats = (
            self.data
            .get("dwell_time", {})
            .get(task_name, {})
            .get(normalized_location)
        )
        if not stats or stats.get("AVG") is None:
            return 0.0, "missing_dwell_stat"

        return float(stats["AVG"]), "dwell_time.AVG"

    def _lookup_break_duration(self, task_before: Optional[str], task_after: Optional[str]) -> Tuple[float, str]:
        if not task_before or not task_after or task_before == task_after:
            return 0.0, "no_transition"

        break_key = f"{task_before}_to_{task_after}"
        break_time = self.data.get("break_time", {}).get(break_key)
        if break_time is None:
            return 0.0, "missing_break_stat"

        return float(break_time), "break_time"

    def classify_action(self, action_text: str, observation: Optional[str] = None) -> Optional[str]:
        command = self._normalize_name(action_text) or ""
        if re.fullmatch(r"go to .+", command):
            return "TeleportFull"
        if re.fullmatch(r"take .+? from .+", command):
            return "PickupObject"
        if re.fullmatch(r"put .+? (in/on|in|on) .+", command):
            return "PutObject"
        if re.fullmatch(r"open .+", command):
            return "OpenObject"
        if re.fullmatch(r"close .+", command):
            return "CloseObject"
        if re.fullmatch(r"toggle .+", command):
            normalized_observation = self._normalize_name(observation) or ""
            if re.search(r"\boff\b", normalized_observation):
                return "ToggleObjectOff"
            return "ToggleObjectOn"
        return None

    def extract_navigation_target(self, action_text: str) -> Optional[str]:
        command = self._normalize_name(action_text) or ""
        match = re.fullmatch(r"go to (.+)", command)
        return match.group(1) if match else None

    def record_step(
        self,
        timestep: int,
        task_before: Optional[str],
        action_text: str,
        action_success: bool,
        location_after: Optional[str],
        task_after: Optional[str],
        observation: Optional[str] = None,
        use_dwell_time: bool = False,
    ) -> Dict[str, Any]:
        action_name = self.classify_action(action_text, observation=observation)
        action_duration, action_source = self._lookup_action_duration(task_before, action_name)
        step_begin = self.current_time
        events = []

        events.append(self._advance({
            "kind": "action",
            "timestep": timestep,
            "task_name": task_before,
            "action": action_text,
            "action_name": action_name,
            "success": action_success,
            "source": action_source,
        }, action_duration))

        navigation_target = self.extract_navigation_target(action_text)
        if use_dwell_time and action_name == "TeleportFull" and action_success:
            dwell_location = location_after or navigation_target
            dwell_duration, dwell_source = self._lookup_dwell_duration(task_before, dwell_location)
            events.append(self._advance({
                "kind": "dwell",
                "timestep": timestep,
                "task_name": task_before,
                "location": self._normalize_name(dwell_location),
                "source": dwell_source,
            }, dwell_duration))

        if task_after and task_before and task_after != task_before:
            break_duration, break_source = self._lookup_break_duration(task_before, task_after)
            events.append(self._advance({
                "kind": "break",
                "timestep": timestep,
                "from_task": task_before,
                "to_task": task_after,
                "source": break_source,
            }, break_duration))

        step_end = self.current_time
        return {
            "time_begin": self._round_time(step_begin),
            "time_end": self._round_time(step_end),
            "duration": self._round_time(step_end - step_begin),
            "current_time": self._round_time(self.current_time),
            "timing_events": events,
        }
