import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple


REARRANGEMENT_SUBTASK_END_STEPS: Dict[int, int] = {
    1: 13,
    2: 19,
    3: 24,
    4: 31,
    5: 34,
    6: 38,
    7: 43,
    8: 47,
    9: 52,
    10: 56,
    11: 62,
    12: 67,
    13: 75,
    14: 78,
    15: 79,
}


@dataclass
class StopperDecision:
    should_stop: bool
    reason: str = ""


@dataclass
class RearrangementStepStopper:
    expected_end_steps: Dict[int, int] = field(default_factory=lambda: dict(REARRANGEMENT_SUBTASK_END_STEPS))
    task_name: str = "rearrangement"
    completed_subtasks_seen: Set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.expected_subtask_by_step = {
            step: subtask_id
            for subtask_id, step in self.expected_end_steps.items()
        }

    def check(
        self,
        timestep: int,
        observation: str,
        subtask_done: bool,
        env: Optional[Any] = None,
    ) -> StopperDecision:
        completed = self._extract_completed_subtask(observation, env)
        expected_subtask_at_step = self.expected_subtask_by_step.get(timestep)
        current_task_name = getattr(env, "current_task_name", None)

        if completed:
            completed_task_name, completed_subtask_id = completed
            if completed_task_name == self.task_name:
                self.completed_subtasks_seen.add(completed_subtask_id)

                if (
                    expected_subtask_at_step is not None
                    and expected_subtask_at_step != completed_subtask_id
                ):
                    return StopperDecision(
                        should_stop=True,
                        reason=(
                            f"Step {timestep} should end subtask {expected_subtask_at_step} "
                            f"in the {self.task_name} task, but ended subtask {completed_subtask_id}."
                        ),
                    )

        if (
            expected_subtask_at_step is not None
            and expected_subtask_at_step not in self.completed_subtasks_seen
            and current_task_name == self.task_name
        ):
            return StopperDecision(
                should_stop=True,
                reason=(
                    f"Step {timestep} should end subtask {expected_subtask_at_step} "
                    f"in the {self.task_name} task, but no matching completion was observed."
                ),
            )

        return StopperDecision(should_stop=False)

    def _extract_completed_subtask(
        self,
        observation: str,
        env: Optional[Any],
    ) -> Optional[Tuple[str, int]]:
        explicit_match = re.search(
            r"You completed subtask\s+(\d+)\s+of the\s+([A-Za-z0-9_]+)\s+task",
            observation,
        )
        if explicit_match:
            return explicit_match.group(2), int(explicit_match.group(1))

        implicit_match = re.search(r"You completed subtask\s+(\d+)", observation)
        if implicit_match:
            current_task_name = getattr(env, "current_task_name", None)
            if current_task_name:
                return current_task_name, int(implicit_match.group(1))

        return None
