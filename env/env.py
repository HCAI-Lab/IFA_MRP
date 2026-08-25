import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
from ai2thor.controller import Controller

from .constants import RECEPTACLES
from .constraints import EXCLUDE, LOCATIONS
from .tasks import tasks


# Receptacles whose contents are normally described using "in".
# Other receptacles, such as CounterTop and DiningTable, use "on".
IN_RECEPTACLE_TYPES: Set[str] = {
    "Bathtub",
    "BathtubBasin",
    "Bowl",
    "Box",
    "Cabinet",
    "CoffeeMachine",
    "Cup",
    "Drawer",
    "Fridge",
    "GarbageCan",
    "LaundryHamper",
    "Microwave",
    "Mug",
    "Pan",
    "Pot",
    "Safe",
    "Sink",
    "SinkBasin",
    "Toaster",
    "Toilet",
}

RECEPTACLE_REMINDER = "* You must close any opened receptacles right away after using them to complete the task."


def append_receptacle_reminder(subtask_text: str) -> str:
    subtask_text = subtask_text.strip()
    if subtask_text and subtask_text[-1] not in ".!?:": # add colon as well
        subtask_text = f"{subtask_text}."
    return f"{subtask_text} {RECEPTACLE_REMINDER}"



def assign_unique_ids(controller: Controller) -> Dict[str, str]:
    unique_id_map: Dict[str, str] = {}
    category_counts: Dict[str, int] = {}
    for obj in controller.last_event.metadata.get("objects", []):
        object_type = obj.get("objectType")
        object_id = obj.get("objectId")
        if object_type in {None, "Floor"} or object_id is None:
            continue

        category_counts[object_type] = category_counts.get(object_type, 0) + 1
        unique_id_map[object_id] = f"{object_type} {category_counts[object_type]}"

    return unique_id_map


def format_subtask_description(subtask_info: Dict[str, Any]) -> str:
    """
    Format subtask description by augmenting receptacle mentions with exact receptacle names
    from the subtask's trajectories (e.g. 'drawer 3', 'cabinet 8').
    """
    raw_subtask = subtask_info.get("subtask", "")
    subtask_text = raw_subtask.translate(str.maketrans("‘’“”", "''\"\""))
    trajectories = subtask_info.get("trajectories", [])
    if not trajectories:
        return subtask_text

    source_recep = None
    dest_recep = None

    for action in trajectories:
        m_take = re.search(r"take\s+.+?\s+from\s+([a-zA-Z]+\s+\d+)", action, re.IGNORECASE)
        if m_take:
            source_recep = m_take.group(1).lower()

        m_put = re.search(r"put\s+.+?\s+in/on\s+([a-zA-Z]+\s+\d+)", action, re.IGNORECASE)
        if m_put:
            dest_recep = m_put.group(1).lower()

    if source_recep and dest_recep:
        parts = re.split(r"(\s+and\s+place\s+|\s+and\s+)", subtask_text, maxsplit=1)
        if len(parts) == 3:
            part1, joiner, part2 = parts
            src_type = source_recep.split()[0]
            if src_type in part1.lower():
                part1 = re.sub(rf"\b({src_type})\b", rf"\1 ({source_recep})", part1, count=1, flags=re.IGNORECASE)

            dst_type = dest_recep.split()[0]
            if dst_type in part2.lower():
                part2 = re.sub(rf"\b({dst_type})\b", rf"\1 ({dest_recep})", part2, count=1, flags=re.IGNORECASE)
            return part1 + joiner + part2

    if dest_recep:
        recep_type = dest_recep.split()[0]
        if recep_type == "stoveburner":
            if "stove burners" in subtask_text.lower():
                return re.sub(r"\b(stove burners)\b", f"\\1 ({dest_recep})", subtask_text, count=1, flags=re.IGNORECASE)
            elif "stove" in subtask_text.lower():
                return re.sub(r"\b(stove)\b", f"\\1 ({dest_recep})", subtask_text, count=1, flags=re.IGNORECASE)

        if recep_type in subtask_text.lower():
            return re.sub(rf"\b({recep_type})\b", rf"\1 ({dest_recep})", subtask_text, count=1, flags=re.IGNORECASE)

    return subtask_text


TASK_ORDER: Tuple[str, ...] = ("exploration", "rearrangement", "findNplace")


class HouseholdEnvironment:
    def __init__(self, scene: str = "FloorPlan2", field_of_view: float = 90.0):
        self._validate_scene(scene)
        self.controller = Controller(
            scene=scene,
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=90.0,
            width=300,
            height=300,
            fieldOfView=field_of_view,
        )
        self.scene = scene
        self.current_location: Optional[str] = None
        self.inventory: List[Dict[str, Any]] = []
        self.unique_id_map: Dict[str, str] = assign_unique_ids(self.controller)
        self.current_goal: Optional[Dict[str, Any]] = None
        self.task_order: Tuple[str, ...] = TASK_ORDER
        self.current_task_index: int = 0
        self.current_task_name: Optional[str] = None
        self.current_subtask_id: int = 1
        self.subtask_ids: List[int] = []
        self.completed_subtasks: Dict[str, Set[int]] = {task_name: set() for task_name in tasks}
    
    def _validate_scene(self, scene: str) -> None:
        if scene not in LOCATIONS:
            raise ValueError(f"Unknown scene '{scene}'. Available scenes: {list(LOCATIONS)}")
        
        
    ## Public interface ============================================================
    def reset(self, scene: Optional[str] = None, task_name: Optional[str] = None) -> str:
        if scene:
            self._validate_scene(scene)
            self.scene = scene

        event = self.controller.reset(scene=self.scene)
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(event.metadata.get("errorMessage", f"Failed to reset {self.scene}"))

        self.current_location = None
        self.inventory = []
        self.unique_id_map = assign_unique_ids(self.controller)
        self.current_goal = None
        self.completed_subtasks = {task_name: set() for task_name in tasks}

        initial_obs = self._get_initial_observation()
        target_task = task_name if (task_name and task_name in tasks) else self.task_order[0]
        task_msg = self.set_task(target_task, start_subtask_id=1)
        return f"{initial_obs} {task_msg}"

    def set_task(self, task_name: str, start_subtask_id: int = 1) -> str:
        """Set active task (exploration, rearrangement, findNplace) and initialize subtasks in order."""
        if task_name not in tasks:
            raise ValueError(f"Unknown task '{task_name}'. Available tasks: {list(tasks.keys())}")
        if task_name in self.task_order:
            self.current_task_index = self.task_order.index(task_name)
        self.current_task_name = task_name
        self.subtask_ids = sorted(tasks[task_name].keys())
        if start_subtask_id not in tasks[task_name]:
            start_subtask_id = self.subtask_ids[0]
        self.current_subtask_id = start_subtask_id

        subtask_info = tasks[task_name][start_subtask_id]
        try:
            self.set_subtask_goal((task_name, start_subtask_id))
        except (FileNotFoundError, ValueError):
            goal_state = self.get_subtask_metadata(subtask_info["trajectories"])
            self.set_subtask_goal(goal_state)

        subtask_text = format_subtask_description(subtask_info)
        if task_name in ["rearrangement", "findNplace"]:
            subtask_text = append_receptacle_reminder(subtask_text)
        return f"The {task_name} task begins. Your first subtask is to: {subtask_text}"

    def next_subtask(self) -> Tuple[bool, str, str]:
        """
        Advance to the next subtask in order, and automatically transition to the next task when a task finishes.
        Returns (has_next, transition_type, message).
        transition_type is "same_task", "new_task", or "none".
        """
        if not self.current_task_name or not self.subtask_ids:
            return False, "none", "No active task set."

        current_idx = self.subtask_ids.index(self.current_subtask_id) if self.current_subtask_id in self.subtask_ids else -1
        next_idx = current_idx + 1

        if next_idx < len(self.subtask_ids):
            self.current_subtask_id = self.subtask_ids[next_idx]
            subtask_info = tasks[self.current_task_name][self.current_subtask_id]
            try:
                self.set_subtask_goal((self.current_task_name, self.current_subtask_id))
            except (FileNotFoundError, ValueError):
                goal_state = self.get_subtask_metadata(subtask_info["trajectories"])
                self.set_subtask_goal(goal_state)
            subtask_text = format_subtask_description(subtask_info)
            if self.current_task_name in ["rearrangement", "findNplace"]:
                subtask_text = append_receptacle_reminder(subtask_text)
            return True, "same_task", f"Your next subtask is to: {subtask_text}"

        next_task_index = self.current_task_index + 1
        if next_task_index < len(self.task_order):
            next_task_name = self.task_order[next_task_index]
            self.set_task(next_task_name, start_subtask_id=1)
            subtask_info = tasks[self.current_task_name][self.current_subtask_id]
            subtask_text = format_subtask_description(subtask_info)
            if next_task_name in ["rearrangement", "findNplace"]:
                subtask_text = append_receptacle_reminder(subtask_text)
            return True, "new_task", f"The {next_task_name} task begins. Your first subtask is to: {subtask_text}"
        else:
            self.current_goal = None
            return False, "none", "You have completed all tasks."

    @property
    def current_subtask(self) -> Optional[Dict[str, Any]]:
        """Return details of the current active subtask."""
        if not self.current_task_name or self.current_subtask_id not in tasks.get(self.current_task_name, {}):
            return None
        subtask_info = tasks[self.current_task_name][self.current_subtask_id]
        total_sub = len(self.subtask_ids)
        total_tasks = len(self.task_order)
        idx = self.subtask_ids.index(self.current_subtask_id) if self.current_subtask_id in self.subtask_ids else 1
        return {
            "task_name": self.current_task_name,
            "task_index": self.current_task_index + 1,
            "total_tasks": total_tasks,
            "subtask_id": self.current_subtask_id,
            "subtask_index": idx,
            "total_subtasks": total_sub,
            "subtask": format_subtask_description(subtask_info),
            "trajectories": subtask_info["trajectories"],
        }

    ## Task and subtask completion evaluation ============================================
    def get_task_completion(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the completion status of a specific task (or current active task if None).
        Provides task-specific subtask count separately (e.g. 7 for exploration, 15 for rearrangement, 8 for findNplace).
        """
        target_task = task_name or self.current_task_name
        if not target_task or target_task not in tasks:
            return {"task_name": target_task, "error": "Unknown or inactive task"}

        all_subtask_ids = sorted(tasks[target_task].keys())
        total_subtasks = len(all_subtask_ids)
        completed_ids = sorted(list(self.completed_subtasks.get(target_task, set())))
        completed_count = len(completed_ids)
        is_complete = (completed_count == total_subtasks and total_subtasks > 0)
        percentage = (completed_count / total_subtasks * 100.0) if total_subtasks > 0 else 0.0

        return {
            "task_name": target_task,
            "completed_subtask_ids": completed_ids,
            "completed_subtasks": completed_count,
            "total_subtasks": total_subtasks,
            "percentage": round(percentage, 1),
            "is_complete": is_complete,
            "status": f"{completed_count}/{total_subtasks} subtasks ({percentage:.1f}%)",
        }

    def get_overall_completion(self) -> Dict[str, Any]:
        """
        Return overall completion status across all tasks ("whole tasks"),
        providing task breakdown with separate subtask counts for each task.
        """
        task_details = {}
        completed_tasks_count = 0

        for t_name in self.task_order:
            t_info = self.get_task_completion(t_name)
            task_details[t_name] = t_info
            if t_info.get("is_complete", False):
                completed_tasks_count += 1

        total_tasks_count = len(self.task_order)
        overall_percentage = (
            (completed_tasks_count / total_tasks_count * 100.0)
            if total_tasks_count > 0
            else 0.0
        )
        is_all_complete = (
            completed_tasks_count == total_tasks_count
            and total_tasks_count > 0
        )

        per_task_status = ", ".join([f"'{t}': {info['status']}" for t, info in task_details.items()])

        return {
            "task_breakdown": task_details,
            "completed_tasks_count": completed_tasks_count,
            "total_tasks_count": total_tasks_count,
            "completed_tasks": [t for t, info in task_details.items() if info.get("is_complete")],
            "overall_percentage": round(overall_percentage, 1),
            "is_all_complete": is_all_complete,
            "status": f"Completed Tasks: {completed_tasks_count}/{total_tasks_count} ({overall_percentage:.1f}%) | Per-task: [{per_task_status}]",
        }

    def get_completion_summary(self) -> str:
        """
        Return a formatted human-readable summary of completion separately for each task and whole tasks.
        """
        overall = self.get_overall_completion()
        lines = [f"Overall Progress: {overall['completed_tasks_count']}/{overall['total_tasks_count']} tasks completed ({overall['overall_percentage']}%)"]
        lines.append("Per-Task Subtask Progress:")
        for t_name, t_info in overall["task_breakdown"].items():
            symbol = "✓" if t_info.get("is_complete") else " "
            lines.append(f"  [{symbol}] Task '{t_name}': {t_info.get('status')}")
        return "\n".join(lines)

    def evaluate_task(self, task_name: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if a specific task (or current active task if None) is completed.
        Returns (is_complete, task_completion_dict).
        """
        info = self.get_task_completion(task_name)
        return info.get("is_complete", False), info

    def evaluate_all_tasks(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if all tasks ("whole tasks") are completed.
        Returns (is_all_complete, overall_completion_dict).
        """
        info = self.get_overall_completion()
        return info.get("is_all_complete", False), info

    @property
    def completion_status(self) -> Dict[str, Any]:
        """Property returning overall completion status dictionary."""
        return self.get_overall_completion()

    @property
    def is_current_task_complete(self) -> bool:
        """Return True if the current task is fully complete."""
        return self.get_task_completion().get("is_complete", False)

    @property
    def is_all_tasks_complete(self) -> bool:
        """Return True if all tasks are completed."""
        return self.get_overall_completion()["is_all_complete"]

    def _with_frame(self, result):
        obs, done, success = result
        if self.current_goal is not None:
            is_complete, _ = self.evaluate_subtask()
            done = is_complete
            if is_complete:
                completed_subtask_id = self.current_subtask_id
                completed_task_name = self.current_task_name
                if completed_task_name:
                    self.completed_subtasks.setdefault(completed_task_name, set()).add(completed_subtask_id)

                has_next, transition_type, next_msg = self.next_subtask()
                if has_next:
                    if transition_type == "new_task":
                        obs = f"{obs} You completed subtask {completed_subtask_id} of the {completed_task_name} task. {next_msg}"
                    else:
                        obs = f"{obs} You completed subtask {completed_subtask_id}. {next_msg}"
                else:
                    obs = f"{obs} You completed subtask {completed_subtask_id} of the {completed_task_name} task. {next_msg}"
        return (obs, done, success, self.frame.copy())

    def get_subtask_metadata(self, trajectories: List[str]) -> Dict[str, Any]:
        """Return task-relevant metadata (agent state, target objects, target receptacles) for a subtask."""
        event = self.controller.last_event
        metadata = event.metadata if event else {}
        agent_info = metadata.get("agent", {})
        all_objects = metadata.get("objects", [])

        target_queries = set()
        for command in trajectories:
            cmd = command.strip().lower()
            m = re.fullmatch(r"take (.+?) from (.+)", cmd)
            if m:
                target_queries.add(m.group(1))
                target_queries.add(m.group(2))
                continue
            m = re.fullmatch(r"put (.+?) (?:in/on|in|on) (.+)", cmd)
            if m:
                target_queries.add(m.group(1))
                target_queries.add(m.group(2))
                continue
            m = re.fullmatch(r"(?:open|close|toggle) (.+)", cmd)
            if m:
                target_queries.add(m.group(1))
                continue
            m = re.fullmatch(r"go to (.+)", cmd)
            if m:
                target_queries.add(m.group(1))
                continue

        matched_object_ids = set()
        for query in target_queries:
            norm_query = self._normalize_identifier(query)
            for obj in all_objects:
                obj_id = obj.get("objectId")
                if not obj_id or obj.get("objectType") == "Floor":
                    continue

                unique_name = self.unique_id_map.get(obj_id, obj.get("objectType", ""))
                norm_unique = self._normalize_identifier(unique_name)
                norm_type = self._normalize_identifier(obj.get("objectType", ""))

                if norm_query in (norm_unique, norm_type) or norm_unique.startswith(norm_query):
                    matched_object_ids.add(obj_id)

        additional_parents = set()
        for obj in all_objects:
            if obj.get("objectId") in matched_object_ids:
                for parent_id in (obj.get("parentReceptacles") or []):
                    additional_parents.add(parent_id)
        matched_object_ids.update(additional_parents)

        target_objects_data = {}
        for obj in all_objects:
            obj_id = obj.get("objectId")
            if obj_id in matched_object_ids:
                unique_name = self.unique_id_map.get(obj_id, obj.get("objectType", ""))
                parents = obj.get("parentReceptacles") or []
                parent_names = [self.unique_id_map.get(p_id, p_id) for p_id in parents if p_id in self.unique_id_map]

                target_objects_data[unique_name] = {
                    "objectId": obj_id,
                    "objectType": obj.get("objectType"),
                    "position": obj.get("position"),
                    "rotation": obj.get("rotation"),
                    "parentReceptacles": parents,
                    "parentReceptacleNames": parent_names,
                    "isOpen": obj.get("isOpen", False),
                    "openable": obj.get("openable", False),
                    "isToggled": obj.get("isToggled", False),
                    "toggleable": obj.get("toggleable", False),
                    "isPickedUp": obj.get("isPickedUp", False),
                    "pickupable": obj.get("pickupable", False),
                }

        inventory_names = [
            self.unique_id_map.get(item.get("objectId"), item.get("objectType", ""))
            for item in self.inventory
        ]

        source_receptacle_names = set()
        dest_receptacle_names = set()
        for command in trajectories:
            cmd = command.strip().lower()
            m_take = re.fullmatch(r"take (.+?) from (.+)", cmd)
            if m_take:
                source_receptacle_names.add(m_take.group(2).strip())
            m_put = re.fullmatch(r"put (.+?) (?:in/on|in|on) (.+)", cmd)
            if m_put:
                dest_receptacle_names.add(m_put.group(2).strip())

        return {
            "scene": self.scene,
            "agent": {
                "current_location": self.current_location,
                "position": agent_info.get("position"),
                "rotation": agent_info.get("rotation"),
                "inventory": inventory_names,
            },
            "target_objects": target_objects_data,
            "source_receptacle_names": list(source_receptacle_names),
            "dest_receptacle_names": list(dest_receptacle_names),
        }

    def set_subtask_goal(self, goal: Union[Dict[str, Any], str, Tuple[str, int], List[Any]]) -> None:
        """
        Set the target subtask goal for evaluation.
        `goal` can be:
        - A metadata dictionary containing "state" or "agent"/"target_objects"
        - A JSON file path (e.g. "rearrangement_metadata.json")
        - A tuple/list (task_name, subtask_id), which automatically loads from `{task_name}_metadata.json`
        """
        if isinstance(goal, (list, tuple)) and len(goal) == 2:
            task_name, subtask_id = goal
            pkg_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(pkg_dir, "metadata", f"{task_name}_metadata.json"),
                os.path.join("env", "metadata", f"{task_name}_metadata.json"),
                os.path.join("metadata", f"{task_name}_metadata.json"),
                f"{task_name}_metadata.json",
            ]
            path = next((cand for cand in candidates if os.path.exists(cand)), None)
            if not path:
                raise FileNotFoundError(f"Metadata file for task '{task_name}' not found.")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            matching = [item for item in data if item.get("subtask_id") == subtask_id]
            if not matching:
                raise ValueError(f"Subtask ID {subtask_id} not found in {path}")
            self.current_goal = matching[0].get("state", matching[0])
        elif isinstance(goal, str):
            pkg_dir = os.path.dirname(__file__)
            candidates = [
                goal,
                os.path.join(pkg_dir, "metadata", goal),
                os.path.join("env", "metadata", goal),
                os.path.join("metadata", goal),
            ]
            path = next((cand for cand in candidates if os.path.exists(cand)), None)
            if not path:
                raise FileNotFoundError(f"Goal file '{goal}' not found.")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.current_goal = data.get("state", data)
        elif isinstance(goal, dict):
            self.current_goal = goal.get("state", goal)
        else:
            raise TypeError(f"Unsupported goal type: {type(goal)}")

    def evaluate_subtask(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if the current environment state satisfies the active subtask goal.
        Returns (is_complete: bool, details: Dict[str, Any]).
        """
        if self.current_goal is None:
            return False, {"reason": "No subtask goal set"}

        goal_agent = self.current_goal.get("agent", {})
        goal_targets = self.current_goal.get("target_objects", {})
        details: Dict[str, Any] = {"criteria": []}
        all_passed = True

        # 1. Check agent location if specified in goal
        target_location = goal_agent.get("current_location")
        if target_location:
            loc_match = (
                self.current_location is not None
                and self._normalize_identifier(self.current_location) == self._normalize_identifier(target_location)
            )
            details["criteria"].append({
                "type": "location",
                "expected": target_location,
                "actual": self.current_location,
                "passed": loc_match,
            })
            if not loc_match:
                all_passed = False

        # 2. Check target object / receptacle states
        last_event = self.controller.last_event
        all_objects = {
            self.unique_id_map.get(obj["objectId"], obj.get("objectType", "")): obj
            for obj in (last_event.metadata.get("objects", []) if last_event else [])
            if obj.get("objectId")
        }

        for target_name, target_state in goal_targets.items():
            current_obj = all_objects.get(target_name)
            if not current_obj:
                obj_type = target_state.get("objectType")
                matching = [o for o in all_objects.values() if o.get("objectType") == obj_type]
                current_obj = matching[0] if matching else None

            if not current_obj:
                details["criteria"].append({
                    "object": target_name,
                    "passed": False,
                    "reason": "Object not found in scene",
                })
                all_passed = False
                continue

            # Check parent receptacles
            expected_parent_names = target_state.get("parentReceptacleNames", [])
            if expected_parent_names:
                current_parents = current_obj.get("parentReceptacles") or []
                current_parent_names = [
                    self.unique_id_map.get(p_id, p_id)
                    for p_id in current_parents
                    if p_id in self.unique_id_map
                ]

                exp_norm = {self._normalize_identifier(p) for p in expected_parent_names}
                act_norm = {self._normalize_identifier(p) for p in current_parent_names}
                receptacle_match = bool(exp_norm.intersection(act_norm))

                details["criteria"].append({
                    "object": target_name,
                    "type": "parent_receptacle",
                    "expected": expected_parent_names,
                    "actual": current_parent_names,
                    "passed": receptacle_match,
                })
                if not receptacle_match:
                    all_passed = False

            # Check isOpen (for container receptacles like microwave/drawer)
            if "isOpen" in target_state and target_state.get("openable"):
                expected_open = target_state["isOpen"]
                actual_open = current_obj.get("isOpen", False)
                open_match = (expected_open == actual_open)
                details["criteria"].append({
                    "object": target_name,
                    "type": "isOpen",
                    "expected": expected_open,
                    "actual": actual_open,
                    "passed": open_match,
                })
                if not open_match:
                    all_passed = False

            # Check isToggled (for toggleable items like faucet)
            if "isToggled" in target_state and target_state.get("toggleable"):
                expected_toggled = target_state["isToggled"]
                actual_toggled = current_obj.get("isToggled", False)
                toggled_match = (expected_toggled == actual_toggled)
                details["criteria"].append({
                    "object": target_name,
                    "type": "isToggled",
                    "expected": expected_toggled,
                    "actual": actual_toggled,
                    "passed": toggled_match,
                })
                if not toggled_match:
                    all_passed = False

        return all_passed, details
    
    def step(self, text_command: str) -> Tuple[str, bool, bool, Any]:
        """
        Execute one of the following command formats:

            1. go to {recep}
            2. take {obj} from {recep}
            3. put {obj} in/on {recep}
            4. open {recep}
            5. close {recep}
            6. toggle {obj/recep}

        Returns:
            observation, done, success, frame
        """
        command = self._normalize_command(text_command)

        if not command:
            return self._with_frame(("Nothing happens.", False, False))
            
        try:
            # 1. go to {recep}
            navigation_match = re.fullmatch(r"go to (.+)", command)
            if navigation_match:
                location_name = navigation_match.group(1)
                return self._with_frame(self._go_to_location(location_name))

            # 2. take {obj} from {recep}
            take_match = re.fullmatch(r"take (.+?) from (.+)", command)
            if take_match:
                object_name, receptacle_name = take_match.groups()
                return self._with_frame(self._pickup_object(target_name=object_name, source_receptacle_name=receptacle_name))

            # 3. put {obj} in/on {recep}
            put_match = re.fullmatch(r"put (.+?) (in/on|in|on) (.+)", command)
            if put_match:
                object_name, preposition, receptacle_name = put_match.groups()
                return self._with_frame(self._put_object(object_name, preposition, receptacle_name))
            
            # 4. open {recep}
            open_match = re.fullmatch(r"open (.+)", command)
            if open_match:
                return self._with_frame(self._toggle_container(open_match.group(1), open_action=True))

            # 5. close {recep}
            close_match = re.fullmatch(r"close (.+)", command)
            if close_match:
                return self._with_frame(self._toggle_container(close_match.group(1), open_action=False))

            # 6. toggle {obj/recep}
            toggle_match = re.fullmatch(r"toggle (.+)", command)
            if toggle_match:
                return self._with_frame(self._toggle_object(toggle_match.group(1)))

            return self._with_frame(("Nothing happens.", False, False))
        except (TimeoutError, RuntimeError, Exception) as e:
            print(f"⚠️ [AI2-THOR Step Error]: {e}")
            return self._with_frame(("Nothing happens.", False, False))


    def close(self) -> None:
        """Stop the AI2-THOR controller."""
        self.controller.stop()

    @property
    def frame(self):
        """Return the most recent RGB frame."""
        return self.controller.last_event.frame

    @property
    def available_locations(self) -> List[str]:
        """Return named navigation locations in the current scene."""
        return list(LOCATIONS[self.scene].keys())


    ## Navigation ============================================================
    def _go_to_location(self, location_query: str) -> Tuple[str, bool, bool]:
        """Teleport to a predefined location and return an ALFWorld-style observation."""
        location_name = self._resolve_location_name(location_query)
        if location_name is None:
            return "Nothing happens", False, False

        event = self.controller.step(**LOCATIONS[self.scene][location_name])
        if not event.metadata.get("lastActionSuccess", False):
            return "Nothing happens.", False, False

        self.current_location = location_name
        self._sync_inventory(event)
        observation = self._get_text_observation(location_name)
        return observation, False, True


    def _resolve_location_name(self, location_query: str) -> Optional[str]:
        """
        Resolve a case-insensitive location query.
        Examples: 
            'table 1' -> 'Table 1'
            'microwave & coffee machine' -> 'Microwave&CoffeeMachine'
        """
        normalized_query = self._normalize_identifier(location_query)
        for location_name in self.available_locations:
            if self._normalize_identifier(location_name) == normalized_query:
                return location_name

        return None


    ## Observation generation ============================================================
    def _get_initial_observation(self) -> str:
        """Return the observation before visiting a named location."""
        locations = [location.lower() for location in self.available_locations]
        location_description = self._format_plain_list(locations)
        return f"You have entered {self.scene.lower()}. You can go to {location_description}."
        # return (f"You are in the room with {len(locations)} available locations: {location_description}. You should complete 15 subtasks to complete a rearrangment task.\nYour first subtask is to: {}")


    def _get_text_observation(self, location_name: str) -> str:
        """ Create a receptacle-aware ALFWorld-style observation. """
        self._sync_inventory(self.controller.last_event)
        all_objects = self.controller.last_event.metadata.get("objects", [])
        objects_by_id = {obj["objectId"]: obj for obj in all_objects if obj.get("objectId")}
        interactable_objects = self._get_filtered_interactable_objects(location_name, all_objects)
        return self._create_alfworld_observation(location_name, interactable_objects, objects_by_id)


    def _get_current_position_interactable_objects(self) -> List[Dict[str, Any]]:
        """Return interactable object records for the agent's current named location."""
        if self.current_location is None:
            return []

        last_event = getattr(self.controller, "last_event", None)
        all_objects = last_event.metadata.get("objects", []) if last_event else []
        return self._get_filtered_interactable_objects(self.current_location, all_objects)


    def get_current_position_object_names(self) -> List[str]:
        """Return canonical object names currently available at the agent's position."""
        object_names = []
        seen_names = set()
        for item in self._get_current_position_interactable_objects():
            name = item["unique_name"].lower()
            if name in seen_names:
                continue
            seen_names.add(name)
            object_names.append(name)

        return object_names


    def get_current_position_affordances(self) -> List[Dict[str, Any]]:
        """Return openable/pickupable metadata for objects at the current position."""
        affordances = []
        seen_names = set()

        for item in self._get_current_position_interactable_objects():
            obj = item["metadata"]
            name = item["unique_name"].lower()
            if name in seen_names:
                continue

            openable = bool(obj.get("openable", False))
            pickupable = bool(obj.get("pickupable", False))
            if not openable and not pickupable:
                continue

            seen_names.add(name)
            record = {
                "name": name,
                "openable": openable,
                "pickupable": pickupable,
                "isOpen": bool(obj.get("isOpen", False)),
                "distance": obj.get("distance", float("inf")),
            }
            affordances.append(record)

        return sorted(affordances, key=lambda item: (item["distance"], item["name"]))


    def get_observation_affordances(self, observation: str) -> List[Dict[str, Any]]:
        """Return current-position affordances; observation text is ignored to avoid task-text leakage."""
        return self.get_current_position_affordances()


    def format_observation_affordances(self, observation: str) -> str:
        affordances = self.get_observation_affordances(observation)
        openable_count = sum(1 for item in affordances if item["openable"])
        pickupable_count = sum(1 for item in affordances if item["pickupable"])
        both_count = sum(1 for item in affordances if item["openable"] and item["pickupable"])
        total_count = len(affordances)

        lines = [
            f"Object affordances (openable: {openable_count} & pickupable: {pickupable_count} & both: {both_count} & Total : {total_count})"
        ]
        for item in affordances:
            parts = [
                f"openable={item['openable']}",
                f"isOpen={item['isOpen']}",
                f"pickupable={item['pickupable']}",
            ]
            lines.append(f"- {item['name']}: {', '.join(parts)}")

        return "\n".join(lines)


    def calculate_observation_entropy(self, observation: str) -> Dict[str, Any]:
        affordances = self.get_observation_affordances(observation)
        num_openable_obj = sum(1 for item in affordances if item["openable"])
        num_pickupable_obj = sum(1 for item in affordances if item["pickupable"])
        num_both_obj = sum(1 for item in affordances if item["openable"] and item["pickupable"])
        total_unique_obj = len(affordances)
        total_obj = num_openable_obj + num_pickupable_obj
        h_j_openable_pickupable = math.log(total_obj) if total_obj > 0 else 0.0

        return {
            "num_openable_obj": num_openable_obj,
            "num_pickupable_obj": num_pickupable_obj,
            "num_both_obj": num_both_obj,
            "total_unique_obj": total_unique_obj,
            "total_obj": total_obj,
            "h_j_openable_pickupable": h_j_openable_pickupable,
        }


    def format_observation_entropy(self, observation: str) -> str:
        entropy_info = self.calculate_observation_entropy(observation)
        total_obj = entropy_info["total_obj"]
        entropy_value = entropy_info["h_j_openable_pickupable"]
        formula = f"log({total_obj})" if total_obj > 0 else "0"

        return (
            "Scene entropy: "
            f"H_j_openable_pickupable={entropy_value:.3f} "
            f"({formula}; "
            f"openable={entropy_info['num_openable_obj']}, "
            f"pickupable={entropy_info['num_pickupable_obj']}, "
            f"both={entropy_info['num_both_obj']}, "
            f"total_obj={total_obj}, "
            f"total_unique={entropy_info['total_unique_obj']})"
        )


    def _get_filtered_interactable_objects(self, location_name: str, all_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Return non-excluded interactable objects. """
        excluded_names = set(EXCLUDE.get(self.scene, {}).get(location_name, []))
        held_object_ids = self._get_held_object_ids()
        interactable_objects = []
        for obj in all_objects:
            object_id = obj.get("objectId")
            unique_name = self.unique_id_map.get(object_id)
            if (
                not obj.get("isInteractable", False)
                or obj.get("objectType") == "Floor"
                or object_id is None
                or unique_name is None
                or unique_name in excluded_names
                or object_id in held_object_ids
                or obj.get("isPickedUp", False)
            ):
                continue

            interactable_objects.append({"metadata": obj, "unique_name": unique_name})

        return sorted(interactable_objects, key=lambda item: item["metadata"].get("distance", float("inf")))
    
    

    def _create_alfworld_observation(self, location_name: str, interactable_objects: List[Dict[str, Any]], objects_by_id: Dict[str, Dict[str, Any]]) -> str:
        """
        Group interactable objects under their nearest interactable parent receptacle.
        Example:
            You arrive at table 1. On the countertop 2, you see a tomato 1, a mug 1, an egg 1, and a bread 1.
        """
        if not interactable_objects:
            sentences = [f"You arrive at {location_name.lower()}."]
            inventory_observation = self._get_inventory_observation()
            if inventory_observation:
                sentences.append(inventory_observation)
            sentences.append("You do not see any interactable objects nearby.")
            return " ".join(sentences)

        interactable_ids = {item["metadata"]["objectId"] for item in interactable_objects}
        children_by_receptacle: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        standalone_objects: List[Dict[str, Any]] = []
        for item in interactable_objects:
            obj = item["metadata"]

            parent_id = self._get_nearest_interactable_parent(
                obj=obj,
                interactable_ids=interactable_ids,
                objects_by_id=objects_by_id,
            )

            if parent_id is None:
                standalone_objects.append(item)
            else:
                children_by_receptacle[parent_id].append(item)

        ## Receptacles used as sentence headings should not also appear in the standalone object list.
        used_receptacle_ids = set(children_by_receptacle.keys())
        standalone_objects = [
            item for item in standalone_objects
            if item["metadata"]["objectId"] not in used_receptacle_ids
        ]
        sentences = [f"You arrive at {location_name.lower()}."]
        inventory_observation = self._get_inventory_observation()
        if inventory_observation:
            sentences.append(inventory_observation)

        ## Outer receptacles first, nested receptacles afterward.
        sorted_receptacle_ids = sorted(
            children_by_receptacle.keys(),
            key=lambda receptacle_id: (
                self._get_receptacle_depth(receptacle_id, objects_by_id),
                objects_by_id[receptacle_id].get("distance", float("inf")),
            ),
        )

        for receptacle_id in sorted_receptacle_ids:
            receptacle_metadata = objects_by_id.get(receptacle_id)
            if receptacle_metadata is None:
                continue

            receptacle_name = self.unique_id_map.get(receptacle_id)
            if receptacle_name is None:
                continue

            contained_items = sorted(
                children_by_receptacle[receptacle_id],
                key=lambda item: item["metadata"].get("distance", float("inf")),
            )
            contained_names = [item["unique_name"] for item in contained_items]
            preposition = self._get_receptacle_preposition(receptacle_metadata)
            sentences.append(f"{preposition.capitalize()} the {receptacle_name.lower()}, you see {self._format_object_list(contained_names)}.")

        if standalone_objects:
            standalone_objects.sort(key=lambda item: item["metadata"].get("distance", float("inf")))
            standalone_names = [item["unique_name"] for item in standalone_objects]
            standalone_description = self._format_object_list(standalone_names)
            if children_by_receptacle:
                sentences.append(f"Looking around you, you also see {standalone_description}.")
            else:
                sentences.append(f"Looking around you, you see {standalone_description}.")

        return " ".join(sentences)



    def _get_nearest_interactable_parent(self, obj: Dict[str, Any], interactable_ids: Set[str], objects_by_id: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Return the deepest interactable parent receptacle.
        For example, when an apple is in a bowl and the bowl is on a countertop, Bowl is selected as Apple's nearest parent.
        """
        parent_ids = obj.get("parentReceptacles") or []
        candidate_parent_ids = []
        for parent_id in parent_ids:
            if parent_id not in interactable_ids:
                continue

            parent_metadata = objects_by_id.get(parent_id)
            if parent_metadata is None:
                continue

            if not self._is_receptacle(parent_metadata):
                continue

            candidate_parent_ids.append(parent_id)

        if not candidate_parent_ids:
            return None

        return max(candidate_parent_ids, key=lambda parent_id: self._get_receptacle_depth(parent_id, objects_by_id))


    def _get_receptacle_depth(self, receptacle_id: str, objects_by_id: Dict[str, Dict[str, Any]]) -> int:
        """ Estimate receptacle nesting depth using parentReceptacles. """
        receptacle = objects_by_id.get(receptacle_id, {})
        return len(receptacle.get("parentReceptacles") or [])


    def _is_receptacle(self, obj: Dict[str, Any]) -> bool:
        """ Check both AI2-THOR metadata and constants.RECEPTACLES. """
        return bool(obj.get("receptacle", False) or obj.get("objectType") in RECEPTACLES)


    def _get_receptacle_preposition(self, receptacle_metadata: Dict[str, Any],) -> str:
        """ Return 'in' for container-like receptacles and 'on' for surface-like receptacles. """
        receptacle_type = receptacle_metadata.get("objectType", "")
        if receptacle_type in IN_RECEPTACLE_TYPES:
            return "in"
        
        return "on"



    ## Object actions ============================================================
    def _find_interactable_object(self, query_name: str, require_receptacle: bool = False) -> Optional[Dict[str, Any]]: 
        """
        Find a currently interactable object using its canonical name, such as 'Mug 1'. Exact canonical-name matches are prioritized.
        """
        query = self._normalize_object_query(query_name)
        if self.current_location is None:
            return None

        all_objects = self.controller.last_event.metadata.get("objects", [])
        items = self._get_filtered_interactable_objects(location_name=self.current_location, all_objects=all_objects)
        candidates = []
        for item in items:
            obj = item["metadata"]
            if require_receptacle and not self._is_receptacle(obj):
                continue

            canonical_name = item["unique_name"]
            if self._normalize_object_query(canonical_name) == query:
                return obj

            object_type = self._normalize_object_query(obj.get("objectType", ""))
            if object_type == query:
                candidates.append(obj)

        return min(candidates, key=lambda obj: obj.get("distance", float("inf"))) if candidates else None


    def _find_held_object(self, query_name: str) -> Optional[Dict[str, Any]]:
        """ Find an object currently held by the agent. """
        self._sync_inventory(self.controller.last_event)
        query = self._normalize_object_query(query_name)
        exact_matches, type_matches = [], []

        for item in self.inventory:
            object_id = item.get("objectId")
            canonical_name = self.unique_id_map.get(object_id, item.get("objectType", ""))
            if self._normalize_object_query(canonical_name) == query:
                exact_matches.append(item)
            elif self._normalize_object_query(item.get("objectType", "")) == query:
                type_matches.append(item)

        return (exact_matches or type_matches or [None])[0]


    def _pickup_object(self, target_name: str, source_receptacle_name: str) -> Tuple[str, bool, bool]:
        """
        Execute: take {obj} from {recep}
        The specified object must currently be in or on the specified receptacle.
        """
        target_obj = self._find_interactable_object(query_name=target_name)
        if target_obj is None:
            return "Nothing happens.", False, False

        source_receptacle = self._find_interactable_object(query_name=source_receptacle_name, require_receptacle=True)
        if source_receptacle is None:
            return "Nothing happens.", False, False

        parent_receptacle_ids = set(target_obj.get("parentReceptacles") or [])
        source_receptacle_id = source_receptacle["objectId"]

        is_valid_source = False
        if not parent_receptacle_ids or source_receptacle_id in parent_receptacle_ids:
            is_valid_source = True
        else:
            all_objects = self.controller.last_event.metadata.get("objects", [])
            query_source = self._normalize_object_query(source_receptacle_name)
            for p_id in parent_receptacle_ids:
                p_obj = next((o for o in all_objects if o.get("objectId") == p_id), None)
                if p_obj:
                    p_type = self._normalize_object_query(p_obj.get("objectType", ""))
                    p_name = self._normalize_object_query(self.unique_id_map.get(p_id, ""))
                    if query_source in p_type or query_source in p_name or p_type in query_source:
                        is_valid_source = True
                        break

        if not is_valid_source:
            return "Nothing happens.", False, False

        event = self.controller.step(action="PickupObject", objectId=target_obj["objectId"])
        if not event.metadata.get("lastActionSuccess", False):
            event = self.controller.step(action="PickupObject", objectId=target_obj["objectId"], forceAction=True)
            if not event.metadata.get("lastActionSuccess", False):
                return "Nothing happens.", False, False

        self._sync_inventory(event)
        target_canonical_name = self.unique_id_map.get(target_obj["objectId"], target_name)
        source_canonical_name = self.unique_id_map.get(source_receptacle["objectId"], source_receptacle_name)
        return (f"You take the {target_canonical_name.lower()} from the {source_canonical_name.lower()}.", False, True)


    def _put_object(self, item_name, preposition, receptacle_name):
        receptacle = self._find_interactable_object(receptacle_name, require_receptacle=True)
        if receptacle is None: 
            return f"You don't see any '{receptacle_name}' nearby.", False, False
        
        recep_name = self.unique_id_map.get(receptacle["objectId"], receptacle_name).lower()
        expected = self._get_receptacle_preposition(receptacle)
        if preposition == "in/on": 
            preposition = expected
        elif preposition != expected: 
            return f"You must put objects {expected} the {recep_name}, not {preposition} it.", False, False
        
        held_obj = self._find_held_object(item_name)
        if held_obj is None:
            obj = self._find_interactable_object(item_name)
            if obj is None: 
                return f"You don't see any '{item_name}' nearby.", False, False
            
            if not obj.get("pickupable", False): 
                return "Nothing happens.", False, False
            
            event = self.controller.step(action="PickupObject", objectId=obj["objectId"], forceAction=True)
            if not event.metadata.get("lastActionSuccess", False):
                return f"Could not pick up {item_name}. {event.metadata.get('errorMessage', '')}", False, False
            
            self._sync_inventory(event)
            held_obj = self._find_held_object(item_name)
            receptacle = self._find_interactable_object(receptacle_name, require_receptacle=True)
            if held_obj is None or receptacle is None: 
                return "Nothing happens.", False, False
            
        obj_name = self.unique_id_map.get(held_obj["objectId"], item_name).lower()
        event = self.controller.step(action="PutObject", objectId=receptacle["objectId"], forceAction=True, placeStationary=True)
        if not event.metadata.get("lastActionSuccess", False):
            return "Nothing happens.", False, False

        self._sync_inventory(event)
        
        return f"You put the {obj_name} {preposition} the {recep_name}.", False, True
    
    
    def _toggle_container(self, target_name: str, open_action: bool) -> Tuple[str, bool, bool]:
        obj = self._find_interactable_object(target_name)
        if obj is None: 
            return "Nothing happens.", False, False
        
        event = self.controller.step(action="OpenObject" if open_action else "CloseObject", objectId=obj["objectId"])
        if not event.metadata.get("lastActionSuccess", False): 
            return "Nothing happens.", False, False
        
        name = self.unique_id_map.get(obj["objectId"], obj.get("objectType", target_name)).lower()
        observation = f"You {'open' if open_action else 'close'} the {name}."
        if open_action and self.current_location is not None:
            observation += f" {self._get_text_observation(self.current_location)}"
            
        return observation, False, True
    

    def _toggle_object(self, target_name: str) -> Tuple[str, bool, bool]:
        obj = self._find_interactable_object(target_name)
        if obj is None: 
            return "Nothing happens.", False, False
        
        name = self.unique_id_map.get(obj["objectId"], target_name).lower()
        if not obj.get("toggleable", False): 
            return "Nothing happens.", False, False
        
        currently_on = obj.get("isToggled", False)
        event = self.controller.step(action="ToggleObjectOff" if currently_on else "ToggleObjectOn", objectId=obj["objectId"], forceAction=True)
        if not event.metadata.get("lastActionSuccess", False): 
            return "Nothing happens.", False, False
        
        return f"You toggle the {name} {'off' if currently_on else 'on'}.", False, True


    ## Inventory and movement helpers ============================================================
    def _sync_inventory(self, event) -> None:
        self.inventory = list(event.metadata.get("inventoryObjects", []))


    def _get_held_object_ids(self) -> Set[str]:
        return {
            item.get("objectId")
            for item in self.inventory
            if item.get("objectId")
        }


    def _get_inventory_observation(self) -> str:
        self._sync_inventory(self.controller.last_event)
        if not self.inventory:
            return ""

        names = [self.unique_id_map.get(item.get("objectId"), item.get("objectType", "unknown object")) for item in self.inventory]
        return f"You are holding {self._format_plain_list([name.lower() for name in names])}."


    def _format_movement_response(self, event, success_message: str) -> Tuple[str, bool, bool]:
        if event.metadata.get("lastActionSuccess", False):
            return success_message, False, True
        return "Nothing happens.", False, False


    def _build_unique_id_map(self, objects: List[Dict[str, Any]]) -> Dict[str, str]:
        counts = defaultdict(int)
        unique_id_map = {}
        for obj in objects:
            object_id = obj.get("objectId")
            object_type = obj.get("objectType")
            if object_id is None or object_type in {None, "Floor"}:
                continue

            counts[object_type] += 1
            unique_id_map[object_id] = f"{object_type} {counts[object_type]}"

        return unique_id_map


    def _format_object_list(self, object_names: List[str]) -> str:
        names = [self._add_indefinite_article(name) for name in object_names]
        if len(names) <= 1:
            return "".join(names)
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"

        return f"{', '.join(names[:-1])}, and {names[-1]}"


    def _add_indefinite_article(self, object_name: str) -> str:
        name = object_name.strip().lower()
        if not name:
            return "an object"

        article = "an" if name[0] in "aeiou" else "a"
        return f"{article} {name}"


    def _format_plain_list(self, values: List[str]) -> str:
        if len(values) <= 1:
            return "".join(values)
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"

        return f"{', '.join(values[:-1])}, and {values[-1]}"


    def _normalize_command(self, command: str) -> str:
        return re.sub(r"\s+", " ", command.strip().lower())


    def _normalize_object_query(self, query: str) -> str:
        query = self._normalize_command(query)
        return query[4:] if query.startswith("the ") else query


    def _normalize_identifier(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())


    def render(self, figsize=(6, 6), title=None) -> None:
        plt.figure(figsize=figsize)
        plt.imshow(self.controller.last_event.frame)
        plt.axis("off")
        plt.title(title or self.current_location or self.scene)
        plt.show()
