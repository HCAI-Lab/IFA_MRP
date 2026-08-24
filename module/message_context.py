import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from prompt.builders import findnplace_icl_messages, icl_messages
from prompt.instructions import ICL_INSTRUCTION_BDI


FINDNPLACE_CONTEXT_NOTICE = """You have already successfully finished the exploration and rearrangement tasks. Now, you will begin a find and place task. All the previous interaction transcripts are hidden.
To complete the find and place task successfully, you should complete given subtasks based on current observations and subtask-relevant memories from previous tasks and their corresponding subtasks.
"""


@dataclass
class ObservationContextResult:
    mem_prompt: str = ""
    started_findnplace: bool = False


def get_subtask_key(env: Any) -> Optional[tuple[Any, Any]]:
    if hasattr(env, "current_task_name"):
        return (env.current_task_name, env.current_subtask_id)
    return None


def extract_subtask_description(observation: str, env: Any) -> Optional[str]:
    """Extract the current subtask text from an observation or environment state."""
    for pattern in (
        r"Your next subtask is to:\s*(.*?)(?=\.|\n|$)",
        r"Your first subtask is to:\s*(.*?)(?=\.|\n|$)",
        r"Your task is to:\s*(.*?)(?=\.|\n|$)",
    ):
        match = re.search(pattern, observation)
        if match:
            return match.group(1).strip()

    if hasattr(env, "current_subtask") and env.current_subtask and "subtask" in env.current_subtask:
        return env.current_subtask["subtask"]

    return None


def format_retrieved_memories(
    mem_module: Any,
    subtask_desc: str,
    top_k: int,
    current_timestep: int,
    max_timestep: Optional[int],
) -> str:
    try:
        return mem_module.format_retrieved_memories_prompt(
            subtask_desc,
            top_k=top_k,
            current_timestep=current_timestep,
            max_timestep=max_timestep,
        )
    except TypeError:
        return mem_module.format_retrieved_memories_prompt(
            subtask_desc,
            top_k=top_k,
            current_timestep=current_timestep,
        )


class MessageContextManager:
    """Owns main/findNplace prompts plus the full exported transcript."""

    def __init__(self, env: Any, mem_module: Any, top_k: int):
        self.env = env
        self.mem_module = mem_module
        self.top_k = top_k
        self.messages = icl_messages(
            instruction=ICL_INSTRUCTION_BDI,
            icl_example_path="prompt/hfes_icl_example_bdi.json",
        )
        self.all_messages = list(self.messages)
        self.messages_findnplace: Optional[List[Dict[str, Any]]] = None
        self.last_subtask_key: Optional[tuple[Any, Any]] = None
        self.findnplace_start_timestep: Optional[int] = None

    def append_prompt_message(
        self,
        prompt_messages: List[Dict[str, Any]],
        role: str,
        content: str,
        timing_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt_messages.append({"role": role, "content": content})
        message = {"role": role, "content": content}
        if timing_metadata:
            message.update(timing_metadata)
        self.all_messages.append(message)
        return message

    def append_transcript_message(self, role: str, content: str) -> None:
        self.all_messages.append({"role": role, "content": content})

    def append_initial_observation(self, observation: str) -> None:
        self.append_prompt_message(
            self.messages,
            "user",
            f"Task Description: {observation}",
        )

    def active_messages(self, current_task_name: Optional[str]) -> List[Dict[str, Any]]:
        if current_task_name == "findNplace" and self.messages_findnplace is not None:
            return self.messages_findnplace
        return self.messages

    def is_new_subtask(self, subtask_key: Any) -> bool:
        return subtask_key != self.last_subtask_key

    def start_initial_findnplace_context(self, observation: str) -> str:
        if self.env.current_task_name != "findNplace":
            return ""

        self.findnplace_start_timestep = 0
        return self.start_findnplace_context(
            observation,
            0,
            get_subtask_key(self.env),
        )

    def note_findnplace_started(self, step: int) -> bool:
        if self.findnplace_start_timestep is not None:
            return False
        self.findnplace_start_timestep = step
        return True

    def append_observation(
        self,
        observation: str,
        step: int,
        subtask_key: Any,
        is_findnplace: bool,
        timing_metadata: Optional[Dict[str, Any]] = None,
    ) -> ObservationContextResult:
        user_msg_content = f"Observation at step {step}: {observation}"
        started_findnplace = False
        mem_prompt = ""

        if is_findnplace:
            started_findnplace = self.note_findnplace_started(step)

            if self.is_new_subtask(subtask_key):
                if self.messages_findnplace is None:
                    mem_prompt = self.start_findnplace_context(
                        observation,
                        step,
                        subtask_key,
                        timing_metadata,
                    )
                else:
                    mem_prompt = self.append_findnplace_subtask_context(
                        observation,
                        step,
                        subtask_key,
                        timing_metadata,
                    )
            elif self.messages_findnplace is not None:
                self.append_prompt_message(
                    self.messages_findnplace,
                    "user",
                    user_msg_content,
                    timing_metadata,
                )
            return ObservationContextResult(
                mem_prompt=mem_prompt,
                started_findnplace=started_findnplace,
            )

        if self.is_new_subtask(subtask_key):
            self.last_subtask_key = subtask_key
        self.append_prompt_message(self.messages, "user", user_msg_content, timing_metadata)
        return ObservationContextResult()

    def append_findnplace_memory_prompt(self, observation: str, step: int, subtask_key: Any) -> str:
        self.last_subtask_key = subtask_key
        subtask_desc = extract_subtask_description(observation, self.env)
        if not subtask_desc:
            return ""

        max_memory_timestep = (
            self.findnplace_start_timestep - 1
            if self.findnplace_start_timestep is not None
            else None
        )
        mem_prompt = format_retrieved_memories(
            mem_module=self.mem_module,
            subtask_desc=subtask_desc,
            top_k=self.top_k,
            current_timestep=step,
            max_timestep=max_memory_timestep,
        )
        if mem_prompt and self.messages_findnplace is not None:
            self.append_prompt_message(self.messages_findnplace, "user", mem_prompt)
        return mem_prompt

    def start_findnplace_context(
        self,
        observation: str,
        step: int,
        subtask_key: Any,
        timing_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        subtask_desc = extract_subtask_description(observation, self.env)
        self.messages_findnplace = findnplace_icl_messages(
            instruction=ICL_INSTRUCTION_BDI,
            icl_example_path="prompt/hfes_icl_example_find_bdi.json",
        )
        self.append_transcript_message(
            "system",
            "=== Message Context Separated for the 'find and place' task. ===",
        )

        self.append_prompt_message(self.messages_findnplace, "user", FINDNPLACE_CONTEXT_NOTICE)
        self.append_prompt_message(self.messages_findnplace, "assistant", "OK.")

        if subtask_desc:
            self.append_prompt_message(
                self.messages_findnplace,
                "user",
                f"The find and place task begins. You have entered a room. You can go to fridge, sink, table 1, microwave, stove, table 2, and door. Your first subtask is to: {subtask_desc}",
            )

        self.append_prompt_message(
            self.messages_findnplace,
            "user",
            f"Observation at step {step}: {observation}",
            timing_metadata,
        )
        self.last_subtask_key = subtask_key

        return self.append_findnplace_memory_prompt(observation, step, subtask_key)

    def append_findnplace_subtask_context(
        self,
        observation: str,
        step: int,
        subtask_key: Any,
        timing_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.messages_findnplace is None:
            return self.start_findnplace_context(observation, step, subtask_key, timing_metadata)

        self.append_prompt_message(
            self.messages_findnplace,
            "user",
            f"Observation at step {step}: {observation}",
            timing_metadata,
        )
        return self.append_findnplace_memory_prompt(observation, step, subtask_key)
