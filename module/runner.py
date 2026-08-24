import time
from typing import Any, Dict, Optional
from termcolor import colored
from .action_retry import generate_valid_action
from .activation_tracker import ActivationTracker
from .artifacts import export_run_artifacts
from .message_context import MessageContextManager, get_subtask_key
from .runner_time import get_action_observation_time, get_current_virtual_time
from .time_board import AgentTimeBoard
from .utils import add_frame_overlay


def _load_default_stopper() -> Optional[Any]:
    try:
        from .subtask_stopper import RearrangementStepStopper
    except ImportError:
        return None
    return RearrangementStepStopper()


def run_agent_loop(
    env: Any,
    dm_agent: Any,
    memory_encoding: Any = None,
    memory_store: Any = None,
    max_timesteps: int = 120,
    top_k: int = 10,
    step_delay: float = 0.5,
    save_video: bool = True,
    save_chat: bool = True,
    output_video_path: str = "execution_video.gif",
    output_messages_path: str = "stacked_messages.json",
    output_findnplace_messages_path: str = "chat_findnplace.json",
    output_memories_path: Optional[str] = None,
    use_subtask_stopper: bool = True,
    subtask_stopper: Optional[Any] = None,
    format_retry_count: int = 10,
    format_retry_delay: float = 10.0,
    emulate_timing: bool = True,
    timing_data_path: str = "data/user_data.json",
    output_timeline_path: Optional[str] = None,
    debug: bool = False,
    debug_memories: bool = False,
    use_dwell_time: bool = False,
    activation_decay: float = 0.5,
    retrieval_F: float = 1.0,
    retrieval_f: float = 1.0,
    output_actr_retrieval_plot_path: Optional[str] = "actr_retrieval_times.png",
    output_modified_retrieval_plot_path: Optional[str] = "modified_retrieval_times.png",
    output_combined_retrieval_plot_path: Optional[str] = "combined_retrieval_times.png",
    display_retrieval_plots: bool = True,
    print_summary: bool = False,
) -> Dict[str, Any]:
    """
    Executes the dual-agent interaction loop (Decision Agent + Memory Encoding Agent),
    manages prompt history, and exports stacked messages and video animations upon completion.
    Includes configurable step_delay between LLM API calls and adjustable top_k for subtask memory retrieval.
    Separates message context for findNplace task to rely on current observations & retrieved encoded memories.
    When debug is enabled, detailed timeline, affordance, entropy, and activation output are printed.
    """
    mem_module = memory_encoding or memory_store
    if mem_module is None:
        raise ValueError("Must provide either memory_encoding or memory_store to run_agent_loop.")
    debug_memories = debug or debug_memories
    active_stopper = subtask_stopper
    if active_stopper is None and use_subtask_stopper:
        active_stopper = _load_default_stopper()
    time_board = AgentTimeBoard(timing_data_path) if emulate_timing else None
    activation_tracker = ActivationTracker(
        decay=activation_decay,
        retrieval_F=retrieval_F,
        retrieval_f=retrieval_f,
    )

    initial_obs = env.reset()
    context = MessageContextManager(env=env, mem_module=mem_module, top_k=top_k)

    # Reset and encode the initial observation in memory.
    mem_module.memories = []
    mem_module.encode_and_add_memory(timestep=0, action="initial_state", observation=initial_obs)

    context.append_initial_observation(initial_obs)
    print(colored("Task Description:", "black", attrs=["bold"]), initial_obs)

    completed_tasks_seen = set()
    collected_frames = []
    timestep = 1
    stopper_triggered = False
    stopper_reason = ""
    context.start_initial_findnplace_context(initial_obs)

    while timestep <= max_timesteps and not env.is_all_tasks_complete:
        if step_delay > 0:
            time.sleep(step_delay)

        # Generate and parse the decision-agent action.
        task_before_step = getattr(env, "current_task_name", None)
        active_messages = context.active_messages(task_before_step)
        llm_output, parsed_action = generate_valid_action(
            dm_agent=dm_agent,
            active_messages=active_messages,
            step=timestep,
            format_retry_count=format_retry_count,
            format_retry_delay=format_retry_delay,
        )
        assistant_log_msg = context.append_prompt_message(active_messages, "assistant", llm_output)

        # Execute the action in the environment.
        try:
            obs, subtask_done, action_success, frame = env.step(parsed_action)
        except Exception as err:
            print(colored(f"⚠️ [AI2-THOR Step Timeout/Error]: {err}. Continuing...", "yellow"))
            obs = "Nothing happens."
            subtask_done = False
            action_success = False
            frame = getattr(env, "frame", None)
        task_after_step = getattr(env, "current_task_name", None)

        timing_metadata = None
        if time_board is not None:
            timing_metadata = time_board.record_step(
                timestep=timestep,
                task_before=task_before_step,
                action_text=parsed_action,
                action_success=action_success,
                location_after=getattr(env, "current_location", None),
                task_after=task_after_step,
                observation=obs,
                use_dwell_time=use_dwell_time,
            )
            assistant_log_msg.update(timing_metadata)

        # Collect frame with step/action overlay.
        if save_video and frame is not None:
            overlay_frame = add_frame_overlay(
                frame,
                step_num=timestep,
                action=parsed_action,
                task_name=task_after_step,
            )
            collected_frames.append(overlay_frame)

        # Encode the step observation in memory.
        mem_module.encode_and_add_memory(timestep=timestep, action=parsed_action, observation=obs)

        current_sub_key = get_subtask_key(env)
        is_new_subtask = context.is_new_subtask(current_sub_key)
        observation_time = get_action_observation_time(timing_metadata, time_board, timestep)
        activation_report = ""
        is_findnplace = task_after_step == "findNplace"
        if is_findnplace and is_new_subtask:
            target_info = activation_tracker.extract_target_info(env)
            current_virtual_time = get_current_virtual_time(timing_metadata, time_board, timestep)
            activation_report = activation_tracker.format_activation_report(
                target_info=target_info,
                current_time=current_virtual_time,
                task_name=task_after_step,
                subtask_id=getattr(env, "current_subtask_id", None),
                include_debug=debug,
            )
            activation_tracker.start_search(
                target_info=target_info,
                current_time=current_virtual_time,
                task_name=task_after_step,
                subtask_id=getattr(env, "current_subtask_id", None),
            )

        context_result = context.append_observation(
            observation=obs,
            step=timestep,
            subtask_key=current_sub_key,
            is_findnplace=is_findnplace,
            timing_metadata=timing_metadata,
        )
        if context_result.started_findnplace:
            print(colored("🔄 Separating message context for 'findNplace' task...", "cyan", attrs=["bold"]))

        print(colored(f"Observation {timestep} ({observation_time:.2f}s -):", "blue"), obs)
        if activation_report:
            print(colored(activation_report, "red", attrs=["bold"]))
        if debug and hasattr(env, "format_observation_affordances"):
            print(colored(
                env.format_observation_affordances(obs),
                "green",
                attrs=["bold"],
            ))
        if debug and hasattr(env, "format_observation_entropy"):
            print(colored(
                env.format_observation_entropy(obs),
                "cyan",
                attrs=["bold"],
            ))
        if debug and time_board is not None and timing_metadata is not None:
            print(colored(
                time_board.format_step_calculation(timing_metadata),
                "magenta",
                attrs=["bold"],
            ))
        activation_tracker.update_from_observation(
            env=env,
            observation=obs,
            timestamp=observation_time,
            timestep=timestep,
            task_name=task_after_step,
            subtask_id=getattr(env, "current_subtask_id", None),
            location=getattr(env, "current_location", None),
        )
        activation_tracker.update_search_from_observation(
            env=env,
            observation=obs,
            observation_time=observation_time,
            timestep=timestep,
            location=getattr(env, "current_location", None),
        )
        if debug_memories and context_result.mem_prompt:
            print(colored(context_result.mem_prompt, "magenta"))
        print("-" * 50)

        if active_stopper is not None:
            stopper_decision = active_stopper.check(
                timestep=timestep,
                observation=obs,
                subtask_done=subtask_done,
                env=env,
            )
            if stopper_decision.should_stop:
                stopper_triggered = True
                stopper_reason = stopper_decision.reason
                stop_msg = f"=== Subtask Stopper Triggered at step {timestep}: {stopper_reason} ==="
                print(colored(stop_msg, "red", attrs=["bold"]))
                context.append_transcript_message("system", stop_msg)
                break

        # Check completion status and print summaries when requested.
        if print_summary and subtask_done:
            print("=" * 60)
            print(colored(f"✓ Subtask completed! Updated Completion Summary:", "green", attrs=["bold"]))
            print(env.get_completion_summary())
            print("=" * 60)

        if print_summary:
            overall_info = env.get_overall_completion()
            for t_name, t_status in overall_info["task_breakdown"].items():
                if t_status["is_complete"] and t_name not in completed_tasks_seen:
                    completed_tasks_seen.add(t_name)
                    print(colored(f"🎉 Task '{t_name}' completed! ({t_status['status']})", "green", attrs=["bold"]))

        if env.is_all_tasks_complete:
            break

        timestep += 1

    # === Final Completion Summary ===
    if print_summary:
        print("\n" + "=" * 60)
        if env.is_all_tasks_complete:
            print(colored("🎉 ALL TASKS COMPLETED SUCCESSFULLY!", "green", attrs=["bold"]))
        else:
            print(colored(f"⏱️ Loop finished at Timestep {timestep - 1}/{max_timesteps}.", "yellow", attrs=["bold"]))

        print("\n" + colored("Final Completion Summary:", "black", attrs=["bold"]))
        print(env.get_completion_summary())

    retrieval_plots = export_run_artifacts(
        save_chat=save_chat,
        all_messages=context.all_messages,
        messages_findnplace=context.messages_findnplace,
        memories=mem_module.memories,
        time_board=time_board,
        collected_frames=collected_frames,
        activation_reports=activation_tracker.reports,
        display_retrieval_plots=display_retrieval_plots,
        output_messages_path=output_messages_path,
        output_findnplace_messages_path=output_findnplace_messages_path,
        output_memories_path=output_memories_path,
        output_timeline_path=output_timeline_path,
        save_video=save_video,
        output_video_path=output_video_path,
        output_actr_retrieval_plot_path=output_actr_retrieval_plot_path,
        output_modified_retrieval_plot_path=output_modified_retrieval_plot_path,
        output_combined_retrieval_plot_path=output_combined_retrieval_plot_path,
    )

    retrieval_time_features = activation_tracker.get_retrieval_time_features()

    return {
        "messages": context.all_messages,
        "messages_findnplace": context.messages_findnplace or [],
        "memories": mem_module.memories,
        "frames": collected_frames,
        "timeline": time_board.timeline if time_board is not None else [],
        "current_time": round(time_board.current_time, 2) if time_board is not None else None,
        "activation_reports": activation_tracker.reports,
        "retrieval_plots": retrieval_plots,
        "actr_retrieval_plot": retrieval_plots.get("actr"),
        "modified_retrieval_plot": retrieval_plots.get("modified"),
        "combined_retrieval_plot": retrieval_plots.get("combined"),
        "retrieval_time_features": retrieval_time_features,
        "actr_retrieval_time_features": retrieval_time_features["actr_only"],
        "modified_retrieval_time_features": retrieval_time_features["modified_only"],
        "combined_retrieval_time_features": retrieval_time_features["combined"],
        "activation_episodes": activation_tracker.episodes,
        "search_records": activation_tracker.search_records,
        "completed": env.is_all_tasks_complete,
        "stopper_triggered": stopper_triggered,
        "stopper_reason": stopper_reason,
    }
