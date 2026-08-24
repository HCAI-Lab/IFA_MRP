from typing import Any, Dict, List, Optional

from termcolor import colored

from .utils import save_json, save_video_animation


def export_run_artifacts(
    *,
    save_chat: bool,
    all_messages: List[Dict[str, Any]],
    messages_findnplace: Optional[List[Dict[str, Any]]],
    memories: List[Dict[str, Any]],
    time_board: Optional[Any],
    collected_frames: List[Any],
    activation_reports: List[Dict[str, Any]],
    display_retrieval_plots: bool,
    output_messages_path: Optional[str],
    output_findnplace_messages_path: Optional[str],
    output_memories_path: Optional[str],
    output_timeline_path: Optional[str],
    save_video: bool,
    output_video_path: Optional[str],
    output_actr_retrieval_plot_path: Optional[str],
    output_modified_retrieval_plot_path: Optional[str],
    output_combined_retrieval_plot_path: Optional[str],
) -> Dict[str, Any]:
    """Save run artifacts and return retrieval plot handles keyed by model/mode."""
    print("\n" + "=" * 60)
    print(colored("Exporting Execution Artifacts...", "yellow", attrs=["bold"]))

    if save_chat and output_messages_path:
        save_json(all_messages, output_messages_path)
    if save_chat and output_findnplace_messages_path:
        save_json(messages_findnplace or [], output_findnplace_messages_path)
    if output_memories_path:
        save_json(memories, output_memories_path)
    if output_timeline_path and time_board is not None:
        save_json(time_board.timeline, output_timeline_path)

    retrieval_plots = {}
    if activation_reports and display_retrieval_plots:
        from utils.retrieval_time_plots import (
            plot_actr_retrieval_times,
            plot_combined_retrieval_times,
            plot_modified_retrieval_times,
        )

        retrieval_plot_specs = (
            ("actr", output_actr_retrieval_plot_path, plot_actr_retrieval_times, "ACT-R"),
            ("modified", output_modified_retrieval_plot_path, plot_modified_retrieval_times, "modified"),
            ("combined", output_combined_retrieval_plot_path, plot_combined_retrieval_times, "combined"),
        )
        for plot_key, output_path, plot_func, label in retrieval_plot_specs:
            if not output_path:
                continue
            retrieval_plots[plot_key] = plot_func(
                activation_reports,
                output_path=output_path,
            )
            print(colored(f"Saved {label} retrieval time plot: {output_path}", "green", attrs=["bold"]))

    if save_video and collected_frames and output_video_path:
        save_video_animation(collected_frames, output_video_path, interval=200)

    print(colored("✓ All artifacts successfully saved!", "green", attrs=["bold"]))
    return retrieval_plots
