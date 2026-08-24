import math
from typing import Any, Iterable, List, Mapping

import matplotlib.pyplot as plt


def _finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _add_value_labels(ax: Any, x_positions: List[int], values: List[float], y_offset: int = 8) -> None:
    for x_position, value in zip(x_positions, values):
        if not math.isfinite(value):
            continue
        ax.annotate(
            f"{value:.2f}",
            xy=(x_position, value),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _make_labels(activation_reports: Iterable[Mapping[str, Any]]) -> List[str]:
    labels = []
    for report in activation_reports:
        subtask_id = report.get("subtask_id")
        target = report.get("target", "unknown")
        labels.append(f"{subtask_id}: {target}" if subtask_id is not None else target)
    return labels


def plot_actr_retrieval_times(
    activation_reports: Iterable[Mapping[str, Any]],
    output_path: str = "actr_retrieval_times.png",
) -> Any:
    activation_reports = list(activation_reports)
    labels = _make_labels(activation_reports)
    actr_times = [
        _finite_or_nan(report.get("actr_retrieval_time", math.nan))
        for report in activation_reports
    ]
    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.plot(x_positions, actr_times, marker="o", linewidth=2, label="ACT-R retrieval time")
    _add_value_labels(ax, x_positions, actr_times)

    ax.set_title("ACT-R Retrieval Time by Find&Place Target")
    ax.set_ylabel("Retrieval time")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.18)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.show()
    return fig


def plot_modified_retrieval_times(
    activation_reports: Iterable[Mapping[str, Any]],
    output_path: str = "modified_retrieval_times.png",
) -> Any:
    activation_reports = list(activation_reports)
    labels = _make_labels(activation_reports)
    modified_times = [
        _finite_or_nan(report.get("modified_retrieval_time", math.nan))
        for report in activation_reports
    ]
    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.plot(x_positions, modified_times, marker="o", linewidth=2, label="Modified retrieval time")
    _add_value_labels(ax, x_positions, modified_times)

    ax.set_title("Modified Retrieval Time by Find&Place Target")
    ax.set_ylabel("Retrieval time")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.18)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.show()
    return fig


def plot_combined_retrieval_times(
    activation_reports: Iterable[Mapping[str, Any]],
    output_path: str = "combined_retrieval_times.png",
) -> Any:
    activation_reports = list(activation_reports)
    labels = _make_labels(activation_reports)
    actr_times = [
        _finite_or_nan(report.get("actr_retrieval_time", math.nan))
        for report in activation_reports
    ]
    modified_times = [
        _finite_or_nan(report.get("modified_retrieval_time", math.nan))
        for report in activation_reports
    ]
    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.plot(x_positions, actr_times, marker="o", linewidth=2, label="ACT-R retrieval time")
    ax.plot(x_positions, modified_times, marker="o", linewidth=2, label="Modified retrieval time")

    _add_value_labels(ax, x_positions, actr_times, y_offset=8)
    _add_value_labels(ax, x_positions, modified_times, y_offset=20)

    ax.set_title("Retrieval Time Comparison by Find&Place Target")
    ax.set_ylabel("Retrieval time")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.18)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.show()
    return fig
