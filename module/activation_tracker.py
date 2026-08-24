import math
import os
import re
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt


class ActivationTracker:
    """Track object visibility over virtual time and calculate activation values."""

    def __init__(self, decay: float = 0.5, retrieval_F: float = 1.0, retrieval_f: float = 1.0):
        self.decay = decay
        self.retrieval_F = retrieval_F
        self.retrieval_f = retrieval_f
        self.active: Dict[str, Dict[str, Any]] = {}
        self.episodes: List[Dict[str, Any]] = []
        self.reports: List[Dict[str, Any]] = []
        self.search_records: List[Dict[str, Any]] = []
        self.current_search: Optional[Dict[str, Any]] = None

    @staticmethod
    def calc_retrieval_time(activation: float, F: float = 1.0, f: float = 1.0) -> float:
        if math.isnan(activation):
            return math.nan
        if activation == float("-inf"):
            return math.inf
        if activation == float("inf"):
            return 0.0
        try:
            return F * math.exp(-(f * activation))
        except OverflowError:
            return math.inf

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = value.translate(str.maketrans("‘’“”", "''\"\"")).lower()
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _name_variants(cls, value: str) -> Set[str]:
        normalized = cls._normalize_text(value)
        without_article = re.sub(r"^(?:a|an|the)\s+", "", normalized)
        without_number = re.sub(r"\s+\d+$", "", without_article)
        variants = {normalized, without_article, without_number}
        variants.update(re.sub(r"[\s_-]+", "", item) for item in list(variants))
        return {item for item in variants if item}

    @classmethod
    def _target_aliases(cls, target_name: str) -> Set[str]:
        aliases = cls._name_variants(target_name)
        compact_aliases = {re.sub(r"[\s_-]+", "", alias) for alias in aliases}
        aliases.update(compact_aliases)
        return aliases

    @classmethod
    def _matches_target(cls, object_name: str, target_aliases: Set[str]) -> bool:
        return bool(cls._name_variants(object_name).intersection(target_aliases))

    @classmethod
    def extract_target_info(cls, env: Any) -> Dict[str, Any]:
        subtask = getattr(env, "current_subtask", None) or {}
        subtask_text = subtask.get("subtask", "")
        trajectories = subtask.get("trajectories", []) or []

        display_target = None
        quoted_match = re.search(r"['\"]([^'\"]+)['\"]", cls._normalize_text(subtask_text))
        if quoted_match:
            display_target = quoted_match.group(1).strip()

        canonical_target = None
        for action in trajectories:
            take_match = re.search(r"\btake\s+(.+?)\s+from\b", cls._normalize_text(action))
            if take_match:
                canonical_target = take_match.group(1).strip()
                break

        target_name = canonical_target or display_target
        if not target_name:
            find_match = re.search(r"\bfind\s+(.+?)\s+and\b", cls._normalize_text(subtask_text))
            if find_match:
                target_name = find_match.group(1).strip()

        return {
            "target_name": target_name,
            "display_target": display_target or target_name,
            "canonical_target": canonical_target,
            "subtask": subtask_text,
            "aliases": cls._target_aliases(target_name) if target_name else set(),
        }

    def _extract_observed_object_names(self, env: Any, observation: str) -> List[str]:
        if hasattr(env, "get_current_position_object_names"):
            return [
                self._normalize_text(name)
                for name in env.get_current_position_object_names()
            ]

        if not observation:
            return []

        normalized_observation = self._normalize_text(observation)
        last_event = getattr(getattr(env, "controller", None), "last_event", None)
        all_objects = last_event.metadata.get("objects", []) if last_event else []
        observed = []
        seen = set()

        for obj in all_objects:
            object_id = obj.get("objectId")
            unique_name = getattr(env, "unique_id_map", {}).get(object_id)
            if not object_id or not unique_name:
                continue

            name = self._normalize_text(unique_name)
            if name in seen:
                continue

            if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", normalized_observation):
                observed.append(name)
                seen.add(name)

        return observed

    def _entropy_info(self, env: Any, observation: str) -> Dict[str, Any]:
        if hasattr(env, "calculate_observation_entropy"):
            return env.calculate_observation_entropy(observation)
        return {
            "total_obj": 0,
            "h_j_openable_pickupable": 0.0,
            "num_openable_obj": 0,
            "num_pickupable_obj": 0,
            "num_both_obj": 0,
            "total_unique_obj": 0,
        }

    def _start_episode(
        self,
        object_name: str,
        timestamp: float,
        entropy_info: Dict[str, Any],
        task_name: Optional[str],
        subtask_id: Optional[int],
        timestep: int,
        location: Optional[str],
    ) -> None:
        self.active[object_name] = {
            "object_name": object_name,
            "visible_begin": timestamp,
            "last_visible_time": timestamp,
            "task_name": task_name,
            "subtask_id": subtask_id,
            "first_timestep": timestep,
            "last_timestep": timestep,
            "location": location,
            "current_scene_entropy": entropy_info["h_j_openable_pickupable"],
            "current_total_obj": entropy_info["total_obj"],
            "weighted_scene_entropy": 0.0,
            "weighted_total_obj": 0.0,
        }

    def _extend_episode(
        self,
        episode: Dict[str, Any],
        timestamp: float,
        entropy_info: Dict[str, Any],
        timestep: int,
        location: Optional[str],
    ) -> None:
        delta = max(0.0, timestamp - episode["last_visible_time"])
        if delta > 0:
            episode["weighted_scene_entropy"] += episode["current_scene_entropy"] * delta
            episode["weighted_total_obj"] += episode["current_total_obj"] * delta

        episode["last_visible_time"] = timestamp
        episode["last_timestep"] = timestep
        episode["location"] = location or episode["location"]
        episode["current_scene_entropy"] = entropy_info["h_j_openable_pickupable"]
        episode["current_total_obj"] = entropy_info["total_obj"]

    def _carry_episode_to(self, episode: Dict[str, Any], timestamp: float) -> None:
        delta = max(0.0, timestamp - episode["last_visible_time"])
        if delta > 0:
            episode["weighted_scene_entropy"] += episode["current_scene_entropy"] * delta
            episode["weighted_total_obj"] += episode["current_total_obj"] * delta
            episode["last_visible_time"] = timestamp

    def _finalize_episode(self, object_name: str, timestamp: Optional[float] = None) -> None:
        episode = self.active.pop(object_name, None)
        if episode is None:
            return

        if timestamp is not None:
            self._carry_episode_to(episode, timestamp)

        duration = max(0.0, episode["last_visible_time"] - episode["visible_begin"])
        if duration > 0:
            scene_entropy = episode["weighted_scene_entropy"] / duration
            total_obj = episode["weighted_total_obj"] / duration
        else:
            scene_entropy = episode["current_scene_entropy"]
            total_obj = episode["current_total_obj"]

        self.episodes.append({
            "object_name": episode["object_name"],
            "visible_begin": round(episode["visible_begin"], 2),
            "visible_end": round(episode["last_visible_time"], 2),
            "observation_duration": round(duration, 2),
            "scene_entropy": scene_entropy,
            "total_obj": total_obj,
            "task_name": episode["task_name"],
            "subtask_id": episode["subtask_id"],
            "first_timestep": episode["first_timestep"],
            "last_timestep": episode["last_timestep"],
            "location": episode["location"],
        })

    def update_from_observation(
        self,
        env: Any,
        observation: str,
        timestamp: float,
        timestep: int,
        task_name: Optional[str],
        subtask_id: Optional[int],
        location: Optional[str],
    ) -> None:
        observed_names = set(self._extract_observed_object_names(env, observation))
        entropy_info = self._entropy_info(env, observation)

        for object_name in list(self.active):
            if object_name not in observed_names:
                self._finalize_episode(object_name, timestamp=timestamp)

        for object_name in observed_names:
            if object_name in self.active:
                self._extend_episode(
                    self.active[object_name],
                    timestamp=timestamp,
                    entropy_info=entropy_info,
                    timestep=timestep,
                    location=location,
                )
            else:
                self._start_episode(
                    object_name=object_name,
                    timestamp=timestamp,
                    entropy_info=entropy_info,
                    task_name=task_name,
                    subtask_id=subtask_id,
                    timestep=timestep,
                    location=location,
                )

    def _candidate_episodes(self, target_aliases: Set[str]) -> List[Dict[str, Any]]:
        candidates = [
            episode
            for episode in self.episodes
            if self._matches_target(episode["object_name"], target_aliases)
        ]
        for episode in self.active.values():
            if not self._matches_target(episode["object_name"], target_aliases):
                continue
            duration = max(0.0, episode["last_visible_time"] - episode["visible_begin"])
            if duration > 0:
                scene_entropy = episode["weighted_scene_entropy"] / duration
                total_obj = episode["weighted_total_obj"] / duration
            else:
                scene_entropy = episode["current_scene_entropy"]
                total_obj = episode["current_total_obj"]
            candidates.append({
                "object_name": episode["object_name"],
                "visible_begin": round(episode["visible_begin"], 2),
                "visible_end": round(episode["last_visible_time"], 2),
                "observation_duration": round(duration, 2),
                "scene_entropy": scene_entropy,
                "total_obj": total_obj,
                "task_name": episode["task_name"],
                "subtask_id": episode["subtask_id"],
                "first_timestep": episode["first_timestep"],
                "last_timestep": episode["last_timestep"],
                "location": episode["location"],
            })
        return sorted(candidates, key=lambda item: item["visible_end"])

    def calculate_activation(
        self,
        target_info: Dict[str, Any],
        current_time: float,
    ) -> Dict[str, Any]:
        target_aliases = target_info.get("aliases") or set()
        episodes = self._candidate_episodes(target_aliases)
        evidence = []
        base_terms = []
        modified_terms = []

        for episode in episodes:
            elapsed_time = max(1e-9, current_time - episode["visible_end"])
            base_term = math.pow(elapsed_time, -self.decay)
            base_terms.append(base_term)

            observation_duration = episode["observation_duration"]
            total_obj = episode["total_obj"]
            scene_entropy = episode["scene_entropy"]
            modified_activation_partial = 0.0
            s_j = 0.0
            modified_decay = 0.0

            if scene_entropy > 0:
                modified_decay = self.decay * math.sqrt(scene_entropy)
            if observation_duration > 0 and total_obj > 0 and modified_decay > 0:
                s_j = observation_duration / total_obj
                modified_base = (s_j * s_j) * elapsed_time
                if modified_base > 0:
                    modified_activation_partial = math.pow(modified_base, -modified_decay)
                    if modified_activation_partial > 0:
                        modified_terms.append(modified_activation_partial)

            evidence.append({
                **episode,
                "elapsed_time": elapsed_time,
                "base_term": base_term,
                "s_j": s_j,
                "modified_decay": modified_decay,
                "modified_activation_partial": modified_activation_partial,
                "modified_activation": modified_activation_partial,
            })

        base_sum = sum(base_terms)
        base_level_activation = math.log(base_sum) if base_sum > 0 else float("-inf")
        modified_sum = sum(modified_terms)
        modified_activation_total = math.log(modified_sum) if modified_sum > 0 else float("-inf")
        actr_retrieval_time = self.calc_retrieval_time(
            base_level_activation,
            F=self.retrieval_F,
            f=self.retrieval_f,
        )
        modified_retrieval_time = self.calc_retrieval_time(
            modified_activation_total,
            F=self.retrieval_F,
            f=self.retrieval_f,
        )
        return {
            "target": target_info.get("display_target") or target_info.get("target_name"),
            "canonical_target": target_info.get("canonical_target"),
            "current_time": current_time,
            "decay": self.decay,
            "retrieval_F": self.retrieval_F,
            "retrieval_f": self.retrieval_f,
            "base_level_activation": base_level_activation,
            "modified_activation_total": modified_activation_total,
            "modified_activation_sum": modified_sum,
            "actr_retrieval_time": actr_retrieval_time,
            "modified_retrieval_time": modified_retrieval_time,
            "evidence": evidence,
        }

    @staticmethod
    def _fmt(value: float) -> str:
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return f"{value:.3f}"

    def format_activation_report(
        self,
        target_info: Dict[str, Any],
        current_time: float,
        task_name: Optional[str],
        subtask_id: Optional[int],
        include_debug: bool = False,
    ) -> str:
        activation = self.calculate_activation(target_info, current_time)
        self.reports.append({
            "task_name": task_name,
            "subtask_id": subtask_id,
            **activation,
        })

        target = activation["target"] or "unknown target"
        canonical = activation["canonical_target"]
        canonical_note = f" (canonical: {canonical})" if canonical and canonical != target else ""
        retrieval_time_line = (
            f"Predicted Memory Retrieval Time: {self._fmt(activation['modified_retrieval_time'])}"
        )
        if not include_debug:
            return retrieval_time_line

        lines = [
            f"Activation for {task_name} subtask {subtask_id} target '{target}'{canonical_note}",
            f"ACT-R base-level activation: {self._fmt(activation['base_level_activation'])}",
            (
                f"Modified activation total: {self._fmt(activation['modified_activation_total'])} "
                f"(log(sum modified partials); sum={self._fmt(activation['modified_activation_sum'])})"
            ),
            retrieval_time_line,
        ]

        if not activation["evidence"]:
            lines.append("Evidence: no prior visibility episodes for this target")
            return "\n".join(lines)

        lines.append("Evidence:")
        for episode in activation["evidence"]:
            location = episode.get("location") or "unknown location"
            lines.append(
                f"- {episode['object_name']} at {location}: "
                f"visible {episode['visible_begin']:.2f}s -> {episode['visible_end']:.2f}s, "
                f"duration={episode['observation_duration']:.2f}s, "
                f"elapsed={episode['elapsed_time']:.2f}s, "
                f"entropy={episode['scene_entropy']:.3f}, "
                f"total_obj={episode['total_obj']:.3f}, "
                f"S_j={episode['s_j']:.3f}, "
                f"modified_decay={episode['modified_decay']:.3f}, "
                f"modified_partial={episode['modified_activation_partial']:.3f}"
            )

        return "\n".join(lines)

    def start_search(
        self,
        target_info: Dict[str, Any],
        current_time: float,
        task_name: Optional[str],
        subtask_id: Optional[int],
    ) -> Dict[str, Any]:
        target = target_info.get("display_target") or target_info.get("target_name") or "unknown target"
        record = {
            "task_name": task_name,
            "subtask_id": subtask_id,
            "target": target,
            "canonical_target": target_info.get("canonical_target"),
            "aliases": sorted(target_info.get("aliases") or []),
            "search_start_time": round(current_time, 2),
            "spotted_time": None,
            "search_time": None,
            "spotted_object_name": None,
            "spotted_timestep": None,
            "spotted_location": None,
        }
        self.search_records.append(record)
        self.current_search = record
        return record

    def update_search_from_observation(
        self,
        env: Any,
        observation: str,
        observation_time: float,
        timestep: int,
        location: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if self.current_search is None or self.current_search.get("spotted_time") is not None:
            return None

        target_aliases = set(self.current_search.get("aliases") or [])
        if not target_aliases:
            return None

        for object_name in self._extract_observed_object_names(env, observation):
            if not self._matches_target(object_name, target_aliases):
                continue

            spotted_time = max(float(observation_time), float(self.current_search["search_start_time"]))
            search_time = max(0.0, spotted_time - float(self.current_search["search_start_time"]))
            self.current_search.update({
                "spotted_time": round(spotted_time, 2),
                "search_time": round(search_time, 2),
                "spotted_object_name": object_name,
                "spotted_timestep": timestep,
                "spotted_location": location,
            })
            return self.current_search

        return None

    def get_retrieval_time_features(self) -> Dict[str, List[Dict[str, Any]]]:
        actr_only = []
        modified_only = []
        combined = []

        for report in self.reports:
            base = {
                "task_name": report.get("task_name"),
                "subtask_id": report.get("subtask_id"),
                "target": report.get("target"),
                "canonical_target": report.get("canonical_target"),
            }
            actr_only.append({
                **base,
                "actr_retrieval_time": report.get("actr_retrieval_time"),
            })
            modified_only.append({
                **base,
                "modified_retrieval_time": report.get("modified_retrieval_time"),
            })
            combined.append({
                **base,
                "actr_retrieval_time": report.get("actr_retrieval_time"),
                "modified_retrieval_time": report.get("modified_retrieval_time"),
            })

        return {
            "actr_only": actr_only,
            "modified_only": modified_only,
            "combined": combined,
        }

    @staticmethod
    def _ensure_plot_dir(output_path: str) -> None:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def _plot_labels(records: List[Dict[str, Any]]) -> List[str]:
        labels = []
        for record in records:
            subtask_id = record.get("subtask_id")
            target = record.get("target") or "unknown"
            labels.append(f"{subtask_id}: {target}" if subtask_id is not None else target)
        return labels

    @staticmethod
    def _finite_or_nan(value: Any) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return math.nan
        return numeric_value if math.isfinite(numeric_value) else math.nan

    @staticmethod
    def _annotate_plot_values(
        ax: Any,
        x_positions: List[int],
        values: List[float],
        color: str,
        y_offset: int = 8,
    ) -> None:
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
                color=color,
            )

    def save_activation_plot(self, output_path: str) -> bool:
        if not output_path or not self.reports:
            return False

        self._ensure_plot_dir(output_path)
        labels = self._plot_labels(self.reports)
        x_positions = list(range(len(self.reports)))
        width = 0.36
        actr_values = []
        modified_values = []

        for report in self.reports:
            actr = report.get("base_level_activation", float("nan"))
            actr_values.append(actr if math.isfinite(actr) else math.nan)
            modified_values.append(report.get("modified_activation_total", math.nan))

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
        left_positions = [x - width / 2 for x in x_positions]
        right_positions = [x + width / 2 for x in x_positions]
        ax.bar(left_positions, actr_values, width, label="ACT-R base-level", color="#d95f02")
        ax.bar(right_positions, modified_values, width, label="Modified activation", color="#1b9e77")
        ax.set_title("Activation Values by Find&Place Target")
        ax.set_ylabel("Activation value")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        for idx, report in enumerate(self.reports):
            actr = report.get("base_level_activation", float("nan"))
            if not math.isfinite(actr):
                ax.text(left_positions[idx], 0, "no evidence", rotation=90, va="bottom", ha="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return True

    def save_retrieval_time_plot(self, output_path: str, mode: str = "combined") -> bool:
        if not output_path or not self.reports:
            return False

        if mode not in {"actr", "modified", "combined"}:
            raise ValueError("mode must be one of: actr, modified, combined")

        self._ensure_plot_dir(output_path)
        labels = self._plot_labels(self.reports)
        x_positions = list(range(len(self.reports)))
        actr_values = [
            self._finite_or_nan(report.get("actr_retrieval_time"))
            for report in self.reports
        ]
        modified_values = [
            self._finite_or_nan(report.get("modified_retrieval_time"))
            for report in self.reports
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
        if mode in {"actr", "combined"}:
            ax.plot(
                x_positions,
                actr_values,
                marker="o",
                linewidth=2,
                label="ACT-R retrieval time",
                color="#d95f02",
            )
            self._annotate_plot_values(
                ax,
                x_positions,
                actr_values,
                color="#d95f02",
                y_offset=8 if mode == "actr" else 8,
            )
        if mode in {"modified", "combined"}:
            ax.plot(
                x_positions,
                modified_values,
                marker="o",
                linewidth=2,
                label="Modified retrieval time",
                color="#1b9e77",
            )
            self._annotate_plot_values(
                ax,
                x_positions,
                modified_values,
                color="#1b9e77",
                y_offset=8 if mode == "modified" else 20,
            )

        titles = {
            "actr": "ACT-R Retrieval Time by Find&Place Target",
            "modified": "Modified Retrieval Time by Find&Place Target",
            "combined": "Retrieval Time Comparison by Find&Place Target",
        }
        ax.set_title(titles[mode])
        ax.set_ylabel("Virtual seconds")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.margins(y=0.18)
        if mode == "combined":
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return True
