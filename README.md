# BDI Memory Agent for Household Tasks

This repository contains a dual-agent household task runner built around AI2-THOR. A decision-making agent chooses actions using Belief-Desire-Intention style reasoning, while a memory-encoding agent converts observations into structured memories that can be retrieved during later find-and-place subtasks.

The runner tracks virtual action time, object exposure, scene entropy, activation values, and predicted memory retrieval time for find-and-place behavior.

## Repository Structure

```text
agents/
  dm_agent.py              Decision-making agent wrapper
  me_agent.py              Memory encoding and retrieval logic
  llm_gpt.py               OpenAI chat model wrapper
  llm_public.py            Optional Hugging Face model wrapper

env/
  env.py                   AI2-THOR household environment
  tasks.py                 Exploration, rearrangement, and findNplace tasks
  constraints.py           Scene locations and excluded objects
  constants.py             Object and receptacle constants
  metadata/                Goal-state metadata for task evaluation

module/
  runner.py                Main agent loop
  message_context.py       Main/findNplace prompt-context management
  activation_tracker.py    Activation and retrieval-time calculations
  time_board.py            Virtual action/dwell/break timing model
  action_retry.py          LLM action parsing and retry logic
  artifacts.py             Chat, video, timeline, and plot export helpers
  subtask_stopper.py       Optional expected-step subtask stopper

prompt/
  instructions.py          Agent instructions
  builders.py              ICL prompt builders
  hfes_icl_example_bdi.json
  hfes_icl_example_find_bdi.json

utils/
  retrieval_time_plots.py  Retrieval-time plotting utilities

data/
  user_data.json           Timing data used by the virtual time board
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Platform Notes

### Apple Silicon Mac

This project has been tested on an ARM-based Mac.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then open `main.ipynb` and run the notebook cells from top to bottom.

### Linux

Linux setup instructions will be added later.

## Quick Start

Open `main.ipynb` and run the cells from top to bottom. The notebook initializes the AI2-THOR household environment, creates the decision-making and memory-encoding agents, runs `run_agent_loop`, and saves the configured outputs.

## Debug Options

`run_agent_loop` always prints the predicted retrieval time during find-and-place subtasks using this label:

```text
Predicted Memory Retrieval Time:
```

Set `debug=True` to print detailed timeline calculations, object affordances, scene entropy, and activation evidence/values. Set `debug_memories=True` to print the retrieved subtask-relevant memories injected into the find-and-place prompt.

## Outputs

Depending on the options passed to `run_agent_loop`, the code can generate:

- `chat.json`: full stacked message transcript
- `chat_findnplace.json`: find-and-place message context
- `video.gif`: execution video
- `actr_retrieval_times.png`: ACT-R retrieval-time plot
- `modified_retrieval_times.png`: modified retrieval-time plot
- `combined_retrieval_times.png`: combined retrieval-time plot
- timeline and memory JSON files if output paths are provided

These are generated artifacts, not required source files.
