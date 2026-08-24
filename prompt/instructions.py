ICL_INSTRUCTION = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 
At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation at that step (formatted as "observation at step {timestep}: <observation>"). You should first think about the current condition and plan for your future actions, and then output your action in this turn. 
If the observation output "Nothing happens", that means the previous action is invalid and you should try more options.
Always close any opened receptacles after using them.
Your output must strictly follow this format: "Thought: <your thoughts>
Action: <your next action>"

AVAILABLE ACTIONS:
1. go to {locations}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj/recep}
"""


ICL_INSTRUCTION_BDI = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal using Belief-Desire-Intention (BDI) reasoning. 
At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation at that step (formatted as "observation at step {timestep}: <observation>"). You should update your belief about the current state, specify your goal (desire), and output your intention (executable action) in this turn. 
If the observation output "Nothing happens", that means the previous action is invalid and you should try more options.
Always close any opened receptacles after using them.
Your output must strictly follow this format:
"Belief: <your belief about the current environment state>
Desire: <your target goal>
Intention: <your next executable action>"

AVAILABLE ACTIONS:
1. go to {locations}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj/recep}
"""


MEMORY_ENCODER_SYSTEM_INSTRUCTION = """You are an intelligent Memory Encoding Agent for an autonomous household robot.
Your task is to convert raw environmental observations into structured, highly concise natural-language memory traces.

Given an interaction (Timestep, Action, Observation):
Extract and summarize:
- Current Location
- Observed Receptacles & Objects (with exact item names/numbers)
- Container/Receptacle state (open, closed, empty, or containing items)

Keep your response extremely concise (1-2 lines maximum) focusing strictly on physical objects, locations, and container contents.
"""