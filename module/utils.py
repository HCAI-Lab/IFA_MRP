import os
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from termcolor import colored

def add_frame_overlay(frame, step_num: int, action: str, task_name: str = ""):
    """ Add a top banner overlay showing the step number, task name, and executed action. """
    frame = frame.copy()
    h, w = frame.shape[:2]
    action_clean = str(action).translate(str.maketrans("‘’“”", "''\"\""))
    task_clean = str(task_name).translate(str.maketrans("‘’“”", "''\"\""))
    
    cv2.rectangle(frame, (0, 0), (w, 42), (0, 0, 0), -1)
    line1 = f"Task: {task_clean} | Step {step_num}" if task_clean else f"Step {step_num}"
    line2 = f"Action: {action_clean}"
    
    cv2.putText(frame, line1, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def save_video_animation(frames, output_path: str = "execution_video.gif", interval: int = 200):
    """ Render and save frame sequence to GIF animation. """
    if not frames:
        print(f"No frames available to save to {output_path}")
        return
    
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    image = ax.imshow(frames[0])

    def update(idx):
        image.set_data(frames[idx])
        return (image,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
    anim.save(output_path, savefig_kwargs={"pad_inches": 0})
    plt.close(fig)
    print(colored(f"🎬 Saved execution video animation: {output_path}", "green", attrs=["bold"]))

def save_json(data, output_path: str):
    """ Export data dictionary or list to pretty JSON file. """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(colored(f"📄 Saved JSON output: {output_path}", "green", attrs=["bold"]))
