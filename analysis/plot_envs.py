import gym
import numpy as np
import matplotlib.pyplot as plt

# 1. Create the Pong environment
env = gym.make("Pong-v4")
reset_out = env.reset()
if isinstance(reset_out, tuple):
    obs, _ = reset_out
else:
    obs = reset_out

# 2. Warm up by taking 40 random steps
warmup_steps = 40
for _ in range(warmup_steps):
    action = env.action_space.sample()
    step_out = env.step(action)
    if len(step_out) == 4:
        obs, reward, done, info = step_out
    else:
        obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated

    if done:
        # If the game ended during warm-up, reset again
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out

# 3. Capture four frames: the current obs + 3 more random steps
frames = [obs.copy()]
for _ in range(3):
    action = env.action_space.sample()
    step_out = env.step(action)
    if len(step_out) == 4:
        frame, reward, done, info = step_out
    else:
        frame, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    frames.append(frame.copy())

env.close()

# 4. Plot the 4 frames in a 1×4 row
plt.figure(figsize=(16, 4))   # wider figure to accommodate 4 panels

for i, frame in enumerate(frames, start=1):
    plt.subplot(1, 4, i)
    plt.imshow(frame)
    plt.axis("off")

plt.tight_layout()
output_filename = "pong_4frames_row_high_quality.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')