import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the Boxing environment
env = gym.make("Pong-v4")

reset_out = env.reset()
if isinstance(reset_out, tuple):
    obs, _ = reset_out
else:
    obs = reset_out

# 2. Warm up by taking 20 random steps (so we're not at the very start)
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
        # If the game ended during warm‐up, reset again
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out

# 3. Now capture frame1 (after warm-up)
frame1 = obs.copy()

# 4. Take one more random action to get frame2
action = env.action_space.sample()
step_out = env.step(action)
if len(step_out) == 4:
    frame2, reward, done, info = step_out
else:
    frame2, reward, terminated, truncated, info = step_out
    done = terminated or truncated

env.close()

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(frame1)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(frame2)
plt.axis("off")

plt.tight_layout()
plt.show()

