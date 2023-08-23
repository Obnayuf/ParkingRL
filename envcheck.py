import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import imageio

# config = {
#             "observation": {
#                 "type": "Kinematics"
#             },
#             "action": {
#                 "type": "DiscreteMetaAction"
#             },
#             "simulation_frequency": 15,  # [Hz]
#             "policy_frequency": 1,  # [Hz]
#             "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
#             "screen_width": 600,  # [px]
#             "screen_height": 150,  # [px]
#             "centering_position": [0.3, 0.5],
#             "scaling": 5.5,
#             "show_trajectories": True,
#             "render_agent": True,
#             "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
#             "manual_control": False,
#             "real_time_rendering": False
#         }
def normalize_image(img):
    min_val = np.min(img)
    max_val = np.max(img)
    normalized = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

output_dir = "check_gifs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

env = gym.make("parking-v0", render_mode='rgb_array')
# env.configure(config)
obs, info = env.reset()
done = truncated = False
ans = []
print(obs["image"].shape)
length = 0
while not (done or truncated):
    length += 1
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(reward)
    image = obs["image"][0] + obs["image"][1]*1.5 +  obs["image"][2]*2
    ans.append(np.squeeze(image))
    #env.render()

print(length)
ans_normalized = [normalize_image(img) for img in ans]

gif_filename = os.path.join(output_dir, f"check.gif")
imageio.mimsave(gif_filename, ans_normalized, duration=0.05)
# for i in range(len(ans)):
#     plt.clf()
#     plt.imshow(ans[i],cmap=plt.cm.gray, interpolation='nearest')
#     plt.show(block=False)
#     plt.pause(0.05)
