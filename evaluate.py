import pickle
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import os

class CustomModel(TorchModelV2, nn.Module):
    # ... (您之前提供的模型结构代码)
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Parameters for the image network (CNN)
        channels, kernel, stride = model_config["custom_model_config"]["conv_config"]
        linear = channels * ((100 - kernel) // stride + 1) * ((160 - kernel) // stride + 1)  # 计算线性层的输入节点数
        # Image Network
        self.image_model = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear, 256),
            nn.ReLU(),
        )

        # Vector Network
        self.vector_model = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
        )

        # Common Layers for Policy and Value Networks
        self.common_layer = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)
        self.feature = None

    def forward(self, input_dict, state, seq_lens):

        vector = input_dict["obs"]["vector"].float()
        image = input_dict["obs"]["image"].float()
        image_output = self.image_model(image)
        vector_output = self.vector_model(vector)

        x = torch.cat([image_output, vector_output], dim=1)
        self.feature = nn.functional.relu(self.common_layer(x))

        return self.policy_head(self.feature), state

    def value_function(self):
        return torch.reshape(self.value_head(self.feature), [-1])


def load_model_with_weights(num_outputs, model_config, weights):
    # 直接创建模型实例
    model = CustomModel(None, None, num_outputs, model_config, "CustomModel")
    for k, v in weights.items():
        if isinstance(v, np.ndarray):
            weights[k] = torch.tensor(v)
    del weights["append_bias_layer.log_std"]
    # Use the state_dict method to load weights
    model.load_state_dict(weights)

    return model

def normalize_image(img):
    min_val = np.min(img)
    max_val = np.max(img)
    normalized = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def evaluate(model, env, num_episodes=10, render=False):
    total_rewards = []


    # Ensure the model is in eval mode (important for things like dropout layers)
    model.eval()

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for episode in range(num_episodes):
            obs,info = env.reset()
            done = truncated = False
            total_reward = 0
            ans = []
            while not (done or truncated):
                # Prepare observation for the model
                input_dict = {
                    "obs": {
                        "vector": torch.tensor([obs['vector']]),  # Assuming 'vector' is part of the observation
                        "image": torch.tensor([obs['image']])     # Assuming 'image' is part of the observation
                    }
                }
                # Use the model to predict the action
                logits,_ = model(input_dict, None, None)
                action = logits[0,:2]
                
                # Take the action in the environment
                obs, reward, done, truncated, info = env.step(action)
                image = obs["image"][0] + obs["image"][1]*1.5 +  obs["image"][2]*2
                ans.append(np.squeeze(image))
                total_reward += reward

            total_rewards.append(total_reward)
            ans_normalized = [normalize_image(img) for img in ans]

            gif_filename = os.path.join(output_dir, f"evaluation_{episode}.gif")
            imageio.mimsave(gif_filename, ans_normalized, duration=0.05)
    # for i in range(len(ans)):
    #     plt.clf()
    #     plt.imshow(ans[i],cmap=plt.cm.gray, interpolation='nearest')
    #     plt.show(block=False)
    #     plt.pause(0.05)

    # Return average reward over the episodes
    return sum(total_rewards) / num_episodes


if __name__ == "__main__":


    # 1. 从检查点中加载模型参数
    with open("/home/wrjs/ray_results/PPO_my_env_2023-08-20_23-04-38otnqgvru/checkpoint_000401/policies/default_policy/policy_state.pkl", "rb") as f:
        data = pickle.load(f)

    # 准备模型所需的其他参数
    num_outputs = 2  # 由您提供的值
    model_config = {
        "custom_model_config": {
            "conv_config": [32, 5, 2]  # 您需要提供正确的值，或者使用默认值
        }
    }

    # 从您之前加载的数据中获取权重
    weights = data["weights"]

    # 使用上述函数加载模型并设置权重
    output_dir = "gifs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_model_with_weights(num_outputs, model_config, weights)
    env = gym.make("parking-v0", render_mode='rgb_array')
    # Evaluate the model on the environment
    average_reward = evaluate(model, env, num_episodes=10, render=True)
    print(f"Average reward over 10 episodes: {average_reward}")
