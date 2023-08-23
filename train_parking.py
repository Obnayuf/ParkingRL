import torch
from torch import nn
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.visionnet import VisionNetwork as TorchVision
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import AppendBiasLayer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import ray
from ray import tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
import torch.nn.init as init
import argparse
import os

torch, nn = try_import_torch()

def init_hidden_layer(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        init.orthogonal_(layer.weight.data, gain=init.calculate_gain('relu')) # 使用 \sqrt{2} 的缩放因子
        init.constant_(layer.bias.data, 0)

# 策略输出层的初始化
def init_policy_layer(layer):
    if isinstance(layer, nn.Linear):
        init.orthogonal_(layer.weight.data, gain=0.01)  # 使用 0.01 的缩放因子
        init.constant_(layer.bias.data, 0)

# 值输出层的初始化
def init_value_layer(layer):
    if isinstance(layer, nn.Linear):
        init.orthogonal_(layer.weight.data, gain=1.0)  # 使用 1.0 的缩放因子
        init.constant_(layer.bias.data, 0)

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Parameters for the image network (CNN)
        channels, kernel, stride = model_config["custom_model_config"]["conv_config"]
        linear = channels * ((100 - kernel) // stride + 1) * ((160 - kernel) // stride + 1)  # 计算线性层的输入节点数
        self.free_log_std = True

        if self.free_log_std:
            assert num_outputs % 2 == 0, "num_outputs must be divisible by two"
            num_outputs = num_outputs // 2
            self.append_bias_layer = AppendBiasLayer(num_outputs)

        # Image Network
        self.image_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 25 * 40, 128),
            nn.ReLU(),
        )

        # Vector Network
        self.vector_policy_model = nn.Sequential(
            nn.Linear(12, 128),
            nn.Tanh(),
        )

        self.vector_value_model = nn.Sequential(
            nn.Linear(12, 128),
            nn.Tanh(),
        )
        # Common Layers for Policy and Value Networks
        #self.common_layer = nn.Linear(512, 256)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, num_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.feature = None

        # 你的模型中各层的初始化
        self.image_model.apply(init_hidden_layer)    # 对 image_model 中的层进行初始化

        self.vector_value_model.apply(init_hidden_layer)   # 对 vector_model 中的层进行初始化
        self.vector_policy_model.apply(init_hidden_layer)   # 对 vector_model 中的层进行初始化

        self.policy_head.apply(init_policy_layer)    # 对 policy_head 进行初始化
        self.value_head.apply(init_value_layer)      # 对 value_head 进行初始化

    def forward(self, input_dict, state, seq_lens):

        vector = input_dict["obs"]["vector"].float()
        image = input_dict["obs"]["image"].float()
        image_output  = self.image_model(image)
        vector_output = self.vector_policy_model(vector)

        self.feature = torch.cat([image_output, self.vector_value_model(vector)], dim=1)

        x = torch.cat([image_output, vector_output], dim=1)

        logits = torch.tanh(self.policy_head(x))
        # 如果启用了free_log_std，使用AppendBiasLayer添加log_std到输出上
        if self.free_log_std:
            logits = self.append_bias_layer(logits)
        return logits, state


    def value_function(self):
        if self.feature is None:
            print("model bug")
        return torch.reshape(self.value_head(self.feature), [-1])



ModelCatalog.register_custom_model("my_custom_model", CustomModel)

def env_creator(env_config):
    return gym.make("parking-v0", render_mode='rgb_array')  # return an env instance

register_env("my_env", env_creator)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--stop-iters", type=int, default=5000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1e7, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=-1, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    default=True,
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # 在ray.init()之前设置这个环境变量:


    args = parser.parse_args()
    checkpoint_freq = 100
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=args.local_mode,_temp_dir = "/home/wrjs/big/ray_temp")
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment("my_env", env_config={})
        .framework(args.framework)
        .rollouts(num_rollout_workers=30)
        .training(
            model={
                "custom_model": "my_custom_model",
                "custom_model_config": {
                    "conv_config": [16, 5, 2],  # channels, kernel, stride
                },
            },
            train_batch_size=512,
            sgd_minibatch_size = 16,
            num_sgd_iter=60,
            lambda_=0.96,
            entropy_coeff=0.02,
            clip_param=0.2,
            use_gae = True,
        )
        # Use GPUs 8
        .resources(num_gpus=8)
    )
    config["gamma"] = 0.96
    config["log_level"] = "DEBUG"
    stop = {
        "episode_reward_mean": args.stop_reward,
    }
    config.lr = 1e-3
    algo = config.build()
    for i in range(2000):
        result = algo.train()
        print(pretty_print(result))
        if result["episode_reward_mean"]>=-1:
            break
        if i % checkpoint_freq == 0:  # save model every checkpoint_freq iterations
            checkpoint = algo.save()
            print(f"Checkpoint saved at {checkpoint}")
            print("save once")
    algo.stop()
   
