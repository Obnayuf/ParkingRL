import argparse
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random

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
from environment import BasicEnv_1


torch, nn = try_import_torch()
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
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=5000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1e7, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=12, help="Reward at which we stop training."
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
    args = parser.parse_args()
    checkpoint_freq = 1000
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(BasicEnv_1, env_config= {})
        .framework(args.framework)
        .rollouts(num_rollout_workers=16)
        .training(
            model={
                "fcnet_hiddens": [128,128],  # specify hidden layer sizes
                "fcnet_activation": "tanh",  # specify the activation function
                "vf_share_layers": False,  # change this to False if you don't want to share layers
            },
            train_batch_size=1024*4,
            num_sgd_iter=50,
            lambda_ = 0.96,
            entropy_coeff=0.02,
            clip_param=0.2,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=8)
    )
    config["log_level"]="DEBUG"
    stop = {
        "episode_reward_mean": args.stop_reward,
    }
    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        config.gamma = 0.99
        algo = config.build()
        # run manual training loop and print results after each iteration
        for i in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            if i%checkpoint_freq==0:
                checkpoint = algo.save()
                print("save once")
            # stop training of the target train steps or reward are reached
            if (
                result["episode_reward_mean"] >= args.stop_reward
            ):
                break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, local_dir ="./results"),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()