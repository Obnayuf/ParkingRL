import pickle
with open("/home/wrjs/ray_results/PPO_my_env_2023-08-20_23-04-38otnqgvru/checkpoint_000501/policies/default_policy/policy_state.pkl", "rb") as f:
    data = pickle.load(f)
print(data["policy_spec"])