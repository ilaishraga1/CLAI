from multi_taxi import multi_taxi_v0, maps, FuelType
import ray
import ray.rllib.algorithms.ppo as ppo
import ray.tune as tune
from ray.rllib import ExternalEnv
from ray.rllib.env import PettingZooEnv

from multi_taxi.env.MultiTaxiEnv import MultiTaxiEnv
from ray.air import CheckpointConfig
import pickle


ray.shutdown()
ray.init(ignore_reinit_error=True)

env = multi_taxi_v0.env(
    num_taxis=1,                       # there are 2 active taxis (agents) in the environment
    num_passengers=3,                  # there are 3 passengers in the environment
    max_capacity=[1],               # taxi_0 can carry 1 passenger, taxi_1 can carry 2
    max_fuel=[30],               # taxi_0 has a 30 step fuel limit, taxi1 has infinite fuel
    fuel_type=FuelType.GAS,            # taxis can only refuel at gas stations, marked "G" (only affects taxi_0)
    has_standby_action=True,           # all taxis can perform the standby action
    has_engine_control=[True],  # taxi_0 has engine control actions, taxi_1 does not
    domain_map=maps.HOURGLASS,         # the environment map is the pre-defined HOURGLASS map
    render_mode='human'  # MUST SPECIFY RENDER MODE TO ENABLE RENDERING
)


class MultiTaxiEnvWrapper(ExternalEnv):
    def __init__(self, env):
        super().__init__(action_space, observation_space)
        self.env = env

    def run(self):
        return self.env.run()



# BaseEnv, gymnasium.Env, gym.Env, MultiAgentEnv, VectorEnv, RemoteBaseEnv, ExternalMultiAgentEnv, ExternalEnv
tune.registry.register_env('multi_taxi_env', lambda _: PettingZooEnv(env))

results = tune.run(
    "PPO",
    stop={"training_iteration": 5},
    config={
        "env": "multi_taxi_env",  # "Taxi-v3"
        "num_workers": 1,
        "framework": "tf",  # Or "tf" for TensorFlow
    },
    keep_checkpoints_num=1,
    checkpoint_score_attr="episode_reward_mean",
    checkpoint_at_end=True,
)

tune.run(training=False)


# SELECT_ENV = "Taxi-v3"
#
# config = ppo.PPOConfig()
# config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
# config = config.resources(num_gpus=0)
# config = config.rollouts(num_rollout_workers=1)
# print(config.to_dict())
#
# with open("params.pkl", "wb") as f:
#     pickle.dump(config.to_dict(), f)
#
#
# algo = config.build(env=SELECT_ENV)
#
# N_ITER = 3
# s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
#
# for n in range(N_ITER):
#     result = algo.train()
#     file_name = algo.save(f"checkpoint_{n}")
#
#     print(s.format(
#         n + 1,
#         result["episode_reward_min"],
#         result["episode_reward_mean"],
#         result["episode_reward_max"],
#         result["episode_len_mean"]
#     ))
#
#
#
# policy = algo.get_policy()
# model = policy.model
# # print(model.base_model.summary())
