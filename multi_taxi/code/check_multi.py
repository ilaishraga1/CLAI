import time

from IPython.display import clear_output

from multi_taxi import multi_taxi_v0, maps, FuelType
from multi_taxi.env.reward_tables import TAXI_ENVIRONMENT_REWARDS
from multi_taxi.utils.types import Event

import ray
from ray import tune
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.a3c import A3C



ACTIONS = {0: 'south',
 1: 'north',
 2: 'east',
 3: 'west',
 4: 'pickup',
 5: 'dropoff',
 6: 'standby'}


# using the PettingZoo parallel API here

custom_reward_table = TAXI_ENVIRONMENT_REWARDS
custom_reward_table[Event.BAD_DROPOFF] = -10
custom_reward_table[Event.BAD_PICKUP] = -10
# custom_reward_table[Event.INTERMEDIATE_DROPOFF] = -7
custom_reward_table[Event.PICKUP] = 10
custom_reward_table[Event.OUT_OF_TIME] = -200
custom_reward_table[Event.STEP] = -1

# using the PettingZoo parallel API here
par_env = multi_taxi_v0.parallel_env(
    num_taxis=2,                       # there are 2 active taxis (agents) in the environment
    num_passengers=3,                  # there are 3 passengers in the environment
    max_steps=1000,
    reward_table=custom_reward_table,
    intermediate_dropoff_reward_by_distance=True,
    max_capacity=[1, 2],               # taxi_0 can carry 1 passenger, taxi_1 can carry 2
    max_fuel=[None, None],               # taxi_0 has a 30 step fuel limit, taxi1 has infinite fuel
    fuel_type=FuelType.GAS,            # taxis can only refuel at gas stations, marked "G" (only affects taxi_0)
    has_standby_action=True,           # all taxis can perform the standby action
    has_engine_control=[False, False],  # taxi_0 has engine control actions, taxi_1 does not
    domain_map=maps.SMALL_NO_OBS_MAP,         # the environment map is the pre-defined HOURGLASS map
    render_mode='human'  # MUST SPECIFY RENDER MODE TO ENABLE RENDERING
)
# env.reset(seed=42)

def create_env(config):
    return par_env

def print_actions(actions):
    for taxi, action in actions.items():
        print(f"{taxi} - {ACTIONS[action]}")

env_name = "multi_taxi_env"

ray.init()
tune.register_env(env_name, lambda config: ParallelPettingZooEnv(create_env(config)))

# Initialize the PPO trainer
config = PPOConfig()  \
    .environment(env=env_name, disable_env_checking=True) \
    .framework(framework="tf") \
    .rollouts(num_rollout_workers=0, enable_connectors=False) \
    .debugging(log_level="INFO")
    # .training(
    #         train_batch_size=512,
    #         lr=2e-5,
    #         gamma=0.99,
    #         lambda_=0.9,
    #         use_gae=True,
    #         clip_param=0.4,
    #         grad_clip=None,
    #         entropy_coeff=0.1,
    #         vf_loss_coeff=0.25,
    #         sgd_minibatch_size=64,
    #         num_sgd_iter=10,
    #     )

agent = PPO(config=config)
# agent.restore("ray_results/multi_taxi_env/PPO/PPO_multi_taxi_env_b3aa9_00000_0_2023-08-07_01-58-05/checkpoint_000030/rllib_checkpoint.json")

# Run the policy in the environment
env = create_env({})

reward_sum = 0

obs = env.reset(seed=42)
env.render()

episode_reward = 0.0
term = False
trunc = False

for i in range(1000):
    # Get actions from the policy
    action_dict = agent.compute_actions(obs)
    print_actions(action_dict)
    # Step the environment with the chosen actions
    next_obs, rewards, term, trunc, info = env.step(action_dict)
    
    # Update the episode reward
    episode_reward += sum(rewards.values())

    obs = next_obs

    # time.sleep(0.1)
    env.render()

    print(f"Total Reward: {episode_reward}")

ray.shutdown()
env.close()