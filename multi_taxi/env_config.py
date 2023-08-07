from multi_taxi import multi_taxi_v0, maps, FuelType
from multi_taxi.env.reward_tables import TAXI_ENVIRONMENT_REWARDS
from multi_taxi.utils.types import Event

from ray.rllib.algorithms.ppo import PPO, PPOConfig

env_name = "multi_taxi_env"

config = PPOConfig()  \
    .environment(env=env_name, disable_env_checking=True) \
    .framework(framework="tf") \
    .rollouts(num_rollout_workers=0, enable_connectors=False) \
    .debugging(log_level="INFO")

custom_reward_table = TAXI_ENVIRONMENT_REWARDS
custom_reward_table[Event.BAD_DROPOFF] = -10
custom_reward_table[Event.BAD_PICKUP] = -10
# custom_reward_table[Event.INTERMEDIATE_DROPOFF] = -7
custom_reward_table[Event.PICKUP] = 5
custom_reward_table[Event.OUT_OF_TIME] = -20
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