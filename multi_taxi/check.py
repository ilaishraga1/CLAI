import time

from IPython.display import clear_output
from PIL import Image

from multi_taxi import multi_taxi_v0, maps, FuelType
from multi_taxi.env.reward_tables import TAXI_ENVIRONMENT_REWARDS
from multi_taxi.utils.types import Event

import ray
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.a3c import A3C



ACTIONs = {0: 'south',
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
custom_reward_table[Event.PICKUP] = 5
custom_reward_table[Event.OUT_OF_TIME] = -20
custom_reward_table[Event.STEP] = -1

# using the PettingZoo parallel API here
env = multi_taxi_v0.env(
    num_taxis=1,                       # there are 2 active taxis (agents) in the environment
    num_passengers=3,                  # there are 3 passengers in the environment
    max_steps=1000,
    reward_table=custom_reward_table,
    intermediate_dropoff_reward_by_distance=True,
    max_capacity=[1],               # taxi_0 can carry 1 passenger, taxi_1 can carry 2
    max_fuel=[None],               # taxi_0 has a 30 step fuel limit, taxi1 has infinite fuel
    fuel_type=FuelType.GAS,            # taxis can only refuel at gas stations, marked "G" (only affects taxi_0)
    has_standby_action=True,           # all taxis can perform the standby action
    has_engine_control=[False],  # taxi_0 has engine control actions, taxi_1 does not
    domain_map=maps.SMALL_NO_OBS_MAP,         # the environment map is the pre-defined HOURGLASS map
    render_mode='human'  # MUST SPECIFY RENDER MODE TO ENABLE RENDERING
)
# env.reset(seed=42)

def create_env(config):
    return env

ray.init()
tune.register_env("multi_taxi_env", lambda config: PettingZooEnv(create_env(config)))

# Initialize the PPO trainer
config = {
    "env": "multi_taxi_env",
    "framework": "tf",  # Or "tf" for TensorFlow
    "num_workers": 1,      # Number of worker processes for parallel sampling
    "lr": 0.0005,          # Lower learning rate for more stable learning
    "entropy_coeff": 0.01, # Higher entropy coefficient for more exploration
    "gamma": 0.99,         # Adjust gamma based on the time horizon of the environment
    "rollout_fragment_length": 32,  # Lower number of
    # Add more PPO-specific config parameters as needed
}
agent = PPO(config=config)
agent.restore("checkpoint_reward_640/checkpoint_001082/rllib_checkpoint.json")

# Run the policy in the environment
env = create_env({})

reward_sum = 0

env.reset(seed=42)
env.render()
for a in env.agent_iter():

    # observation, reward, term, trunc, and info given one by one via the `last` method
    observation, reward, term, trunc, info = env.last()
    reward_sum += reward
    if term:  # check done status
        print('success!')
        break
    if trunc:
        print('truncated')
        break

    # get next action from predefined solution
    action = agent.compute_single_action(observation)
    print(f"Chosen action is {ACTIONs[action]}")
    env.step(action)

    # re-render after step
    if  a == env.possible_agents[-1]:
        print(f"reward summary: {reward_sum}")
        # state only changes after both taxis have stepped
        # time.sleep(0.05)  # sleep for animation speed control
        clear_output(wait=True)  # clear previous render for animation effect
        env.render()

env.close()