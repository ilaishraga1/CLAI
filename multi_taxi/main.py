import time

from pprint import pprint
from IPython.display import clear_output

from multi_taxi import multi_taxi_v0, maps, FuelType
from multi_taxi.env.reward_tables import TAXI_ENVIRONMENT_REWARDS
from multi_taxi.utils.types import Event

import ray
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO

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
trainer = PPO(config=config)
trainer.restore("checkpoint_reward_520/checkpoint_000962/rllib_checkpoint.json")
# Train the agent
num_iterations = 10000
checkpoint_freq = 20  # Save checkpoints every 10 iterations

for iteration in range(521, num_iterations):
    result = trainer.train()
    pprint(f"Iteration {iteration}:")
    # pprint(f"{result}")

    # Optionally save checkpoints at specified iterations
    if iteration % checkpoint_freq == 0:
        checkpoint_name = f"checkpoint_reward_{iteration}"
        print(f"Saving checkpoint in {checkpoint_name}")
        checkpoint = trainer.save(checkpoint_name)

# Stop Ray when done
ray.shutdown()


# # exact same solution from the previous example
# solution = {'taxi_0': [0, 0, 4, 2, 2, 0, 5, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#                        6, 6, 6, 6, 6, 6, 6, 6, 6],
#             'taxi_1': [0, 4, 1, 3, 3, 3, 3, 3, 3, 3, 0, 4, 1, 2, 2, 1, 1, 1, 1, 1, 5, 0, 0,
#                        2, 1, 2, 2, 2, 2, 2, 1, 5]}


# # parallel API initial observation given on reset
# # this is a dictionary of observations, with agent names as keys
# observations = par_env.reset(seed=42)
# par_env.render()
# while True:
#     if any(not sol for sol in solution.values()):  # check solution complete without done
#         print('failure')
#         break

#     # arange next action as a joint action to be executed in parallel for all agents
#     joint_action = {agent: solution[agent].pop(0) for agent in par_env.agents}

#     # parallel API gets next observations, rewards, terms, truncs, and infos upon `step`
#     # all values are dictionaries
#     observations, rewards, terms, truncs, infos = par_env.step(joint_action)

#     # re-render after step
#     time.sleep(0.15)  # sleep for animation speed control
#     clear_output(wait=True)
#     par_env.render()  # clear previous render for animation effect

#     if any(terms.values()):  # check dones
#         print('success!')
#         break
#     if all(truncs.values()):
#         print('truncated')
#         break