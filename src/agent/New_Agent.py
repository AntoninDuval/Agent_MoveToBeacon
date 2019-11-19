import random


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

import numpy as np

from agent import QLearning

import time
from absl import app

_NO_OP = actions.FUNCTIONS.no_op.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self):
        super(MoveToBeacon, self).__init__()

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def reduce_from_minimap(self,x,y):
        new_x = (x-22)//3
        new_y = (y-28)//5
        return [new_x, new_y]

    def convert_to_minimap(self,x,y):
        new_x = x*3+22
        new_y = y*5+28
        return [new_x, new_y]

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        # The minimap on this game is a 64x64 but, action only take place in the range x in [22,43] and y in [28,43]

        beacon_y, beacon_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.NEUTRAL).nonzero()

        # We use simplified coordinates to reduce dimensionality
        position_beacon = self.reduce_from_minimap(int(np.mean(beacon_x)), int(np.mean(beacon_y)))

        
        time.sleep(0.2)


        if _ATTACK_SCREEN in obs.observation['available_actions']:
                return actions.FUNCTIONS.Attack_minimap("now", [43,28])
        else:  # select the marine if attack is not possible
                return actions.FUNCTIONS.select_army("select")



def main(unused):
    agent = MoveToBeacon()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="MoveToBeacon",
                    players=[sc2_env.Agent(sc2_env.Race.terran)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)