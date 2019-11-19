import random


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

import numpy as np

from agent.QLearning import QLearningTable

import time
from absl import app

_NO_OP = actions.FUNCTIONS.no_op.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
ACTION_ATTACK = 'attack'

USE_PRE_TRAIN = False

#Creating all actions to add in the Q learning table, using the new dimension (7x3)

smart_actions = []

for mm_x in range(0, 7):
    for mm_y in range(0, 3):
        smart_actions.append(ACTION_ATTACK + '_' + str(mm_x) + '_' + str(mm_y))

class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self):
        super(MoveToBeacon, self).__init__()
        self.qlearn = QLearningTable(actions=smart_actions, use_pre_train=USE_PRE_TRAIN)
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None

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

    def convert_to_minimap(self, x, y):
        new_x = x*3+22
        new_y = y*5+28
        return [new_x, new_y]

    def splitAction(self, action):
        x = 0
        y = 0
        smart_action, x, y = action.split('_')
        x = int(x)
        y = int(y)
        return (smart_action, x, y)

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        if obs.first():
            self.previous_action = None
        '''
        if obs.last():
            score = obs.observation["score_cumulative"][0]
            self.resultat.append(score)
        '''

        # The minimap on this game is a 64x64 but, action only take place in the range x in [22,43] and y in [28,43]
        beacon_y, beacon_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.NEUTRAL).nonzero()

        # We use simplified coordinates to reduce dimensionality
        current_state = self.reduce_from_minimap(int(np.mean(beacon_x)), int(np.mean(beacon_y)))

        if self.previous_reward != obs.observation["score_cumulative"][0] and self.previous_action is not None:
            # if there was a change in the score and an action was done, we feed the Q-table with this action
            reward = 1
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))


        self.previous_reward = obs.observation["score_cumulative"][0]


        rl_action = self.qlearn.choose_action(
            str(current_state))  # we let the q-table decide of the next action and add the state if it doesn't exist

        action,x,y = self.splitAction(rl_action)  # rl_action is a string of form 'attack_0_1'

        target = self.convert_to_minimap(x, y)

        # We checked if the marine has reached his destination, and if not, keep attacking this position
        m_y, m_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()

        self.pos_marine = [m_x, m_y]

        if (self.pos_marine != target) & (_ATTACK_SCREEN in obs.observation['available_actions']):
            return actions.FUNCTIONS.Attack_minimap("now", target)

        self.previous_state = current_state
        self.previous_action = rl_action

        time.sleep(0.5)
        print(self.qlearn.q_table)
        print('Target : ' , target)

        if _ATTACK_SCREEN in obs.observation['available_actions']:
            return actions.FUNCTIONS.Attack_minimap("now", target)
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