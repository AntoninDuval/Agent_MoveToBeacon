import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from absl import app
import time



_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

ACTION_ATTACK = 'attack'

USE_PRE_TRAIN = True


smart_actions = []

for mm_x in range(1, 80):
    for mm_y in range(1, 64):
        if mm_x % 8 == 0 and mm_y % 8 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 4) + '_' + str(mm_y - 4))


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, use_pre_train=False):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # use pre-train q_table
        if use_pre_train:
            self.q_table = pd.read_pickle(DATA_FILE)
            self.epsilon = 0.01
            logger.info('Qtable loaded')
            logger.info(str(self.q_table))

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

    def print_q_table(self, s, a):
        total_rows = self.q_table.count
        q_predict = self.q_table.ix[s, a]
        logger.info('Nombre de ligne dans la Q_table : ', str(total_rows))
        logger.info('action : ', str(a), ' q_value : ', str(q_predict))

class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self):
        super(MoveToBeacon, self).__init__()
        self.resultat = []
        self.episode = 0
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))), use_pre_train=USE_PRE_TRAIN)
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = 0
        self.destination_reached = False
        self.pos_marine = []

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def step(self, obs):

        super(MoveToBeacon, self).step(obs)

        time.sleep(0.5)
        if obs.first():
            self.previous_action = None

        if obs.last():
            score = obs.observation["score_cumulative"][0]
            self.resultat.append(score)
            self.episode += 1
            if self.episode % 20 == 0:
                df = pd.Series(self.resultat)
                df.to_csv(path='C:/Users/Antonin/Agent/score_newversion.csv', sep=';', index=True)

        current_state = np.zeros(275)
        beacon_y, beacon_x = (
                    obs.observation.feature_minimap.player_relative == features.PlayerRelative.NEUTRAL).nonzero()


        for i in range(0, (len(beacon_y) - 1)):
            current_state[i] = beacon_y[i]
        for j in range(0, (len(beacon_x) - 1)):
            current_state[j + 137] = beacon_x[j]
        if self.previous_reward != obs.observation["score_cumulative"][0] and self.previous_action is not None:
            # if there was a change in the score and an action was done, we feed the Q-table with this action
            reward = 1
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        if self.previous_reward == obs.observation["score_cumulative"][
            0] and self.previous_action is not None and self.destination_reached:
            # if the score doesn't change and the marine has reached his destination, we give a negative reward
            reward = -2
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        self.previous_reward = obs.observation["score_cumulative"][0]

        # we don't change anything while the marine hasn't reached his destination
        if self.previous_action != None:
            last_action, x, y = self.splitAction(self.previous_action)
            target = [int(x), int(y)]
            # print('target : ',x,',',y)
            if last_action == ACTION_ATTACK:
                m_y, m_x = (obs.observation.feature_screen.player_relative == features.PlayerRelative.SELF).nonzero()
                if m_x.any():
                    for i in range(0, len(m_x)):
                        pos_marine = [m_x[i], m_y[i]]
                        # print('pos_x(',i,'):',m_x[i],', pos_y(',i,'):',m_y[i])
                        if target == pos_marine or len(m_x) < 8:
                            self.destination_reached = True
                if not self.destination_reached:
                    return actions.FUNCTIONS.Attack_screen("now", target)

        marines = self.get_units_by_type(obs, units.Terran.Marine)

        marine = random.choice(marines)

        self.pos_marine = [marine.x, marine.y]

        if self.pos_marine != self.previous_pos_marine:
            return actions.FUNCTIONS.Attack_screen("now", target)
        self.previous_pos_marine = self.pos_marine


        rl_action = self.qlearn.choose_action(
            str(current_state))  # we let the q-table decide of the next action and add the state if it doesn't exist

        self.qlearn.print_q_table(str(current_state), rl_action)

        self.previous_state = current_state

        self.previous_action = rl_action

        smart_action, x, y = self.splitAction(self.previous_action)
        target = [int(x), int(y)]

        if smart_action == ACTION_ATTACK:
            if _ATTACK_SCREEN in obs.observation['available_actions']:
                self.destination_reached = False
                return actions.FUNCTIONS.Attack_screen("now", target)
            else:  # select the marine if attack is not possible
                return actions.FUNCTIONS.select_army("select")

        return actions.FunctionCall(_NO_OP, [])


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
