import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

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

DATA_FILE = 'sparse_agent_data'

smart_actions = []

for mm_x in range(0, 80):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))




class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

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
        
        q_target = r  # next state is terminal
            
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self):
        
        super(MoveToBeacon, self).__init__()
 
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = 0
        

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    
    def step(self, obs):
        
        super(MoveToBeacon, self).step(obs)
        time.sleep(0.2)
        
        if obs.first():
            self.previous_action = None
            self.previous_state = None
            
        
        current_state= np.zeros(16)
        hot_squares = np.zeros(16)
        
        beacon_y, beacon_x = (obs.observation["screen"][_PLAYER_RELATIVE] == _PLAYER_NEUTRAL).nonzero()

        

        
        for i in range(0, len(beacon_y)):
            y = int(math.ceil((beacon_y[i] + 1) / 16))
            x = int(math.ceil((beacon_y[i] + 1) / 16))
            
            
            hot_squares[((y - 1) * 4) + (x - 1)] = 1
            
        for i in range(0, 16):
            current_state[i] = hot_squares[i]



        if self.previous_reward != obs.observation["score_cumulative"][0] and self.previous_action is not None:
            
            reward = 1
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward,str(current_state))
            
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')           
            self.previous_action = None
            self.previous_state = None
        

        #if the marine hasn't reach the position of the previous action, we keep doing the same action

        if self.previous_action !=None :
            last_action, x, y = self.splitAction(self.previous_action)
            target=[int(x),int(y)]
            if last_action== ACTION_ATTACK:
                m_y, m_x = (obs.observation["screen"][_PLAYER_RELATIVE] == _PLAYER_FRIENDLY).nonzero()
                if m_x.any():
                    m_x = int(round(m_x.mean()))
                    m_y = int(round(m_y.mean()))
                    pos_marine = [m_x,m_y]
                    print ('destination marine : ',x,',',y)
                    print ('position marine : ',x,',',y)
            

                    if target != pos_marine:
                        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
            





        

        rl_action = self.qlearn.choose_action(str(current_state))
        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_reward = obs.observation["score_cumulative"][0]

        smart_action, x, y = self.splitAction(self.previous_action)
        target=[int(x),int(y)]

        if smart_action == ACTION_ATTACK:
            if _ATTACK_SCREEN in obs.observation['available_actions']:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
            else:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        
        return actions.FunctionCall(_NO_OP, [])

            
        
 




