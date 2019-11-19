import pandas as pd
import numpy as np
from agent.Logger import Logger

DATA_FILE = '../data/sparse_agent_data'




class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, use_pre_train=False):
        self.logger = Logger().logger
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # use pre-train q_table
        if use_pre_train:
            self.q_table = pd.read_pickle(DATA_FILE)
            self.epsilon = 0.01
            self.logger.info('Qtable loaded')
            self.logger.info(str(self.q_table))

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
        self.logger.info('Nombre de ligne dans la Q_table : ', str(total_rows))
        self.logger.info('action : ', str(a), ' q_value : ', str(q_predict))
