import numpy as np
from constants import *
import pandas as pd
from tabulate import tabulate


class GridWorld:
    def __init__(self, w : int , h : int, s_states : list, d_states : list, st_state : tuple = DEFAULT_START) -> None:
        self.map = np.zeros((w , h))
        # print(self.map)
        for i in s_states: self.map[i] = SUCCESS_REWARD
        for i in d_states: self.map[i] = DANGER_REWARD
        # print(self.map)
        self.s_states = s_states
        self.d_states = d_states
        self.state = st_state
        self.lr = LEARNING_RATE
        self.itr = NUM_OF_ITERATIONS
        self.stochasticity = EXPLORE_RATE


    def getAction(self):
        actions = [UP, DOWN, LEFT, RIGHT]
        lst = [0,1,2,3]
        if np.random.uniform(0,1) < self.stochasticity: 
            action = actions[np.random.choice(lst)]
        else:
            maxReward = -9999999
            action = None
            for i in actions:
                if (self.state[0] + i[0]) >= 0 and (self.state[0] + i[0] < 10):
                    if (self.state[1] + i[1]) >= 0 and (self.state[1] + i[1] < 10):
                        if self.map[(self.state[0] + i[0], self.state[1] + i[1])] > maxReward:
                            action = i
                            maxReward = self.map[(self.state[0] + i[0], self.state[1] + i[1])]
            if maxReward == 0:
                action = actions[np.random.choice(lst)]
        return action


    def getValue(self, action):
        desired_state = None
        if self.state[0] + action[0] < 0 or self.state[0] + action[0] >= 10 or self.state[1] + action[1] < 0 or self.state[1] + action[1] >= 10:
                desired_state = self.state
        else:
            desired_state = (self.state[0] + action[0], self.state[1] + action[1])

        v = 0
        v += DESIRED_MOVE*(self.map[self.state] + self.lr*self.map[desired_state])
        if action[0] == 0: #left or right
            if self.state[0] - 1 >= 0:
                v += ERROR_MOVE*(self.map[self.state] + self.lr*self.map[self.state[0] - 1, self.state[1]])
            if self.state[0] + 1 < 10:
                v += ERROR_MOVE*(self.map[self.state] + self.lr*self.map[self.state[0] + 1, self.state[1]])
        else:
            if action[1] == 0: #up or down
                if self.state[1] - 1 >= 0:
                    v += ERROR_MOVE*(self.map[self.state] + self.lr*self.map[self.state[0], self.state[1] - 1])
                if self.state[1] + 1 < 10:
                    v += ERROR_MOVE*(self.map[self.state] + self.lr*self.map[self.state[0], self.state[1] + 1])
        # if v >= 100: return 95
        # if v <= -100: return -95
        # else: 
        return v 

    def performAction(self, action, value):
        next_state = None
        if self.state[0] + action[0] < 0 or self.state[0] + action[0] >= 10 or self.state[1] + action[1] < 0 or self.state[1] + action[1] >= 10:
                next_state = self.state
        else :
            next_state = (self.state[0] + action[0], self.state[1] + action[1])
        self.map[self.state] = value
        self.state = next_state

    def isTerminating(self):
        if self.state in self.s_states:
            return True
        if self.state in self.d_states:
            return True
        return False


    def trainAgent(self):
        for _ in range(self.itr):
            #select next state 
            while self.isTerminating() == False:
                action = self.getAction()
                value = self.getValue(action) #Bellman Equation
                self.performAction(action, value)
            
            self.state = DEFAULT_START
    
    def visualize(self):
        x = [[0 for i in range(10)] for _ in range(10)]
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                x[i][j] = self.findDir(i,j)
                
        for i in self.s_states: x[i[0]][i[1]] = u'\u2705'
        for i in self.d_states: x[i[0]][i[1]] = u'\u274C'
        # for i in self.s_states: x[i[0]][i[1]] = u'\1F7E9'
        # for i in self.d_states: x[i[0]][i[1]] = '-100'
        return x
                
    def findDir(self, i , j):
        actions = [UP, DOWN, LEFT, RIGHT]
        maxReward = -9999999
        action = None
        for a in actions:
            if (i + a[0]) >= 0 and (i + a[0] < 10):
                if (j + a[1]) >= 0 and (j + a[1] < 10):
                    if self.map[(i + a[0], j + a[1])] > maxReward:
                        action = a
                        maxReward = self.map[(i + a[0], j + a[1])]
        if action == UP: return u'\u2191'
        if action == DOWN: return u'\u2193'
        if action == LEFT: return u'\u2190'
        if action == RIGHT: return u'\u2192'
    
    
    
            



def main():
    envz = GridWorld(10,10, [(9,9), (5,8), (8,3)], [(5,5), (4,4), (6,6),])
    envz.trainAgent()
    print(envz.map)
    print(pd.DataFrame(envz.visualize()))

main()
