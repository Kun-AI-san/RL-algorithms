# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:28:07 2022

@author: Karthik Vemireddy

"""

import gym
import numpy as np
import random

# create cliffwalking environment
env = gym.make('Taxi-v3')

# create a new instance of cliffwalking, and get the initial state
Q = np.zeros((env.observation_space.n, env.action_space.n))
num_steps = 0

epsilon = 0.1
#Train the agent
for j in range(1,3001):
    state = env.reset()
    print(f"Episode: {j}")
    observation=state[0]
    while True:
        # sample a random action from the list of available actions using epsilon greedy approach
        # this needs to be updated by Q-learning
        q = np.random.uniform(0,1)
        action = None
        if q<epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(Q[observation])
        # perform this action on the environment
        observation_1, reward, terminated, _, info = env.step(action)
        # Tentative best action evaluation
        Q[observation, action] = Q[observation, action] + 0.1*(reward+0.9*np.max(Q[observation_1]) - Q[observation, action])
        
        observation = observation_1
        if terminated:
            break

#Test the train agent
env = gym.make('Taxi-v3', render_mode='human')
state = env.reset()
score = 0
while True:
    # sample an argmax action from the list of available actions
    action = None
    action = np.argmax(Q[observation])
    # perform this action on the environment
    observation_1, reward, terminated, _, info = env.step(action)
    observation = observation_1
    num_steps += 1
    score += reward
    env.render()
    if terminated:
        break
print(f"Total number of steps taken is {num_steps} with total score of {score}")
# end this instance of the taxi environment
env.close()
