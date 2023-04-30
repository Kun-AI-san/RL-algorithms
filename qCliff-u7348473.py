# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:27:39 2022

@author: Karthik Vemireddy
UID: u7348473
"""

import gym
import numpy as np
import random

# create cliffwalking environment
env = gym.make('CliffWalking-v0')

# create a new instance of cliffwalking, and get the initial state
Q = np.zeros((env.observation_space.n, env.action_space.n))
num_steps = 0

epsilon = 0.1
#Train the agent
for j in range(1, 501):
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
env = gym.make('CliffWalking-v0', render_mode='human')
state = env.reset()
score = 0
while True:
    # sample an armax action from the list of available actions 
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
print(f"Total number of steps taken is {num_steps} with a total score of {score}")
# end this instance of the taxi environment
env.close()