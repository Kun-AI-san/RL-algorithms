# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:29:51 2022

@author: Karthik Vemireddy
UID: u7348473
"""

import gym
import numpy as np
import random

#epsilon greedy policy function
def epsilon_greedy_pol(env, observation, Q, epsilon=0.1):
    rand_val = np.random.uniform(0,1)
    if rand_val<epsilon:
        return np.random.choice(env.action_space.n)
    else:
        return np.argmax(Q[observation])
    
# create cliffwalking environment
env = gym.make('Taxi-v3')

# create a new instance of cliffwalking, and get the initial state
Q = np.zeros((env.observation_space.n, env.action_space.n))
num_steps = 0

epsilon = 0.1
#Train the agent
for j in range(1, 3001):
    state = env.reset()
    print(f"Episode: {j}")
    observation=state[0]
    while True:
        # sample a random action from the list of available actions using epsilon greedy approach
        # this needs to be updated by TD (0)
        action = epsilon_greedy_pol(env, observation, Q)
        # perform this action on the environment
        observation_1, reward, terminated, _, info = env.step(action)
        action_1 = epsilon_greedy_pol(env, observation_1, Q)
        #epsilon greedy evaluation
        Q[observation, action] = Q[observation, action] + 0.1*(reward+0.9*Q[observation_1, action_1] - Q[observation, action])
        observation = observation_1
        if terminated:
            break
        
#Test the agent
env = gym.make('Taxi-v3', render_mode='human')
state = env.reset()
score = 0
while True:
    # sample an argmax action from the list of available actions
    q = np.random.uniform(0,1)
    action = None
    action = np.argmax(Q[observation])
    # perform this action on the environment
    observation_1, reward, terminated, _, info = env.step(action)
    print(observation, action)
    if observation==observation_1:
        observation=state[0]
    observation = observation_1
    num_steps += 1
    score += reward
    env.render()
    if terminated:
        break
print(f"Total number of steps taken is {num_steps} with a total score of {score}")
# end this instance of the taxi environment
env.close()