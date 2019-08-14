import gym
import os
import random
import numpy as np
from statistics import mean


env = gym.make('MountainCar-v0')
action_space = env.action_space.n  # 0 is push left, 1 is  no push and 2 is push right
observation_space = env.observation_space  # 0 is position [-1.2 - 0.6], 1 is velocity [-0.07 - 0.07]

print('action_space:',action_space,', observation_space:',observation_space)

# make Q table

v_buckets = [i / 100 for i in range(-7,8,1)]
p_buckets = [i / 10 for i in range(-12,7,1)]
print(v_buckets)
print(p_buckets)

Q = np.zeros([(len(v_buckets) * len(p_buckets)) , action_space])
print(Q.shape)


def get_q_row(state):
    # given state => poistion,velocity, retrieve the row that has 3 values associated with the action
    position = state[0]
    velocity = state[1]
    
    p_index = int(position*10 + 12 + 0.5)
    v_index = int(velocity*100 + 7 + 0.5)
    
    row = p_index * len(v_buckets) + v_index
    return row

# check above
print(get_q_row([-1.2,-0.07]))
print(get_q_row([0.6,0.07]))


EPISODES = 1000
STOP_COUNT = 50
SAVE_MODEL_FOLDER='.'

epsilon = 0.1  # probability of choosing a random action
epsilon_decay = 0.9
epsilon_min = 0.001
alpha = 0.5  # learning rate
gamma = 0.8  # discount rate

consecutive_count = 0

for e in range(1,EPISODES+1):
    observation = env.reset()
    step = 0
    done = False
    total_reward= 0
    state_q_row = get_q_row(observation)
    
    while not done:
        
        step +=1
        
        # get action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state_q_row])
    
        # get next step
        next_observation , reward , done , info = env.step(action)
        next_state_q_row = get_q_row(next_observation)
        
        # modify reward based on previous state: if difference in velocity is high, more reward
        modified_reward = reward +gamma * abs(next_observation[1]) - abs(observation[1])
        
        # update Q table with reward values
        Q[state_q_row, action] = (1 - alpha) * Q[state_q_row, action] + alpha * (
            modified_reward + gamma * np.max(Q[next_state_q_row]) - Q[state_q_row, action])
        
        state_q_row = next_state_q_row
        total_reward += reward
        
        if done:
            if next_observation[0] >= 0.5:
                consecutive_count += 1
            else:
                consecutive_count = 0
                
            print('episode:',e,', score:',total_reward,'m position:', next_observation[0], ',consecutive count:',consecutive_count, ',epsilon:',epsilon)
            if consecutive_count >= STOP_COUNT:
                print('stopping at consecutive count:',consecutive_count)
                np.save('qtable.npy', Q)
                env.close()
                exit()
    # reduce epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
 