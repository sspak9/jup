import numpy as np 
import gym 
import sys

# tr.py [saved model] [env]
model_name = './qtable.npy'
env_name = 'MountainCar-v0'
q_table_row_size = 15

Q = np.load(model_name)
def get_q_row(state):
    # given state => poistion,velocity, retrieve the row that has 3 values associated with the action
    position = state[0]
    velocity = state[1]
    
    p_index = int(position*10 + 12 + 0.5)
    v_index = int(velocity*100 + 7 + 0.5)
    
    row = p_index * q_table_row_size + v_index
    return row

env = gym.make(env_name) 

for i_episode in range(10): 
    observation = env.reset()
    step = 0
    done = False
    total_reward= 0
    state_q_row = get_q_row(observation)

    while not done: 
        env.render()
        action = np.argmax(Q[state_q_row])
        next_observation, reward, done, info = env.step(action)
        total_reward += reward
        step +=1 
        state_q_row = get_q_row(next_observation)
        if done: 
            print("Episode {} finished after {} timesteps, score: {}".format(i_episode+1,step,total_reward)) 
            break

env.close() 