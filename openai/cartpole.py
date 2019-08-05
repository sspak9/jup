import os
import sys
import numpy as np
import pylab
import gym
from collections import deque

from agent import DoubleDQNAgent

EPISODES = 100000 # max episodes
MEAN_ARRAY_SIZE=30 # how many rewards to keep in average array, lander=100
REWARD_VALUE=500 # the reward that must be met for consecutive count, lander=200
STOP_AVG=495 # stop if 100 avg is >= this and consecutive count is >= STOP_COUNT, lander > 200
STOP_COUNT=35
RENDER_FLAG=True
RENDER_COUNT=25 # render if consecutive count >= this
SAVE_MODEL_FOLDER='./save_cartpole_model'
SAVE_PLOT_FOLDER='./save_plot'

# helper function
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

if __name__ == "__main__":
    
    create_dir(SAVE_MODEL_FOLDER)
    create_dir(SAVE_PLOT_FOLDER)

    env = gym.make('CartPole-v1')
    
    state_size = env.observation_space.shape[0]
    if hasattr(env.action_space , 'n'):
        action_size = env.action_space.n
    else:
        action_size = env.action_space.shape[0]

    print('action size:',action_size)

    agent = DoubleDQNAgent(state_size, action_size)
    
    consecutive_count = 0
    last_scores = deque(maxlen=MEAN_ARRAY_SIZE)
    mean_list = []

    for i in range(MEAN_ARRAY_SIZE):
        last_scores.append(0)

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        step = 0

        while not done:
            
            if RENDER_FLAG and consecutive_count >= RENDER_COUNT:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100
            
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)

            # every time step do the training
            agent.train_model()

            score += reward
            state = next_state
            step += 1

            if done:
                score = score if score == 500 else score + 100

                last_scores.append(score)
                mean = np.mean(last_scores)
                mean_list.append(mean)

                if score >= 500 :
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                
                # every episode update the target model to be same with model
                agent.update_target_model()
                
                print('episode:', e,'score:',score,'step:', step,"mean:",mean, 'count:',consecutive_count, 'mem:',len(agent.memory) , 'epsilon:', agent.epsilon)
                
                if mean > 0 and score >= REWARD_VALUE:
                    fname = 'm_avg_{:04d}_c_{:02d}_s_{:03d}_e_{:05d}'.format(int(mean+0.5),consecutive_count,int(score+0.5), int(e))
                    agent.target_model.save(SAVE_MODEL_FOLDER +'/' + fname + '.h5')
                                                                               
                if mean >= STOP_AVG and consecutive_count >= STOP_COUNT:
                    print('met stop criteria: mean >=',STOP_AVG,'and had consecutive step with >=', REWARD_VALUE, consecutive_count , 'steps')
                    pylab.plot(range(1,e+2) , mean_list , 'b')
                    pylab.savefig(SAVE_PLOT_FOLDER + '/cartpole_v1.png')
                    
                    sys.exit()

