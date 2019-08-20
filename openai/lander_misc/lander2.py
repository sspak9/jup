import os
import sys
import numpy as np
import pylab
import gym
from collections import deque
from tensorflow import keras

from agent import DoubleDQNAgent

EPISODES = 100000 # max episodes
MEAN_ARRAY_SIZE=100 # how many rewards to keep in average array, lander=100
REWARD_VALUE=200 # the reward that must be met for consecutive count, lander=200
STOP_AVG=240 # stop if 100 avg is >= this and consecutive count is >= STOP_COUNT, lander > 200
STOP_COUNT=95
RENDER_FLAG=False
RENDER_COUNT=25 # render if consecutive count >= this
SAVE_MODEL_FOLDER='./save_lander_model'
SAVE_PLOT_FOLDER='./save_plot'
SAVE_VIDEO_FOLDER='./save_video'
MODIFIED_REWARD = 1000
ZERO_VALUE=0.00001

# helper function
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

if __name__ == "__main__":
    
    create_dir(SAVE_MODEL_FOLDER)
    create_dir(SAVE_PLOT_FOLDER)
    create_dir(SAVE_VIDEO_FOLDER)

    env = gym.make('LunarLander-v2')

    # uncomment below if you wish to save a video every 100 episodes. requires ffmpeg
    # AND you need to use gyglet 1.3.2 and nothing higher
    env = gym.wrappers.Monitor(env, SAVE_VIDEO_FOLDER, force=True, video_callable=lambda episode_id: episode_id%100==1)
    
    state_size = env.observation_space.shape[0]
    if hasattr(env.action_space , 'n'):
        action_size = env.action_space.n
    else:
        action_size = env.action_space.shape[0]
    
    agent = DoubleDQNAgent(state_size, action_size, 
        **{'memory':1000000 ,'layer1':256 , 'layer2':128})

    # if model defined, use the model
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        model = keras.models.load_model(model_name) 
        model.summary()
        agent.model = model
        agent.target_model = model
        agent.epsilon = 0.001

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
            n_state, reward, done, info = env.step(action)
            next_state = np.reshape(n_state, [1, state_size])
            
            # save the sample <s, a, r, s'> to the replay memory
            # modify reward done and has following:
            modified_reward = reward
            
            if ( step != 999 and (n_state[7] == 1 and n_state[6] == 1) and  # two legs are touched
                abs(n_state[5]) < ZERO_VALUE and # angular velocity = 0
                abs(n_state[4]) < 0.01 and # angle
                abs(n_state[3]) < ZERO_VALUE and abs(n_state[2]) < ZERO_VALUE  and # x,y velocity
                n_state[1] <= 0.0001 and n_state[1] > - 0.01 and # ypos
                abs(n_state[0]) < 0.1): # x pos
                # remove - step and add just the MODIFIED_REWARD if you find that solution does not converge
                modified_reward = reward + (MODIFIED_REWARD - step)
                        
            agent.append_sample(state, action, modified_reward, next_state, done)

            # every time step do the training
            agent.train_model()

            score += reward
            state = next_state
            step += 1

            if done:
                last_scores.append(score)
                mean = np.mean(last_scores)
                mean_list.append(mean)

                if score >= REWARD_VALUE :
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                
                #sync model to second model
                agent.update_target_model()
                flag = '*' if modified_reward != reward else ' '
                print(flag , 'episode:', e,'score:',score,'step:', step,"mean:",mean, 'count:',consecutive_count, 'mem:',len(agent.memory) , 'epsilon:', agent.epsilon)
                
                if mean > 0 and score >= REWARD_VALUE:
                    fname = 'm_avg_{:04d}_c_{:02d}_s_{:03d}_e_{:05d}'.format(int(mean+0.5),consecutive_count,int(score+0.5), int(e))
                    agent.target_model.save(SAVE_MODEL_FOLDER +'/' + fname + '.h5')
                                                                               
                if mean >= STOP_AVG and consecutive_count >= STOP_COUNT:
                    print('met stop criteria: mean >=',STOP_AVG,'and had consecutive step with >=', REWARD_VALUE, consecutive_count , 'steps')
                    pylab.plot(range(1,e+2) , mean_list , 'b')
                    pylab.savefig(SAVE_PLOT_FOLDER + '/lunarlander-v2.png')
                    
                    sys.exit()

