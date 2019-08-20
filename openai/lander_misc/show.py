from tensorflow import keras 
#import keras
import numpy as np 
import gym 
import sys

# tr.py [saved model] [env]
model_name = './lander.h5'
env_name = 'LunarLander-v2'
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    env_name = sys.argv[2]

model = keras.models.load_model(model_name) 
model.summary() 
 
env = gym.make(env_name) 
state_size = env.observation_space.shape[0]
 
for i_episode in range(10): 
    state = env.reset() 
    done = False 
    step = 0 
    score = 0
    while not done: 
        env.render() 
        s = state.reshape((1,state_size)) 
        q_value = model.predict(s) 
        action = np.argmax(q_value[0]) 
        state, reward, done, info = env.step(action)
        score += reward
        step +=1 
        if done: 
            print("Episode {} finished after {} timesteps, score: {}".format(i_episode+1,step,score)) 
            break

env.close() 