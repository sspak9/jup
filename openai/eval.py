#from tensorflow import keras
import keras
import numpy as np
import gym
import os
import traceback
import argparse
 
class ModelEvaluator:
  def __init__(self):
    self.model_name = None
    self.model=None
    self.env = None
 
  def file_exists(self,file_name):
    if os.path.exists(file_name) and os.path.isfile(file_name):
      return True
 
    print('*** file does not exist:', file_name)
    return False
 
  def read_model(self,model_name):
   
    if  model_name:
      if self.file_exists(model_name):
        try:
          self.model = keras.models.load_model(model_name)
          self.model_name = model_name
          #self.model.summary()
          return
        except Exception as ex:
          traceback.print_exc()
 
    #print('in read. model name is:', model_name)
    self.model = None
    self.mode_name = None
 

  def setup_env(self,env_name):
    try:
      self.env = gym.make(env_name)
    except Exception as ex:
      traceback.print_exception(type(ex), ex, ex.__traceback__)
      self.env = None
 
  def close_env(self):
    if self.env:
      self.env.close()
      self.env = None
 
  def get_action(self,state):
    s = state.reshape((1,4))
    result = self.model.predict(s)
    action = np.argmax(result[0])
    return action
 
  def run_model(self):
    # return score , initial_state , [[state , reward , done , info]]
    score = 0
    done = False
    result_list = []
   
    initial_state = self.env.reset()
    state = initial_state
 
    while not done:
      action = self.get_action(state)
      state, reward, done, info = self.env.step(action)
      result_list.append([state,reward,done,info])
      score += 1
      if done:
        return [score , initial_state , result_list]
 
  def get_diff(self, initial_state, result_list):
    pos = initial_state[0]
    vel = initial_state[1]
    ang = initial_state[2]
    pol = initial_state[3]
 
    pos_diff = 0.0
    vel_diff = 0.0
    ang_diff = 0.0
    pol_diff = 0.0
 
    for rec in result_list:
      pos_diff += abs( pos - rec[0][0])
      vel_diff += abs( vel - rec[0][1])
      ang_diff += abs( ang - rec[0][2])
      pol_diff += abs( pol - rec[0][3])
 
      pos = rec[0][0]
      vel = rec[0][1]
      ang = rec[0][2]
      pol = rec[0][3]
 
    return [pos_diff , vel_diff , ang_diff , pol_diff]
 
  def evaluate_model(self, n=1):
    score = 0.0
    diff = 0.0
    sum = [0.0 , 0.0 , 0.0, 0.0]
 
    for i in range(n):
      result = self.run_model()
      score += result[0]
      last_index = len(result[2]) - 1
      diff += abs(result[1][0] - result[2][last_index][0][0])
      r = self.get_diff(result[1], result[2])
      for i in range(len(r)):
        sum[i] += r[i]
 
    print('model:', self.model_name, 'score:' , score / n, 'x:',diff/n, 'pos:' , sum[0] / n , 'vel:', sum[1]/n , 'ang:', sum[2]/n , 'pol:', sum[3]/n)
 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate Models')
  parser.add_argument('--m', default='model.h5', help='set model name. default is model.h5')
  parser.add_argument('--d',  help='set model directory')
  parser.add_argument('--e',  default='CartPole-v1', help='set gym environment')
  args = parser.parse_args()
 
  d = vars(args)
 
  # if dir is none, then treat as one
  ev = ModelEvaluator()
  ev.setup_env(d['e'])
 
  if ( not d['d']):
    print('doing model')
    model_name = d['m']
    print('model_name:',model_name)
    ev.read_model(model_name)
    ev.evaluate_model(10)
  else:
    print('doing dir')
    for file in os.listdir(d['d']):
      if file.endswith('.h5'):
        ev.read_model(d['d'] + '/' + file)
        #print('model is None?:',(ev.model == None))
        ev.evaluate_model(10)
 
 