import gym
import random
import numpy as np
import keras
#from tensorflow import keras

from collections import deque

# Double DQN Agent for simple openai gyms

class DoubleDQNAgent:
  def __init__(self, state_size, action_size,**kwargs):
        
    # get size of state and action
    self.state_size = state_size
    self.action_size = action_size

    self.discount_factor = kwargs.get('discount_factor' , 0.99)
    self.learning_rate = kwargs.get('learning_rate' , 0.001)
    self.epsilon = kwargs.get('epsilon' , 1.0)
    self.epsilon_decay = kwargs.get('epsilon_decay' , 0.999)
    self.epsilon_min = kwargs.get('epsilon_min' , 0.01)
    self.batch_size = kwargs.get('batch_size' , 64)
    self.train_start = kwargs.get('train_start' , 1000)
    self.layer1 = kwargs.get('layer1' , 32)
    self.layer2 = kwargs.get('layer2' , 16)
    mxlen = kwargs.get('memory' , 100000)
    self.memory = deque(maxlen=mxlen)

    # two models for training
    self.model = self.build_model()
    self.target_model = self.build_model()

    # sync target with model
    self.update_target_model()

 # model will be q function
  def build_model(self):
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(self.layer1, input_dim=self.state_size, activation='relu',
      kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(self.layer2, activation='relu',
      kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(self.action_size, activation='linear',
      kernel_initializer='he_uniform'))

    model.summary()
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
    
    return model

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def append_sample(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

  # epsilon-greedy policy
  def get_action(self, state):
    # use random moves using epsilon. The min is always EPSILON_MIN
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    else:
      q_value = self.model.predict(state)
    return np.argmax(q_value[0])

  def train_model(self):
    # don't train until memory is filled with some random moves
    if len(self.memory) < self.train_start:
      return
    
    # if memory not yet filled for replay, set to memory len
    batch_size = min(self.batch_size, len(self.memory))

    # select random sample from the memory
    # remember, the latest input is not really used
    mini_batch = random.sample(self.memory, batch_size)

    # copy the mini batch into the format for prediction
    update_input = np.zeros((batch_size, self.state_size))
    update_target = np.zeros((batch_size, self.state_size))
    action, reward, done = [], [], []

    # copy: state, action, reward, next_state, done
    for i in range(batch_size):
      update_input[i] = mini_batch[i][0]
      action.append(mini_batch[i][1])
      reward.append(mini_batch[i][2])
      update_target[i] = mini_batch[i][3]
      done.append(mini_batch[i][4])

    # predict
    target = self.model.predict(update_input)
    target_next = self.model.predict(update_target)
    target_val = self.target_model.predict(update_target)

    for i in range(self.batch_size):
      # like Q Learning, get maximum Q value at s'
      # But from target model
      if done[i]:
        target[i][action[i]] = reward[i]
      else:
        # the key point of Double DQN
        # selection of action is from model
        # update is from target model
        a = np.argmax(target_next[i])
        target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])

    # make minibatch which includes target q value and predicted q value
    # and do the model fit!
    self.model.fit(update_input, target, 
                  batch_size=self.batch_size,
                  epochs=1, verbose=0)

