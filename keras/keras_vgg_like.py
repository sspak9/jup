#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
#from tensorflow import keras
import keras

print('keras version:', keras.__version__)
print('keras backend:', keras.backend.backend())
print('keras image format:', keras.backend.image_data_format())


# In[50]:


#load data
# chose one and check out the accuracy

#data = keras.datasets.mnist
#data = keras.datasets.fashion_mnist
data = keras.datasets.cifar10


# In[51]:


(x_train, y_train), (x_test, y_test) = data.load_data()


# In[52]:


print('train shape:', x_train.shape)
print('train y:', y_train.shape)
print('test_shape:', x_test.shape)
print('test y:', y_test.shape)


num_labels = len(np.unique(y_train))
image_size = x_train.shape[2]

print('num of labels:', num_labels)
print('image size:', image_size)

# calculate input shape and number of channels
is_channels_first = (keras.backend.image_data_format() == 'channels_first')
shape_len = len(x_train.shape)

if shape_len == 3:
    num_channels = 1
else:
    num_channels = 3

if is_channels_first:
    input_shape = (num_channels , image_szie , image_size)
else:
    input_shape = ( image_size , image_size , num_channels)

print('input shape:', input_shape)


# In[53]:


# convert the shape of data depending on the image data format

if is_channels_first :
    x_train2 = x_train.reshape(x_train.shape[0], num_channels, image_size, image_size)
    x_test2 = x_test.reshape(x_test.shape[0], num_channels, image_size, image_size)
else:
    x_train2 = x_train.reshape(x_train.shape[0], image_size, image_size, num_channels)
    x_test2 = x_test.reshape(x_test.shape[0], image_size, image_size, num_channels)


# In[54]:


# normalize the data: 0.0 to 1.0

x_train2 = x_train2.astype('float32') / 255
x_test2 = x_test2.astype('float32') / 255

#hot encode
y_train2 = keras.utils.to_categorical(y_train)
y_test2 = keras.utils.to_categorical(y_test)


# In[55]:


print("revised x_train shape:", x_train2.shape)
print('revised y_train shape:', y_train2.shape)
print('revised x_test shape:', x_test2.shape)
print('revised y_test shape:', y_test2.shape)
print('input shape:',input_shape)


# In[56]:


num_hidden_layers = 512
# for cifar10 dataset, include epochs to 100 or higher
epochs=200
batch_size=64

print('batch size:', batch_size)
print('epochs:', epochs)
print('hidden dense layer size:', num_hidden_layers)


# In[57]:


#function to copy 1 image to larger image map
def copy_image(target , ty, tx, src):
    for y in range(image_size):
        for x in range(image_size):
            target[ty*image_size+y][tx*image_size+x] = src[y][x]
    return target

def copy_image32(target , ty , tx , src):
    for y in range(image_size):
        for x in range(image_size):
            target[ty*image_size+y][tx*image_size+x][0] = src[y][x][0]
            target[ty*image_size+y][tx*image_size+x][1] = src[y][x][1]
            target[ty*image_size+y][tx*image_size+x][2] = src[y][x][2]
            
    return target

print('\n\ndisplaying few training samples')

# show 20 x 20
ysize = 20
xsize = 20
start_offset = 0
base_index = start_offset + (ysize * xsize)

if image_size == 28:
    image = np.zeros((image_size*ysize, image_size*xsize), dtype=np.int)
else:
    image = np.zeros( (image_size*ysize , image_size*xsize , 3), dtype=np.int)

for y in range(ysize):
    for x in range(xsize):
        index = y*xsize + x
        src = x_train[index + base_index]
        if image_size == 28:
            image = copy_image(image , y ,x , src)
        else:
            image = copy_image32( image , y, x, src)

from matplotlib import pyplot as plt

plt.figure(figsize=(7,7))
plt.grid(False)
plt.xticks([])
plt.yticks([])
if image_size == 28:
    plt.imshow(image , cmap='gray_r')
else:
    plt.imshow(image)
plt.show()
plt.close()


# In[58]:
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
                               
                               
import os
import time
from keras.callbacks import LearningRateScheduler

# helper function to create unique sub folder
def create_folder(folder_name):
    if (not os.path.exists(folder_name)):
        os.makedirs(folder_name)
    new_dir = folder_name + "/{}".format(time.time())
    if (not os.path.exists(new_dir)):
        os.makedirs(new_dir)
    return new_dir
    
save_folder = 'saved_models'
save_dir = create_folder(save_folder)

checkpt_path=save_dir+'/va{val_acc:.5f}-ep{epoch:04d}-ac{acc:.5f}-vl{val_loss:.5f}-l{loss:.5f}.hdf5'
cp_callback = keras.callbacks.ModelCheckpoint(
  checkpt_path ,
  verbose=1
)

from keras.optimizers import Adam

#model
model = keras.models.Sequential()
model.add( keras.layers.Conv2D(64, kernel_size=(3,3), input_shape=input_shape , activation='relu' , padding='same'))
model.add( keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu' , padding='same'))
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add( keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu' , padding='same'))
model.add( keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu' , padding='same'))
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add( keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu' , padding='same'))
model.add( keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu' , padding='same'))
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add( keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu' , padding='same'))
model.add( keras.layers.Conv2D(512, kernel_size=(3,3), activation='relu' , padding='same'))
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add( keras.layers.Flatten())
model.add( keras.layers.Dense(512, activation='relu'))
model.add( keras.layers.Dropout(0.5))
model.add( keras.layers.Dense(10, activation='softmax'))

# compile to model
model.compile(optimizer=Adam(lr=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# show summary
model.summary()


# In[59]:


# data generator
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.025,
    horizontal_flip=True
)


# In[60]:


#train the model with train data

fit_history = model.fit_generator(datagen.flow(x_train2, y_train2,
                                    batch_size=batch_size),
                                    use_multiprocessing=True,
                                    epochs=epochs,
                                    validation_data=(x_test2, y_test2),
                                  	workers=4,
                                  	callbacks=[lr_scheduler , lr_reducer]
                                 )
'''
fit_history = model.fit(x_train2, y_train2,
  epochs=epochs ,
  batch_size=batch_size,
  validation_data=(x_test2,y_test2)
)
'''


# In[ ]:


from matplotlib import pyplot as plt

# show procession of training...
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(fit_history.history['acc'])
plt.plot(fit_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
plt.close()


# In[ ]:




