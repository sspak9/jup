import numpy as np
#from tensorflow import keras
import keras
data = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = data.load_data()