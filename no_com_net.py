####################
# Neural net, simple layer net
####################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Net:
    def __init__(self, input_size, epochs=100):
        self.input_size = input_size
        self.output_size = 1
        self.epochs = epochs
        
        self.model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.tanh, 
                kernel_initializer=tf.initializers.truncated_normal, 
                bias_initializer=tf.initializers.truncated_normal, 
                input_shape=(self.input_size,)),
            layers.Dense(output_size)
        ])

        self.optimizer = tf.keras.optimizers.Adam()

        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

    def train(self, data, targets):

        data = data.reshape((-1,self.input_size))
        targets = targets.reshape((-1,self.output_size))
        print('Training the model')
        self.early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        return self.model.fit(data, targets, epochs=self.epochs, validation_split=0.2, verbose=2,
                        callbacks=[self.early_stop])

    def predict(self, data):
        return self.model.predict(data)

    def prepare_data(self, net_inputs, net_outputs):
        t_data= np.array(net_inputs)
        targets= np.array(net_outputs)
        t_data=t_data[:,:,:,3:5]
        t_data = t_data.reshape(t_data.shape[0]*t_data.shape[1], -1)
        targets = targets.reshape(targets.shape[0]*targets.shape[1], -1)
        return t_data, targets
