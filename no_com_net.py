####################
# Neural net, simple layer net
####################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
            layers.Dense(1)
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
