# Trying to define the simplest possible neural net where the output layer of the neural net is a single
# neuron with a "continuous" (a.k.a floating point) output.  I want the neural net to output a continuous
# value based off one or more continuous inputs.  My real problem is more complex, but this is the simplest
# representation of it for explaining my issue.  Even though I've oversimplified this to look like a simple
# linear regression problem (y=m*x), I want to apply this to more complex neural nets.  But if I can't get
# it working with this simple problem, then I won't get it working for anything more complex.

import tensorflow as tf
import random
import numpy as np

class Net:
    def __init__(self, INPUT_DIMENSION, OUTPUT_DIMENSION, TRAINING_RUNS, BATCH_SIZE, VERF_SIZE, Learning_rate):
        self.INPUT_DIMENSION = int(INPUT_DIMENSION)
        self.OUTPUT_DIMENSION = int(OUTPUT_DIMENSION)
        self.TRAINING_RUNS = TRAINING_RUNS
        self.BATCH_SIZE = BATCH_SIZE
        self.VERF_SIZE = VERF_SIZE
        self.batch_inputs = np.array([])
        self.batch_outputs = np.array([])
        self.n_hidden = 64
        self.n_hidden2 = 128

        # Create a placeholder for our input variable
        self.x = tf.placeholder(tf.float32, [None, self.INPUT_DIMENSION])

        # Create variables for our neural net hidden weights and bias

        self.W_h = tf.Variable(tf.truncated_normal([int(self.INPUT_DIMENSION),int(self.n_hidden)], stddev=0.01)) 
        self.b_h = tf.Variable(tf.truncated_normal([self.n_hidden], stddev=0.01))

        self.W_h2 = tf.Variable(tf.truncated_normal([int(self.n_hidden),int(self.n_hidden2)], stddev=0.01)) 
        self.b_h2 = tf.Variable(tf.truncated_normal([self.n_hidden2], stddev=0.01))



#        self.W = tf.Variable(tf.truncated_normal(shape=([self.INPUT_DIMENSION, self.OUTPUT_DIMENSION]), stddev=0.1))
        self.W = tf.Variable(tf.truncated_normal(shape=([self.n_hidden2, self.OUTPUT_DIMENSION]), stddev=0.01))
        self.b = tf.Variable(tf.truncated_normal(shape=([self.OUTPUT_DIMENSION]), stddev=0.01))


        # hidden layer
        self.h = tf.nn.tanh(tf.matmul(self.x, self.W_h)+self.b_h)
        self.h2 = tf.nn.tanh(tf.matmul(self.h, self.W_h2)+self.b_h2)
        
        # Define the neural net.  Note that since I'm not trying to classify digits as in the tensorflow mnist
        # tutorial, I have removed the softmax op.  My expectation is that 'net' will return a floating point
        # value.
        self.net = tf.matmul(self.h2, self.W) + self.b
#        self.net = tf.matmul(self.x, self.W) + self.b

        # Create a placeholder for the expected result during TRAINING_RUNS 
        self.expected = tf.placeholder(tf.float32, [None, self.OUTPUT_DIMENSION])

        # Same training as used in mnist example
        self.loss = tf.reduce_mean(tf.square(self.expected - self.net))
        # cross_entropy = -tf.reduce_sum(expected*tf.log(tf.clip_by_value(net,1e-10,1.0)))
        self.train_step = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss)

        self.sess = tf.Session()

        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

#INPUT_DIMENSION = 10
#OUTPUT_DIMENSION = 5
#TRAINING_RUNS = 100
#BATCH_SIZE = 10000
#VERF_SIZE = 1

    # Generate a bunch of data points and then package them up in the array format needed by
    # tensorflow
    def set_batch_data(self, t_data, targets):
        self.batch_inputs = t_data
        self.batch_outputs = targets

    def train(self):
        # Perform our training runs

        for i in range(self.TRAINING_RUNS):
            print("trainin run: ", i, )

            #batch_inputs, batch_outputs = generate_batch_data(self.BATCH_SIZE)
            batch_inputs= self.batch_inputs
            batch_outputs = self.batch_outputs


            # I've found that my weights and bias values are always zero after training, and I'm not sure why.
            self.sess.run(self.train_step, feed_dict={self.x: batch_inputs, self.expected: batch_outputs})

            # Test our accuracy as we train...  I am defining my accuracy as the error between what I
            # expected and the actual output of the neural net.
            # accuracy = tf.reduce_mean(tf.sub( expected, net))
            self.accuracy = tf.subtract(self.expected, self.net)  # using just subtract since I made my verification size 1 for debug
            # tf.subtract()
            # Uncomment this to debug
            # import pdb; pdb.set_trace()
            
            #print("W=%f, b=%f" % (sess.run(W), sess.run(b)))



            #batch_inputs, batch_outputs = generate_batch_data(self.VERF_SIZE)   ################
            

            batch_inputs=self.batch_inputs[10:11]
            batch_outputs = self.batch_outputs[10:11]



            actual = self.sess.run(self.net, feed_dict={self.x: batch_inputs})
            result = self.sess.run(self.accuracy, feed_dict={self.x: batch_inputs, self.expected: batch_outputs})

            print("    progress: ")
            print("      inputs: ", batch_inputs)
            print("      outputs:", batch_outputs)
            print("      actual: ", actual) 
            print("      error: ", result)

    def predict(self, batch_inputs):
 
        actual = self.sess.run(self.net, feed_dict={self.x: batch_inputs})
        
        return(actual)




# Generate two arrays, the first array being the inputs that need trained on, and the second array containing outputs.
def generate_test_point():
    x = random.uniform(-8, 8)

    # To keep it simple, output is just -x.
    out = -x

    return (np.array([x]), np.array([out]))


# Generate a bunch of data points and then package them up in the array format needed by
# tensorflow
def generate_batch_data(num):
    xs = []
    ys = []

    for i in range(num):
        x, y = generate_test_point()

        xs.append(x)
        ys.append(y)

    return (np.array(xs), np.array(ys))


#if __name__ == '__main__':
#    N = Net(1,1,100,10000,1)
#    batch_inputs, batch_outputs = generate_batch_data(10000)
#    N.set_batch_data(batch_inputs,batch_outputs)
#    N.train()
