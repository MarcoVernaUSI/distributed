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
    def __init__(self, INPUT_DIMENSION, OUTPUT_DIMENSION, TRAINING_RUNS, BATCH_SIZE, VERF_SIZE):
        self.INPUT_DIMENSION = INPUT_DIMENSION
        self.OUTPUT_DIMENSION = OUTPUT_DIMENSION
        self.TRAINING_RUNS = TRAINING_RUNS
        self.BATCH_SIZE = BATCH_SIZE
        self.VERF_SIZE = VERF_SIZE
        self.batch_inputs = np.array([])
        self.batch_outputs = np.array([])

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
        # Define a single-layer neural net.  Originally based off the tensorflow mnist for beginners tutorial

        # Create a placeholder for our input variable
        x = tf.placeholder(tf.float32, [None, self.INPUT_DIMENSION])

        # Create variables for our neural net weights and bias
        W = tf.Variable(tf.zeros([self.INPUT_DIMENSION, self.OUTPUT_DIMENSION]))
        b = tf.Variable(tf.zeros([self.OUTPUT_DIMENSION]))

        # Define the neural net.  Note that since I'm not trying to classify digits as in the tensorflow mnist
        # tutorial, I have removed the softmax op.  My expectation is that 'net' will return a floating point
        # value.
        net = tf.matmul(x, W) + b

        # Create a placeholder for the expected result during TRAINING_RUNS 
        expected = tf.placeholder(tf.float32, [None, self.OUTPUT_DIMENSION])

        # Same training as used in mnist example
        loss = tf.reduce_mean(tf.square(expected - net))
        # cross_entropy = -tf.reduce_sum(expected*tf.log(tf.clip_by_value(net,1e-10,1.0)))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        sess = tf.Session()

        init = tf.initialize_all_variables()
        sess.run(init)

        # Perform our training runs

        for i in range(self.TRAINING_RUNS):
            print("trainin run: ", i, )

            #batch_inputs, batch_outputs = generate_batch_data(self.BATCH_SIZE)
            batch_inputs= self.batch_inputs
            batch_outputs = self.batch_outputs


            # I've found that my weights and bias values are always zero after training, and I'm not sure why.
            sess.run(train_step, feed_dict={x: batch_inputs, expected: batch_outputs})

            # Test our accuracy as we train...  I am defining my accuracy as the error between what I
            # expected and the actual output of the neural net.
            # accuracy = tf.reduce_mean(tf.sub( expected, net))
            accuracy = tf.subtract(expected, net)  # using just subtract since I made my verification size 1 for debug
            # tf.subtract()
            # Uncomment this to debug
            # import pdb; pdb.set_trace()
            
            #print("W=%f, b=%f" % (sess.run(W), sess.run(b)))

            #batch_inputs, batch_outputs = generate_batch_data(self.VERF_SIZE)   ################
            batch_inputs=self.batch_inputs[10:11]
            batch_outputs = self.batch_outputs[10:11]



            actual = sess.run(net, feed_dict={x: batch_inputs})
            result = sess.run(accuracy, feed_dict={x: batch_inputs, expected: batch_outputs})

            print("    progress: ")
            print("      inputs: ", batch_inputs)
            print("      outputs:", batch_outputs)
            print("      actual: ", actual) 
            print("      error: ", result)


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
