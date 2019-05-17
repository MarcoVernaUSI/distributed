# Trying to define the simplest possible neural net where the output layer of the neural net is a single
# neuron with a "continuous" (a.k.a floating point) output.  I want the neural net to output a continuous
# value based off one or more continuous inputs.  My real problem is more complex, but this is the simplest
# representation of it for explaining my issue.  Even though I've oversimplified this to look like a simple
# linear regression problem (y=m*x), I want to apply this to more complex neural nets.  But if I can't get
# it working with this simple problem, then I won't get it working for anything more complex.

import tensorflow as tf
import random
import numpy as np

class RNet:
    def __init__(self, INPUT_DIMENSION, OUTPUT_DIMENSION, TRAINING_RUNS, BATCH_SIZE, VERF_SIZE, Learning_rate, number):
        self.INPUT_DIMENSION = INPUT_DIMENSION
        self.OUTPUT_DIMENSION = OUTPUT_DIMENSION
        self.TRAINING_RUNS = TRAINING_RUNS
        self.BATCH_SIZE = BATCH_SIZE
        self.VERF_SIZE = VERF_SIZE
        self.batch_inputs = np.array([])
        self.batch_outputs = np.array([])
        self.hidden_units = 64
#        self.n_hidden2 = 16

        with tf.variable_scope(number):
            # Create a placeholder for our input variable
            self.x = tf.placeholder(tf.float32, [None]+self.INPUT_DIMENSION)
            # Create a placeholder for the expected result during TRAINING_RUNS 
            self.expected = tf.placeholder(tf.float32, [None] + self.OUTPUT_DIMENSION)

            self.init_state = cells.zero_state(self.x.shape[0], dtype=tf.float32)
       
            rnn_output, self.final_state = self._lstm(self.x,'64', self.init_state, out_hidden_state=True,)

            rshape = rnn_output.shape

            rnn_output = tf.reshape(rnn_output, [-1, rshape[2]])

            # Create variables for our neural net weights and bias
    #        self.W = tf.Variable(tf.zeros([self.n_hidden, self.OUTPUT_DIMENSION]))
            

           # self.W = tf.Variable(tf.truncated_normal(shape=([rnn_output.shape[1], 128]), stddev=0.1))
        


            #self.b = tf.Variable(tf.truncated_normal(shape=[128], stddev=0.1))

            #self.hidden = tf.nn.tanh(tf.matmul(rnn_output, self.W) + self.b)


            ###########################



            self.W = tf.Variable(tf.truncated_normal(shape=([rnn_output.get_shape().as_list()[1], self.OUTPUT_DIMENSION[1]]), stddev=0.1))
        


            self.b = tf.Variable(tf.truncated_normal(shape=[self.OUTPUT_DIMENSION[1]], stddev=0.1))

            self.net = tf.matmul(rnn_output, self.W) + self.b

            self.expected_r = tf.reshape(self.expected,[-1]+[self.OUTPUT_DIMENSION[1]])



            # hidden layer
    #        self.h = tf.nn.relu(tf.matmul(self.x, self.W_h)+self.b_h)
    #        self.h2 = tf.nn.relu(tf.matmul(self.h, self.W_h2)+self.b_h2)
            
    #        self.h = tf.matmul(self.x, self.W_h)+self.b_h
            
            # Define the neural net.  Note that since I'm not trying to classify digits as in the tensorflow mnist
            # tutorial, I have removed the softmax op.  My expectation is that 'net' will return a floating point
            # value.
    #        self.net = tf.matmul(self.h, self.W) + self.b
           


            # Same training as used in mnist example
            self.loss = tf.reduce_mean(tf.square(self.expected_r - self.net))
            # cross_entropy = -tf.reduce_sum(expected*tf.log(tf.clip_by_value(net,1e-10,1.0)))
            self.train_step = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss)

            # Network prediction
            self.prediction = tf.multinomial(self.net, 1)

        self.sess = tf.Session()

        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

    def _lstm(self,in_tensor, num_hidden_units,  state, out_hidden_state=False,):
        num_hidden_units = [int(x) for x in num_hidden_units.split('.')]
        out_hidden_state = bool(int(out_hidden_state))

        #zero_init = bool(int(zero_init))
        cells = [tf.nn.rnn_cell.LSTMCell(num, name='LSTMCell-%d'%num) for num in num_hidden_units]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        #if zero_init:
        #    batch_size = in_tensor.shape[0]
        #    state = multi_cell.zero_state(batch_size, tf.float32)
        #    outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, dtype=tf.float32, initial_state=state)
        #else:
        outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, dtype=tf.float32, init_state = state)
        out_tensor = outputs_state[1][-1].c if out_hidden_state else outputs_state[0]

        return tf.identity(out_tensor, name='Output')



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
            import pdb; pdb.set_trace()  # breakpoint 6a89d2da //

            # Test our accuracy as we train...  I am defining my accuracy as the error between what I
            # expected and the actual output of the neural net.
            # accuracy = tf.reduce_mean(tf.sub( expected, net))
            self.accuracy = tf.subtract(self.expected_r, self.net)  # using just subtract since I made my verification size 1 for debug
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
       #     print("      inputs: ", batch_inputs)
       #     print("      outputs:", batch_outputs)
        #    print("      actual: ", actual)

         #   print("      error: ", result)

    def predict2(self, batch_inputs):
 
        actual = self.sess.run(self.net, feed_dict={self.x: batch_inputs})
        
        return(actual)

    def predict(self, batch_inputs, pred_state = None):
        #batch_size = in_tensor.shape[0]
        #state = multi_cell.zero_state(batch_size, tf.float32)
            
        feed = {self.x: batch_inputs, init_state = pred_state}
        _,actual,new_state = self.sess.run([self.net,self.prediction,self.final_state])
   
        return(actual,new_state)



  

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
