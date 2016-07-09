import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np

num_steps = 5
num_units = 2
np_input_data = [np.array([[1.],[2.]]), np.array([[2.],[3.]]), np.array([[3.],[4.]]), np.array([[4.],[5.]]), np.array([[5.],[6.]])]

batch_size = 2

graph = tf.Graph()

with graph.as_default():
    tf_inputs = [tf.placeholder(tf.float32, [batch_size, 1]) for _ in range(num_steps)]

    output_weights = tf.Variable(tf.truncated_normal([num_units, 1], stddev=0.1))
    output_biases = tf.Variable(tf.constant(1.0, shape=[1]))

    lstm = rnn_cell.BasicLSTMCell(num_units)
    initial_state = state = tf.zeros([batch_size, lstm.state_size])
    loss = 0

    with tf.variable_scope("myrnn") as scope:
        for i in range(num_steps-1):
            if i > 0:
                scope.reuse_variables()
            output, state = lstm(tf_inputs[i], state)

            output_z = tf.matmul(output, output_weights)

            loss += tf.reduce_mean(tf.square(output_z - tf_inputs[i+1]))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

num_epochs = 1000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for epoch in range(num_epochs):
        feed_dict={tf_inputs[i]: np_input_data[i] for i in range(len(np_input_data))}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        print(l)