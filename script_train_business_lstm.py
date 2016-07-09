import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import batchgenerator as bg
import numpy as np
import cPickle
import random

num_unrollings = 7
num_features = 35

num_lstm_units = 256
batch_size = 100

graph = tf.Graph()

#--------------Useful functions --------------

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, num_features])
  return b/np.sum(b, 1)[:,None]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction, temperature = 0.6):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, num_features], dtype=np.float)
  t_prediction = np.log(prediction[0]) / temperature
  t_prediction = np.exp(t_prediction) / np.sum(np.exp(t_prediction))
  p[0, sample_distribution(t_prediction)] = 1.0
  #p[0, np.argmax(t_prediction)] = 1.0
  return p

def characters(probabilities, generator):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  #print probabilities.shape, np.argmax(probabilities, 1)
  return [generator.id2char(c) for c in np.argmax(probabilities, 1)]

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

#------------Graph------------------------------

with graph.as_default():
    tf_inputs = [tf.placeholder(tf.float32, [batch_size, num_features]) for _ in range(num_unrollings)]

    tf_cv = [tf.placeholder(tf.float32, [batch_size, num_features]) for _ in range(num_unrollings)]

    lstm = rnn_cell.BasicLSTMCell(num_lstm_units)

    #lstm_init_input = tf.zeros([batch_size, num_features])
    #state = tf.zeros([batch_size, lstm.state_size])
    #lstm(lstm_init_input, state)
    #lstm_output =  tf.zeros([batch_size, num_lstm_units])
    #output_z = tf.zeros([batch_size, num_features])

    saved_output = tf.Variable(tf.zeros([batch_size, num_lstm_units]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, lstm.state_size]), trainable=False)

    output_weights = tf.Variable(tf.truncated_normal([num_lstm_units, num_features], stddev=0.1))
    output_biases = tf.Variable(tf.constant(1.0, shape=[num_features]))

    output = saved_output
    state = saved_state

    train_prediction = [tf.placeholder(tf.float32, [batch_size, num_features]) for _ in range(num_unrollings-1)]

    loss = 0

    with tf.variable_scope('myrnn') as scope:
        for i in range(num_unrollings-1):
            if i > 0:
                scope.reuse_variables()
            lstm_output, state = lstm(tf_inputs[i], state)

            output_z = tf.matmul(tf.nn.relu(lstm_output), output_weights) + output_biases
            loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_z, tf_inputs[i + 1]))
            train_prediction[i] = tf.nn.softmax(output_z)

    #with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
    # Classifier.
    #    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), output_weights, output_biases)
    #    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, tf_inputs[1:])))

    #train_prediction = tf.nn.softmax(logits)

    loss = loss / num_unrollings

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 10000, 0.1, staircase=True)

    learning_rate = tf.train.exponential_decay(.01, global_step, 5000, 0.5, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #gradients, v = zip(*optimizer.compute_gradients(loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    #optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    sample_input = tf.placeholder(tf.float32, shape=[1, num_features])
    sample_reset_flag = tf.placeholder(tf.float32)
    saved_sample_output = tf.Variable(tf.zeros([1, num_lstm_units]))
    saved_sample_state = tf.Variable(tf.zeros([1, lstm.state_size]))

    with tf.variable_scope('myrnn_scope') as sample_scope:
        if sample_reset_flag == 0:
            sample_scope.reuse_variables()
        sample_lstm_output, sample_state = lstm(sample_input, saved_sample_state)
        sample_output_z = tf.matmul(tf.nn.relu(sample_lstm_output), output_weights) + output_biases
        sample_prediction = tf.nn.softmax(sample_output_z)

    saver = tf.train.Saver()

#---------------Data loader and params----------------------------------

[chars, text] = cPickle.load(open('chars_text_declaration.p', 'r'))

generator = bg.BatchGenerator(text=text, vocabulary=chars, num_unrollings=num_unrollings, batch_size=batch_size)

num_epochs = 1000000
cv_period = 1000
loss_check_period = 100
autosave_period = 1000

autosave_file = "autosave.ckpt"

#---------------Session------------------------------

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    #saver.restore(session, autosave_file)

    mean_loss = 0

    for epoch in range(num_epochs):
        batch = np.array(generator.next())
        feed_dict={tf_inputs[i]: batch[i,:,:] for i in range(num_unrollings)}
        _, l, rate = session.run([optimizer, loss, learning_rate], feed_dict=feed_dict)
        mean_loss += l

        if epoch % loss_check_period == 0:
            mean_loss = mean_loss / loss_check_period
            labels = np.concatenate(list(batch)[1:])
            print('-' * 80)
            print 'Iter', epoch, 'mean loss:', mean_loss, 'learning rate:', rate
            predictions = session.run(train_prediction, feed_dict=feed_dict)
            print 'Original/predicted sequence:'
            original_string = ''
            for count in range(num_unrollings):
               original_string +=  generator.characters(batch[count,:,:])[0]
            print original_string
            #print 'Predicted sequence:'
            predicted_string = ' '
            for count in range(num_unrollings-1):
                predicted_string += generator.characters(predictions[count])[0]
            print predicted_string

        if epoch % cv_period == 0:
            print('=' * 80)

            #feed = sample(random_distribution())
            #sentence = generator.characters(feed)[0]
            #print 'initial sentence:'
            #print sentence
            #reset_sample_state.run()
            #prediction = sample_prediction.eval({sample_input: feed})
            #print prediction
            #feed = sample(prediction)

            #print 'feed:'
            #print feed

            #print 'predicted char:'
            #print generator.characters(feed)[0]
            #sentence += generator.characters(feed)[0]
            #print 'new sentence:'
            #print sentence


            for _ in range(5):
                feed = sample(random_distribution())
                sentence = generator.characters(feed)[0]
                prediction = sample_prediction.eval({sample_input:feed, sample_reset_flag:1})
                for _ in range(79):
                    feed = sample(prediction)
                    sentence += generator.characters(feed)[0]
                    prediction = sample_prediction.eval({sample_input: feed, sample_reset_flag:0})
                print(sentence)
            #print prediction.shape
            print('=' * 80)

        if epoch % autosave_period == 0: saver.save(session, autosave_file)
