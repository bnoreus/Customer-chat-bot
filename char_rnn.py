from read import DataReader
import tensorflow as tf 
from model import Model


data_reader = DataReader("data/train.tfrecords")
x,y,seqlen = data_reader.get_ops()
mdl = Model(x,y,seqlen,vocab_size=325,embedding_size=16,lstm_size=128) # In our dataset, we found that we had 325 unique characters.

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

