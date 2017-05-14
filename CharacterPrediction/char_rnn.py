from read import DataReader
from read_legacy import LegacyDataReader
import tensorflow.contrib.slim as slim
import tensorflow as tf 
from model import Model
from time import time
import sys
import numpy as np
import cPickle

with open("data/char_dict.pickle","rb") as f:
	char_dict = cPickle.load(f)
data_reader = LegacyDataReader("data/train.pickle")

inv_char_dict = {v:k for k,v in char_dict.iteritems()}
#data_reader = DataReader("data/train.tfrecords")
#x,y,seqlen = data_reader.get_ops()

input_placeholder = tf.placeholder(tf.int64,[None,None])
target_placeholder = tf.placeholder(tf.int64,[None,None])
seqlen_placeholder = tf.placeholder(tf.int64,[None])
test_input_placeholder = tf.placeholder(tf.int32,[None])
test_length_placeholder = tf.placeholder(tf.int32,shape=())


print len(char_dict)
 # In our dataset, we found that we had 325 unique characters.
mdl = Model(input_placeholder,target_placeholder,seqlen_placeholder,
	vocab_size=len(char_dict),embedding_size=64,rnn_size=256,rnn_layers=2,
	test_input=test_input_placeholder,test_length=test_length_placeholder)
#mdl = Model(x,y,seqlen,vocab_size=325,embedding_size=16,rnn_size=128)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

loss,train_step = mdl.train_step()

t1 = time()
train_loss = 0.0
train_count = 1e-10
for epoch in range(100):
	for i,(x,y,length) in enumerate(data_reader.get_batches(32)):
		

		if i % 10 == 0:
			feed_dict = {test_input_placeholder:[char_dict[u"H"]],test_length_placeholder:200}
			text = sess.run(mdl.test_letters,feed_dict)
			text = [inv_char_dict[t] for t in text]
			print "==== Training loss ",train_loss/train_count," time elapsed",time()-t1," ==== "
			print "Random text:"
			print u"".join(text)

			
			print "\nPredictions on real text:"
			prediction_ids = sess.run(mdl.output_ids,feed_dict={input_placeholder:x,target_placeholder:y,seqlen_placeholder:length})

			print "Target="
			print u"".join([inv_char_dict[t] for t in y[0][:length[0]]])
			print "Prediction="
			print u"".join([inv_char_dict[t] for t in prediction_ids[0][:length[0]]])
			print "==============="
			t1 = time()
			train_loss = 0.0
			train_count = 1e-10

	
		feed_dict = {input_placeholder:x,target_placeholder:y,seqlen_placeholder:length}

		_,L = sess.run([train_step,loss],feed_dict)
		train_loss += L
		train_count += 1.0
		
	