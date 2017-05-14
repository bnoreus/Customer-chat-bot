import tensorflow as tf 
import tensorflow.contrib.slim as slim
from tensorflow.python.layers.core import Dense


class Model:
	def __init__(self,input_indices,target_indices,seq_length,**config):
		vocab_size = config["vocab_size"]
		embedding_size = config["embedding_size"]
		lstm_size = config["rnn_size"]
		rnn_layers = config["rnn_layers"]
		embedding_matrix = tf.get_variable("embedding_matrix",[vocab_size,embedding_size],dtype=tf.float32)
		embedding_layer = tf.nn.embedding_lookup(embedding_matrix,input_indices)
		
		cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(rnn_layers)])

		rnn_outputs,state = tf.nn.dynamic_rnn(cell,embedding_layer,sequence_length=seq_length,dtype=tf.float32)
		

		#softmax_w = tf.get_variable("softmax_w",[lstm_size,vocab_size])
		#softmax_b = tf.get_variable("softmax_b",[vocab_size])

		dense_layer = Dense(vocab_size,name="output_layer")

		outputs = tf.map_fn(lambda x: dense_layer(x),rnn_outputs)
		self.outputs = outputs
		self.output_ids = tf.argmax(self.outputs,2)
		self.softmax_loss = tf.contrib.seq2seq.sequence_loss(outputs,target_indices,tf.sequence_mask(seq_length,dtype=tf.float32))
		optimizer = tf.train.AdamOptimizer(0.002)

		self.train_op = optimizer.minimize(self.softmax_loss)


		if "test_input" in config:
			
			tf.get_variable_scope().reuse_variables()

			test_input = config["test_input"]
			test_length = config["test_length"]
			inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_matrix,test_input,0)

			inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell,inference_helper,cell.zero_state(1,tf.float32),dense_layer)

			_  = tf.contrib.seq2seq.dynamic_decode(inference_decoder,output_time_major=False,
				impute_finished=True,maximum_iterations=test_length,scope="rnn")
			inference_logits = _[0]

			self.test_letters = tf.squeeze(inference_logits.sample_id)
			for variable in tf.trainable_variables():
				print variable
	def train_step(self):
		return self.softmax_loss,self.train_op