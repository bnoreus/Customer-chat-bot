import tensorflow as tf 
import tensorflow.contrib.slim as 

class Model:
	def __init__(self,input_indices,output_indices,seq_length,**config):
		vocab_size = config["vocab_size"]
		embedding_size = config["embedding_size"]
		lstm_size = config["rnn_size"]
		embedding_matrix = tf.get_variable("embedding_matrix",[vocab_size,embedding_size],dtype=tf.float32)
		embedding_layer = tf.nn.embedding_lookup(embedding_matrix,input_indices)
		print embedding_layer.get_shape()
		cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		rnn_outputs,state = tf.nn.dynamic_rnn(cell,embedding_layer,sequence_length=seq_length,dtype=tf.float32)

		print rnn_outputs.get_shape()
		softmax_w = tf.get_variable("softmax_w",[vocab_size,lstm_size])
		softmax_b = tf.get_variable("softmax_b",[vocab_size])

		outputs = tf.map_fn(lambda x: tf.matmul(x,softmax_w)+softmax_b,rnn_outputs)
		print outputs.get_shape()

		softmax_loss = tf.contrib.seq2seq.sequence_loss(self.output,self.input,tf.sequence_mask(seq_length,dtype=tf.float32))

		