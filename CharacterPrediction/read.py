import tensorflow as tf

class DataReader:
	def __init__(self,filename): 
		# first construct a queue containing a list of filenames.
		# this lets a user split up there dataset in multiple files to keep
		# size down
		filename_queue = tf.train.string_input_producer([filename],
														num_epochs=None)
		# Unlike the TFRecordWriter, the TFRecordReader is symbolic
		reader = tf.TFRecordReader()
		# One can read a single serialized example from a filename
		# serialized_example is a Tensor of type string.
		_, serialized_example = reader.read(filename_queue)
		# The serialized example is converted back to actual values.
		# One needs to describe the format of the objects to be returned
		features = tf.parse_single_example(
			serialized_example,
			features={
				# We know the length of both fields. If not the
				# tf.VarLenFeature could be used
				'input': tf.VarLenFeature(tf.int64),
				'target': tf.VarLenFeature(tf.int64),
				'seq_length': tf.FixedLenFeature([],tf.int64)
			})
		# now return the converted data
		input_indices = features['input']
		target_indices = features['target']
		seq_length = features['seq_length']

		input_batch, target_batch, seq_len_batch = tf.train.shuffle_batch(
			[input_indices, target_indices,seq_length], batch_size=32,
			capacity=2000,
			min_after_dequeue=1000)

		self.input_batch = input_batch
		self.target_batch = target_batch
		self.seq_length_batch = tf.cast(seq_len_batch,tf.int32)
		
	def get_ops(self):
		return self.input_batch,self.target_batch,self.seq_length_batch


