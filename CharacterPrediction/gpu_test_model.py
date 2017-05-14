import tensorflow as tf 
import tensorflow.contrib.slim as slim

class GPUTestModel:
	def __init__(self,input_indices,target_indices,seq_length):
		self.s = tf.reduce_sum(input_indices)+tf.reduce_sum(target_indices)+tf.reduce_sum(seq_length)
	def train_step(self):
		return self.s,self.s+1.0