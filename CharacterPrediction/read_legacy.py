import cPickle
import random
import numpy as np

class LegacyDataReader:
	def __init__(self,filename): 
		with open(filename,"rb") as f:
			self.data = cPickle.load(f)
			self.buckets_boundaries = np.array([0,50,150,200,250,300,350,400,450,500])
			self.bucket = [[] for _ in range(len(self.buckets_boundaries))]


	def get_batches(self,batch_size):
		random.shuffle(self.data)

		for x,y,length in self.data:
			if length > 500:
				length = 500
				x = x[:500]
				y = y[:500]
			idx = np.sum(self.buckets_boundaries < length) - 1
			self.bucket[idx].append((x,y,length))
			if len(self.bucket[idx]) == batch_size:
				data_batch = self.bucket[idx]
				x_batch = [_[0] for _ in data_batch]
				y_batch = [_[1] for _ in data_batch]
				len_batch = [_[2] for _ in data_batch]
				yield self.to_dense(x_batch,y_batch,len_batch)
				self.bucket[idx] = []
		print "Flushing out residuals"
		for data_batch in self.bucket:
			x_batch = [_[0] for _ in data_batch]
			y_batch = [_[1] for _ in data_batch]
			len_batch = [_[2] for _ in data_batch]
			if len(data_batch) > 0:
				yield self.to_dense(x_batch,y_batch,len_batch)

	def to_dense(self,x,y,length):
		maxlen = max(length)
		batch_size = len(x)

		x_dense = np.zeros((batch_size,maxlen)).astype(np.int64)
		y_dense = np.zeros((batch_size,maxlen)).astype(np.int64)

		for example_idx in range(len(x)):
			for data_idx,data in enumerate(x[example_idx]):
				x_dense[example_idx][data_idx] = data

		for example_idx in range(len(y)):
			for data_idx,data in enumerate(y[example_idx]):
				y_dense[example_idx][data_idx] = data

		return x_dense,y_dense,length