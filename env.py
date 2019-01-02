import numpy as np
from utils import load_data, zero_mean_unit_variance, vec_to_one_hot_matrix


class JuraEnv(object):
	def __init__(self):
		train_fn = 'data/jura_train_dataframe.pkl'
		test_fn = 'data/jura_val_dataframe.pkl'
		
		input_features = ['xloc', 'yloc']
		# input_features = ['xloc', 'yloc', 'landuse', 'rock']
		output_features = ['Cd', 'Ni', 'Zn']
		train_x, train_y, test_x, test_y = load_data(train_fn, test_fn, input_features, output_features)

		if len(input_features)==4:
			lu_train = vec_to_one_hot_matrix(train_x[:,2].astype(int)-1)
			ro_train = vec_to_one_hot_matrix(train_x[:,3].astype(int)-1)
			train_x = np.concatenate([train_x[:,:2], lu_train, ro_train], axis=1)
			lu_test = vec_to_one_hot_matrix(test_x[:,2].astype(int)-1)
			ro_test = vec_to_one_hot_matrix(test_x[:,3].astype(int)-1)
			test_x = np.concatenate([test_x[:,:2], lu_test, ro_test], axis=1)

		# arrange all data by category
		# predict Cd at test locations given Cd, Ni and Zn at training locations and Ni and Zn at test locations 
		self.num_tasks = len(output_features)
		target_feature = 'Cd'
		self.target_task = output_features.index(target_feature)
		others = list(range(self.num_tasks))
		others.remove(self.target_task)

		# normalize data
		# take log of Cd and Zn measurements
		correct_skewness = ['Cd', 'Zn']
		for f in correct_skewness: 
			idx = output_features.index(f)
			train_y[:, idx] = np.log(train_y[:, idx])
			test_y[:, idx] = np.log(test_y[:, idx])

		# zero mean unit variance normalization
		# in MOAL, this is done across train and test jointly
		all_y = zero_mean_unit_variance(np.concatenate([train_y, test_y], 0)) 
		train_y = all_y[:len(train_x)]
		test_y = all_y[len(train_x):]

		# test set consists of target_task at test locations
		# training set consists of rest all
		# train_ind = [0, 0, ..., 1, 1, ...]
		ind1 = np.arange(self.num_tasks).reshape(1,-1).repeat(len(train_x), 0).flatten()
		ind2 = np.array(others).reshape(1,-1).repeat(len(test_x), 0).flatten()
		x1 = np.repeat(train_x, self.num_tasks, 0)
		x2 = np.repeat(test_x, len(others), 0)
		y1 = train_y.flatten()
		y2 = test_y[:,others].flatten()

		# effective training set 
		self.X = np.concatenate([x1, x2])
		self.Y = np.concatenate([y1, y2])
		self.ind = np.concatenate([ind1, ind2])

		# effective test set
		self.test_X = test_x
		self.test_Y = test_y[:,self.target_task] 
		self.test_ind = np.full(len(test_x), self.target_task)

		# normalize X between -1 and 1
		# x = np.concatenate([self.X, self.test_X])
		# x = 2*((x-x.min(0))/x.max(0)) - 1
		# self.X = x[:len(self.Y), :]
		# self.test_X = x[len(self.Y):, :]
		
	@property
	def num_samples(self):
		return len(self.X)

