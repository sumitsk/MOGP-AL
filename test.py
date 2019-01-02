import numpy as np
from models import HadamardMTGP
from utils import load_data, zero_mean_unit_variance, compute_rmse, compute_mae
import ipdb

# data 
train_fn = 'data/jura_train_dataframe.pkl'
test_fn = 'data/jura_val_dataframe.pkl'
input_features = ['xloc', 'yloc']
target_features = ['Cd', 'Zn', 'Ni']
train_x, train_y, test_x, test_y = load_data(train_fn, test_fn, input_features, target_features)
# in active learning paper, they took log of Cd and Zn measurements and normalized also
# in GPRN paper, they did log transformation and zero mean unit variance normalization acroos each feature

# zero mean normalization
# train_y = zero_mean_unit_variance(train_y, std=1)
# test_y = zero_mean_unit_variance(test_y, std=1)

# predict Cd at test locations given Cd, Ni and Zn at training locations and Ni and Zn at test locations 
num_tasks = len(target_features)

# repeat train_x and train_y for all target features
# train_ind = [0, 0, ..., 1, 1, ...]
ind1 = np.arange(num_tasks).reshape(1,-1).repeat(len(train_x), 0).flatten()
ind2 = np.arange(1,num_tasks).reshape(1,-1).repeat(len(test_x), 0).flatten()
x1 = np.repeat(train_x, num_tasks, 0)
x2 = np.repeat(test_x, num_tasks-1, 0)
y1 = train_y.flatten()
y2 = test_y[:,1:].flatten()

# effective training set 
tx = np.concatenate([x1, x2])
ty = np.concatenate([y1, y2])
tind = np.concatenate([ind1, ind2])

# effective test set
test_ind = np.full(len(test_x), 0)

kernel_params = {'type': 'rbf'}
gp = HadamardMTGP(num_tasks=num_tasks, lr=.1, max_iter=300, kernel_params=kernel_params)
gp.fit(tx, tind, ty, disp=True)
mu = gp.predict(test_x, test_ind)
mae = compute_mae(test_y[:,0], mu)
ipdb.set_trace()