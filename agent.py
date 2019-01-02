import numpy as np
import torch
from utils import compute_mae, entropy_from_var
from models import HadamardMTGP


class Master(object):
    # this class is used for estimating hyperparameters of the GP model from the training dataset
    def __init__(self, env, args, load_fn=None):
        self.env = env
        self.lr = args.lr
        self.max_iter = args.max_iterations
        self.cov_module = args.cov_module
        self.rank = args.rank

        self.gp = self.init_model()
        if load_fn is None:
            # fit model on the entire training set
            self.gp.fit(self.env.X, self.env.ind, self.env.Y, disp=False)
        else:
            self.gp.reset(self.env.X, self.env.ind, self.env.Y, train_mode=False)
            state_dict = torch.load(load_fn)
            self.gp.model.load_state_dict(state_dict['state_dict'])

    def init_model(self):
        cov_module_params = {'type': self.cov_module}
        if self.cov_module == 'sm':
            cov_module_params['num_mixtures'] = 3 
        gp = HadamardMTGP(num_tasks=self.env.num_tasks, rank=self.rank, lr=self.lr, max_iter=self.max_iter,
                          covar_module=cov_module_params)
        return gp

    def save_model(self, filename):
        state = {'state_dict': self.gp.model.state_dict()}
        torch.save(state, filename)

class Agent(object):
    def __init__(self, master):
        self.env = master.env
        self.gp = master.init_model()
        self.sec_gp = master.init_model()
        self.model_hyperparameters = master.gp.model.state_dict()
        self.sampled = np.full(self.env.num_samples, False)
        
        # flags set to True when model hyperparameters are copied from master
        self.gp_loaded = False
        self.sec_gp_loaded = False

        # In the beginning, only the secondary GP's train data is set
        self.update_model(gp='secondary')
        
    def update_model(self, gp='main'):
        if gp == 'main':
            x = self.env.X[self.sampled]
            y = self.env.Y[self.sampled]
            ind = self.env.ind[self.sampled]
            self.gp.reset(x, ind, y, train_mode=False)
            # load model hyperparameters only once
            if not self.gp_loaded:
                self.gp.model.load_state_dict(self.model_hyperparameters)
                self.gp_loaded = True

        elif gp == 'secondary':
            # remaining type t samples (V_{t}\X_{t})
            type_t_rem = (self.env.ind == self.env.target_task) * ~self.sampled 
            # X \cup V_{t}\X_{t} 
            relevant = type_t_rem + self.sampled 

            x = self.env.X[relevant] 
            y = self.env.Y[relevant] 
            ind = self.env.ind[relevant] 
            self.sec_gp.reset(x, ind, y, train_mode=False)
            # load model hyperparameters only once
            if not self.sec_gp_loaded:
                self.sec_gp.model.load_state_dict(self.model_hyperparameters)
                self.sec_gp_loaded = True 
 
        else:
            raise NotImplementedError('Unknown GP provided!')

    def learn(self, iterations, predict_every=5):
        # notations:
        # X - sampled set
        # X_{t} - sampled set of type t
        # V_{t} - entire sampling set of type t
        
        all_mae = []
        # select samples greedily with the highest conditional utility 
        for j in range(iterations):
            # print('Iteration {:d}/{:d}'.format(j+1, iterations))

            utilites = np.full(self.env.num_samples, -np.inf)            
            indices = np.arange(self.env.num_samples)

            # all remaining samples 
            rem = indices[~self.sampled]
            x = self.env.X[rem]
            t = self.env.ind[rem]
            mu, var = self.gp.predict(x, t, return_var=True)
            ent = entropy_from_var(var)

            # all remaining samples of type t
            rem_t = indices[~self.sampled*(self.env.ind==self.env.target_task)]
            x_t = self.env.X[rem_t]
            t_t = np.full(len(x_t), self.env.target_task)
            mu_t, var_t = self.sec_gp.predict(x_t, t_t, return_var=True)
            ent_t = entropy_from_var(var_t)

            utilites[rem] = ent
            utilites[rem_t] -= ent_t

            best_idx = np.argmax(utilites)
            self.sampled[best_idx] = True

            # modify train data of main gp
            self.update_model(gp='main')

            # modify train data of secondary gp if chosen type is different from target type
            if self.env.ind[best_idx] != self.env.target_task:
                self.update_model(gp='secondary')

            if (j+1)%predict_every == 0:
                mu = self.gp.predict(self.env.test_X, self.env.test_ind)
                mae = compute_mae(self.env.test_Y, mu)
                all_mae.append(mae)
                print('Iteration: {:d}/{:d} Test MAE: {:3f}'.format(j+1, iterations, mae))
        return all_mae

