import torch
import gpytorch
import numpy as np
from gpytorch.kernels import RBFKernel, MaternKernel, SpectralMixtureKernel, IndexKernel, GridInterpolationKernel, ProductStructureKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from utils import to_torch, entropy_from_cov


# Hadamard product Multi-task GP model
class HadamardMTGP(object):
    def __init__(self, num_tasks=2, lr=.05, max_iter=200, rank=1, covar_module=None):
        super(HadamardMTGP, self).__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.num_tasks = num_tasks
        self.rank = rank
        self.covar_module = covar_module
        self.model = None

    @property
    def train_x(self):
        self._train_x

    @property
    def train_ind(self):
        return self._train_ind
    
    @property
    def train_y(self):
        return self._train_y
    
    def _prep_train_data(self, x, y_ind, y):
        # prepare training data
        self._train_x = to_torch(x)
        self._train_y_ind = to_torch(y_ind).long()
        train_y = to_torch(y)
        
        # single mean estimate across all categories
        self._train_y_mean = train_y.mean()
        self._train_y = train_y - self._train_y_mean
        
        # category-wise mean
        # BUG: if in train data, there are no samples of a type t, then the corresponding type mean is nan.
        # self._train_y_mean = torch.stack([torch.mean(train_y[self._train_y_ind==i]) for i in range(self.num_tasks)])
        # self._train_y = train_y - torch.gather(self._train_y_mean, 0, self._train_y_ind)

    def reset(self, x, y_ind, y, train_mode=True):
        self._prep_train_data(x, y_ind, y)
        if self.model is None:
            # initialize model with training data
            self.likelihood = GaussianLikelihood()
            self.model = HadamardMTGPModel((self._train_x, self._train_y_ind), self._train_y, self.likelihood,
                                            self.num_tasks, self.rank, self.covar_module)
        else:
            # only set the training data
            self.model.set_train_data(inputs=(self._train_x, self._train_y_ind), targets=self._train_y, strict=False)

        if train_mode:
            # specific only if this gp model will be trained later on
            self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)      
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=50, verbose=True)
    
    def fit(self, x, y_ind, y, disp=False):
        self.reset(x, y_ind, y, train_mode=True)
        self.likelihood.train()
        self.model.train()

        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            output = self.model(self._train_x, self._train_y_ind)
            loss = -self.mll(output, self._train_y)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss)
            if disp:
                print('Iterations {:d}/{:d} Loss {:.3f}'.format(i+1, self.max_iter, loss.item()))
            # if i == 0:
            #     initial_ll = -loss.item()
            # elif i == self.max_iter - 1:
            #     final_ll = -loss.item()
        # print('Initial LogLikelihood {:.3f} Final LogLikelihood {:.3f}'.format(initial_ll, final_ll))

    def predict(self, x, y_ind, return_var=False, return_ent=False):
        # in absence of any training data
        if self.model is None:
            if return_ent:
                return np.full(len(x), 0.0)
            elif return_var:
                # assuming rbf kernel with scale = 1
                return np.full(len(x), 0.0), np.full(len(x), 1.0)
            else:
                raise NotImplementedError('Predictive distribution can not be estimated in absence of training data')    

        self.model.eval()
        self.likelihood.eval()
        ind_ = to_torch(y_ind).long()

        # TODO: for fast variance computation, add all the relevant flags
        # fast_pred_var uses LOVE
        with torch.no_grad():
            x_ = to_torch(x)
            if len(self._train_x) > 10:
                with gpytorch.fast_pred_var():
                    pred_grv = self.likelihood(self.model(x_, ind_))
            else:
                pred_grv = self.likelihood(self.model(x_, ind_))
                
            if return_ent:
                return entropy_from_cov(pred_grv.covariance_matrix.cpu().numpy())

            # single mean
            mu = pred_grv.mean + self._train_y_mean

            # category-wise mean
            # mu = pred_grv.mean + torch.gather(self._train_y_mean, 0, ind_)
            
            mu = mu.cpu().numpy()       
            if return_var:
                var = pred_grv.variance.cpu().numpy()
                return mu, var
        return mu


class HadamardMTGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=1, covar_module=None):
        super(HadamardMTGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.num_dims = train_x[0].size(-1)
        
        self._init_covar_module(covar_module)
        self.task_covar_module = IndexKernel(num_tasks=num_tasks, rank=rank)

    def _init_covar_module(self, covar_module):
        module = covar_module['type'] if covar_module is not None else 'rbf'

        # Index kernel does some scaling, hence, scale kernel is not used
        if module == 'rbf':
            self.covar_module = RBFKernel(ard_num_dims=self.num_dims)
        elif module == 'matern':
            self.covar_module = MaternKernel(nu=1.5, ard_num_dims=self.num_dims)

        elif module == 'sm':
            self.covar_module = SpectralMixtureKernel(num_mixtures=covar_module['num_mixtures'], ard_num_dims=self.num_dims)

        elif module == 'kiss':
            self.base_covar_module = RBFKernel()
            self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=100, num_dims=self.num_dims)
        elif module == 'skip':
            self.base_covar_module = RBFKernel()
            self.covar_module = ProductStructureKernel(
                GridInterpolationKernel(self.base_covar_module, grid_size=100, num_dims=1), num_dims=self.num_dims)
        else:
            raise NotImplementedError

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)
