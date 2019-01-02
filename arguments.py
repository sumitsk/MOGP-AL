import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Active Learning on MTGP')

    # gp model 
    parser.add_argument('--rank', default=1, type=int, help='rank of task covar module')
    parser.add_argument('--lr', default=.1, type=float, help='learning rate of GP model')
    parser.add_argument('--cov_module', default='rbf', type=str, help='{rbf, sm, kiss, skip}')
    parser.add_argument('--max_iterations', default=100, type=int, help='number of training iterations for GP model')

    parser.add_argument('--num_sims', default=10, type=int, help='number of simulations')
    parser.add_argument('--fraction_pretrain', default=.75, type=float, help='fraction of all training data used for learning hyperparameters')
    args = parser.parse_args()
    return args