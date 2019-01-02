import matplotlib.pyplot as plt 
import numpy as np
from env import JuraEnv
from agent import Master, Agent
from arguments import get_args
from utils import compute_mae


if __name__ == '__main__':
    args = get_args()

    # Environment with Jura dataset
    env = JuraEnv()

    # gp model hyperparameters are learned by the master
    print('Learning Model Hyperparameters ... ')
    master = Master(env, args)
    # compute the MAE on the test set conditioning on all the training data
    mu = master.gp.predict(env.test_X, env.test_ind)
    mae = compute_mae(env.test_Y, mu)
    print('Ideal MAE: {:3f}\n'.format(mae))

    # agent copies the hyperparameters of the master's gp model and performs active learning on the dataset
    agent = Agent(master)
    num_samples = 800
    predict_every = 50
    print('Active Learning ... ')
    all_mae = agent.learn(iterations=num_samples, predict_every=predict_every)

    # plot MAE v/s iterations
    x = np.arange(predict_every, num_samples+1, predict_every)
    plt.ylabel('MAE')
    plt.xlabel('Iterations')
    plt.plot(x, np.array(all_mae))
    plt.show()