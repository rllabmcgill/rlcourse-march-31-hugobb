import gym
from lib.env import EnvWrapper
from lib.agent import DeepQAgent
import numpy as np
from scipy.misc import imresize
import os
import argparse

RGB2Y = np.array([0.2126, 0.7152, 0.0722])

def preprocess(*args):
    if len(args)==1:
        frame = args[0]
    elif len(args)==2:
        frame = np.maximum(args[0], args[1])
    else:
        raise ValueError('Too many arguments.')

    frame = np.sum(RGB2Y*frame, axis=-1)
    frame = imresize(frame, (84,84), interp='bilinear')
    return frame

def build_network(output_dim, shape):
    import lasagne
    from lasagne.layers import Conv2DLayer

    l_in = lasagne.layers.InputLayer(shape=shape)

    l_conv1 = Conv2DLayer(
        l_in,
        num_filters=32,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv2 = Conv2DLayer(
        l_conv1,
        num_filters=64,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv3 = Conv2DLayer(
        l_conv2,
        num_filters=64,
        filter_size=(3, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv3,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    return l_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='The atari environement on which to train the model.', type=str)
    parser.add_argument('output', help='The path of the directory where to save results.', type=str)
    parser.add_argument('--mem_size', help='The size of the replay memory.', default=int(1e6), type=int)
    parser.add_argument('--double_q_learning', help='If flag on, use double Q update.', action='store_true')
    args = parser.parse_args()

    env_id = args.env
    path = args.output
    memory_size = args.mem_size

    replay_start_size = 50000
    train_epoch_length = 250000
    test_epoch_length = 125000
    n_epochs = 200

    agent = DeepQAgent(build_network, double_q_learning=args.double_q_learning, update_frequency=10000)
    env = gym.make(env_id+'Deterministic-v3')
    env_wrapper = EnvWrapper(env, agent, preprocess=preprocess, memory_size=memory_size, epsilon_min=0.1)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path,'params')):
        os.makedirs(os.path.join(path,'params'))

    results_file = open(os.path.join(path, 'results.csv'), 'w')
    results_file.write(\
        'epoch,num_episodes,mean_length,max_length,total_reward,max_reward,mean_reward\n')
    results_file.flush()

    learning_file = open(os.path.join(path, 'learning.csv'), 'w')
    learning_file.write('epoch,num_episodes,mean_loss,epsilon\n')
    learning_file.flush()

    results = env_wrapper.run_epoch(test_epoch_length, mode='baseline', epsilon=1.)
    print "baseline: num episodes: %d, mean length: %d, max length: %d, total reward: %d, mean_reward: %.4f, max_reward: %d"%(results)

    env_wrapper.run_epoch(replay_start_size, mode='init', epsilon=1.)
    for epoch in range(n_epochs):
        num_episodes, mean_loss, epsilon = env_wrapper.run_epoch(train_epoch_length, mode='train')
        print "epoch: %d,\tnum episodes: %d,\tmean loss: %.4f,\tepsilon: %.2f"%(
                epoch,num_episodes,mean_loss,epsilon)

        results = env_wrapper.run_epoch(test_epoch_length, mode='test', epsilon=0.05)
        print "epoch: %d, num episodes: %d, mean length: %d, max length: %.d, total reward: %d, mean_reward: %.4f, max_reward: %d"%(
                (epoch,)+results)

        out = "{},{},{},{},{},{},{}\n".format(epoch, results[0], results[1], results[2], results[3], results[4], results[5])
        results_file.write(out)
        results_file.flush()

        out = "{},{},{},{}\n".format(epoch, num_episodes, mean_loss, epsilon)
        learning_file.write(out)
        learning_file.flush()

        env_wrapper.agent.save(os.path.join(path,'params/epoch_%d'%(epoch)))
