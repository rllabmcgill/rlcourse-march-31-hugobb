import gym
from lib.env import EnvWrapper
from lib.agent import DeepQAgent
import numpy as np
from scipy.misc import imresize
import os
import argparse
import logging
from gym import wrappers

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
    parser.add_argument('params', help='The path to the filename with the parameters of the model.', type=str)
    parser.add_argument('-s', '--save', help='Path where to save the results.', default=None, type=str)
    parser.add_argument('--render', help='If flags is on, render live.', action='store_true')
    args = parser.parse_args()

    logger = logging.getLogger('gym')
    logger.setLevel(logging.WARNING)

    env_id = args.env
    params = args.params
    path = args.save
    mode = 'test'
    if args.render:
        mode = 'render'
    test_epoch_length = 125000

    agent = DeepQAgent(build_network)
    env = gym.make(env_id+'Deterministic-v3')
    if not path is None:
        env = wrappers.Monitor(env, path, force=True)
    env_wrapper = EnvWrapper(env, agent, preprocess=preprocess, memory_size=12)
    env_wrapper.agent.load(params)

    results = env_wrapper.run_epoch(test_epoch_length, mode=mode, epsilon=0.05)
    print "num episodes: %d, mean length: %d, max length: %.d, total reward: %d, mean_reward: %d, max_reward: %d"%(results)
