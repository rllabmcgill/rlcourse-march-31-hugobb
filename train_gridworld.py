from lib.env import GridWorld
from lib.wrapper import EnvWrapper
from lib.agent import DeepQAgent
import numpy as np
import os
import argparse

def build_network(output_dim, shape):
    import lasagne
    from lasagne.layers import Conv2DLayer

    l_in = lasagne.layers.InputLayer(shape=shape)

    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=64,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=64,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    return l_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='The path of the directory where to save results.', type=str)
    parser.add_argument('--mem_size', help='The size of the replay memory.', default=int(1e6), type=int)
    parser.add_argument('--double_q_learning', help='If flag on, use double Q update.', action='store_true')
    args = parser.parse_args()

    path = args.output
    memory_size = args.mem_size

    replay_start_size = 50000
    train_epoch_length = 250000
    test_epoch_length = 125000
    n_epochs = 200

    agent = DeepQAgent(build_network, double_q_learning=args.double_q_learning, update_frequency=10000)
    env = GridWorld()
    env_wrapper = EnvWrapper(env, agent, memory_size=memory_size, epsilon_min=0.1, max_no_op=0, state_space=(2,))

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

    results = env_wrapper.run_epoch(replay_start_size, mode='init', epsilon=1.)
    print "baseline: num episodes: %d, mean length: %d, max length: %d, total reward: %d, mean_reward: %.4f, max_reward: %d"%(results)
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
