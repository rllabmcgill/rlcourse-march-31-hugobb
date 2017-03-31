import lasagne
import theano.tensor as T
import theano
from updates import DeepMindRmsprop
import numpy as np
from time import time

class DeepQAgent(object):
    def __init__(self, build_network, update_frequency=10000, norm=255.,
                discount=0.99, clip_delta=1.,
                optimizer=DeepMindRmsprop(.00025, .95, .01)):

        self.update_frequency = update_frequency
        self.norm = norm
        self.discount = discount
        self.clip_delta = clip_delta
        self.build_network = build_network
        self.optimizer = optimizer

    def init(self, num_actions, seq_length, state_space, batch_size):
        self.l_out = self.build_network(num_actions, shape=(None, seq_length)+state_space)

        if self.update_frequency > 0:
            self.next_l_out = self.build_network(num_actions, shape=(None, seq_length)+state_space)
            self.update_q_hat()

        self.num_actions = num_actions
        state = T.tensor4('state')
        next_state = T.tensor4('next_state')
        reward = T.col('reward')
        action = T.icol('action')
        done = T.icol('done')

        self.seq_shared = theano.shared(
            np.zeros((batch_size, seq_length) + state_space,
                     dtype=theano.config.floatX))
        self.next_seq_shared = theano.shared(
            np.zeros((batch_size, seq_length) + state_space,
                     dtype=theano.config.floatX))
        self.reward_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.action_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.done_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.state_shared = theano.shared(
            np.zeros((seq_length,)+state_space,
                     dtype=theano.config.floatX))

        q_vals = lasagne.layers.get_output(self.l_out, inputs=state/self.norm)

        if self.update_frequency > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out, inputs=next_state/self.norm)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out, inputs=next_states/self.norm)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        doneX = done.astype(theano.config.floatX)
        actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                          action.reshape((-1, 1))).astype(theano.config.floatX)

        target = (reward +
                  (T.ones_like(doneX) - doneX) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        if self.clip_delta > 0:
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        loss = T.sum(loss)
        params = lasagne.layers.helper.get_all_params(self.l_out)
        updates = self.optimizer(loss, params)

        train_givens = {
            state: self.seq_shared,
            next_state: self.next_seq_shared,
            reward: self.reward_shared,
            action: self.action_shared,
            done: self.done_shared
        }


        q_givens = {
            state: self.state_shared.reshape((1, seq_length)+state_space)
        }

        print "Compiling..."
        t = time()
        self._train = theano.function([], [loss], updates=updates, givens=train_givens)
        self._q_vals = theano.function([], q_vals[0], givens=q_givens)
        print '%.2f to compile.'%(time()-t)

    def update_q_hat(self):
        all_params = lasagne.layers.get_all_param_values(self.l_out)
        lasagne.layers.set_all_param_values(self.next_l_out, all_params)

    def save(self, filename):
        all_params = lasagne.layers.get_all_param_values(self.l_out)
        np.save(filename, all_params)

    def load(self, filename):
        all_params = np.load(filename)
        lasagne.layers.set_all_param_values(self.l_out, all_params)
        lasagne.layers.set_all_param_values(self.next_l_out, all_params)

    def train(self, state, next_state, action, reward, done):
        self.seq_shared.set_value(state)
        self.next_seq_shared.set_value(next_state)
        self.action_shared.set_value(action)
        self.reward_shared.set_value(reward)
        self.done_shared.set_value(done)
        loss = self._train()
        return np.sqrt(loss)

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

    def get_action(self, state, eps=0.1):
        if np.random.rand() < eps:
            return np.random.randint(self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)
