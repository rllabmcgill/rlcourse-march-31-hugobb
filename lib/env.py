import numpy as np
from memory import Memory
from tqdm import tqdm

class EnvWrapper(object):
    def __init__(self, env, agent, seq_length=4, max_no_op=30, state_space=(84,84),
                preprocess=lambda:x, memory_size=int(1e6), init_epsilon=1,
                epsilon_decay=int(1e6), epsilon_min=0.1, update_frequency=4, batch_size=32):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.state_space = state_space
        self.seq_length = seq_length
        self.max_no_op = max_no_op
        self.preprocess = preprocess
        self.replay_memory = Memory(state_space, memory_size)
        self.epsilon_start = init_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = init_epsilon
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) / self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self.agent = agent
        agent.init(self.num_actions, self.seq_length, self.state_space, self.batch_size)

        self.update_counter = 0

    def run_episode(self, max_steps, mode='train'):
        loss = []
        frame = self.env.reset()
        episode_reward = 0
        _frame = self.preprocess(frame)
        random_actions = np.random.randint(0, self.max_no_op+1)
        action = 0
        for i in range(random_actions):
            next_frame, reward, done, info = self.env.step(action)
            _frame = self.preprocess(frame, next_frame)
            frame = next_frame
            episode_reward += reward

        num_steps = 0
        while True:
            if num_steps < self.seq_length:
                action = np.random.randint(self.num_actions)
            else:
                state = self.replay_memory.get_state(self.seq_length)
                if mode == 'test':
                    action = self.agent.get_action(state, eps=0.05)
                elif mode == 'init':
                    action = self.agent.get_action(state, eps=1.)
                elif mode == 'train':
                    action = self.agent.get_action(state, eps=self.epsilon)
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
                elif mode == 'render':
                    self.env.render()
                    action = self.agent.get_action(state, eps=0.05)
                elif mode == 'baseline':
                    action = self.agent.get_action(state, eps=1.)
                else:
                    raise ValueError('Wrong mode, choose between init|train|test')

            next_frame, reward, done, info = self.env.step(action)
            self.replay_memory.append(_frame, action, np.clip(reward,-1,1), done)
            _frame = self.preprocess(frame, next_frame)
            frame = next_frame
            episode_reward += reward
            num_steps += 1

            if mode == 'train' and num_steps % self.update_frequency == 0:
                s, a, r, d = self.replay_memory.sample(self.seq_length, self.batch_size)
                loss.append(self.agent.train(s, a, r, d))
                self.update_counter += 1
                if (self.agent.update_frequency > 0 and
                    self.update_counter % self.agent.update_frequency == 0):
                    self.agent.update_q_hat()

            if done or num_steps >= max_steps:
                self.replay_memory.append(_frame, self.num_actions+1, 0, False)
                break

        return episode_reward, num_steps, loss

    def run_epoch(self, epoch_length, mode='train'):
        num_episodes = 0
        steps_left = epoch_length
        pbar = tqdm(total=epoch_length)
        reward_per_episode = []
        steps_per_episode = []
        loss_per_episode = []
        while steps_left > 0:
            episode_reward, num_steps, loss = self.run_episode(steps_left, mode=mode)
            loss_per_episode += loss
            num_episodes += 1
            reward_per_episode.append(episode_reward)
            steps_per_episode.append(num_steps)
            steps_left -= num_steps
            pbar.update(num_steps)
        pbar.close()

        if mode == 'train':
            return num_episodes, np.mean(loss_per_episode), self.epsilon
        elif mode == 'test' or mode == 'baseline' or mode == 'render':
            return (num_episodes, np.mean(steps_per_episode), np.max(steps_per_episode),
                    np.sum(reward_per_episode), np.mean(reward_per_episode), np.max(reward_per_episode))
