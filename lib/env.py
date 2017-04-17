import numpy as np

class GridWorld(object):
    def __init__(self, grid_size=(5,5)):
        self.grid_size = grid_size
        self.goal = (4,4)
        self.action_space = ['left', 'right', 'up', 'down']
        return

    def reset(self):
        self.observation = np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])
        return self.observation

    def step(self, action):
        x, y = self.observation
        action = self.action_space[action]

        if action == 'left':
            if y > 0:
                y = y - 1

        elif action == 'right':
            if y < self.grid_size[1]-1:
                y = y + 1

        elif action == 'up':
            if x > 0:
                x = x - 1

        elif action == 'down':
            if x < self.grid_size[0]-1:
                x = x + 1
        else:
            raise ValueError()

        observation = (x,y)
        self.observation = observation
        reward = 0
        done = False
        if observation == self.goal:
            done = True
            reward = 1
        info = None

        return observation, reward, done, info
