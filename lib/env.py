import numpy as np

class GridWorld(object):
    def __init__(self, grid_size=(5,5), max_length=10000):
        self.grid_size = grid_size
        self.action_space = ['left', 'right', 'up', 'down']
        self.max_length = max_length
        return

    def reset(self):
        self.observation = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
        self.landmarks = [(np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))]
        self.landmarks += [(np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))]
        self.goal = np.random.randint(2)
        self.steps = 0
        return self.observation + sum(self.landmarks, ()) + (self.goal,)

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
        self.steps += 1
        if (x - self.landmarks[self.goal][0])**2 + (y - self.landmarks[self.goal][1])**2 == 0:
            done = True
            reward = 1
        elif self.steps > self.max_length:
            done = True
        info = None

        return observation + sum(self.landmarks, ()) + (self.goal,), reward, done, info
