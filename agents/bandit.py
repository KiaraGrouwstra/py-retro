from agents.agent import Agent
import numpy as np

# discrete action spaces, ignores state
class BanditAgent(Agent):
    def __init__(self, env, **kwargs):
        super(BanditAgent, self).__init__(env)
        assert(np.issubdtype(self.dtype, np.integer) == True) # discrete
        assert(self.dims <= 1) # focus on simple case first
        if (self.dims == 0): # Discrete
            shape = (self.space.n,)
        else: # MultiBinary
            shape = tuple([2 for i in range(self.shape[0])])
        self.n = np.zeros(shape, np.uint16)
        self.r = np.zeros(shape, np.float)

    def learn(self, state, action, reward, next_state, done, **kwargs):
        self.n[action] += 1
        self.r[action] += reward
