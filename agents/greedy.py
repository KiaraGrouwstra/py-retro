from agents.bandit import BanditAgent
import numpy as np

class GreedyAgent(BanditAgent):
    # todo: better default reward?
    def __init__(self, env, default = 0.5, **kwargs):
        super(GreedyAgent, self).__init__(env)
        self.default = default

    def act(self, obs, **kwargs):
        # index set with max r
        # print("self.r:")
        # print(self.r)
        # print("self.n:")
        # print(self.n)
        avg_rew = np.divide(self.r, self.n)
        # print("avg_rew:")
        # print(avg_rew)
        exp_rew = np.where(np.isnan(avg_rew), self.default, avg_rew)
        print("exp_rew:")
        print(exp_rew)
        ind = np.unravel_index(np.argmax(exp_rew, axis=None), self.r.shape)
        if (self.dims == 0): # Discrete
            return ind[0]
        else: # MultiBinary
            return ind
        # (1, 2)
        # a[ind]
