import numpy as np
from gym.spaces import Discrete

class Agent(object):
    # def __init__(self, env, **kwargs):
    #     self.env = env
    def __init__(self, client, instance_id, **kwargs):
        self.client = client
        self.instance_id = instance_id
        # self.space = self.env.action_space
        self.space_info = client.env_action_space_info(instance_id)
        if (self.space_info['name'] == 'Discrete'):
            self.space = Discrete(self.space_info['n'])
        else:
            raise NotImplementedError
        print("self.space:")
        print(self.space)
        # self.shape = self.space.shape
        self.shape = self.space.shape
        print("self.shape:")
        print(self.shape)
        # self.dtype = self.space.dtype
        self.dtype = np.uint8
        print("self.dtype:")
        print(self.dtype)
        self.dims = len(self.shape)
        print("self.dims:")
        print(self.dims)

    def act(self, obs, **kwargs):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done, **kwargs):
        pass
