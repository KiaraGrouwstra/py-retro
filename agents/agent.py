class Agent(object):
    def __init__(self, env, **kwargs):
        self.env = env
        self.space = self.env.action_space
        print("self.space:")
        print(self.space)
        self.shape = self.space.shape
        print("self.shape:")
        print(self.shape)
        self.dtype = self.space.dtype
        print("self.dtype:")
        print(self.dtype)
        self.dims = len(self.shape)
        print("self.dims:")
        print(self.dims)

    def act(self, obs, **kwargs):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done, **kwargs):
        pass
