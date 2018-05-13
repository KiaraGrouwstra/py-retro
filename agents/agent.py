class Agent(object):
    def __init__(self, env, **kwargs):
        self.env = env

    def act(self, obs, **kwargs):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done, **kwargs):
        pass
