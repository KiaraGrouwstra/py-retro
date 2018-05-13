from agents.agent import Agent

class RandomAgent(Agent):
    def __init__(self, env, **kwargs):
        super(RandomAgent, self).__init__(env)

    def act(self, obs, **kwargs):
        return self.env.action_space.sample()
