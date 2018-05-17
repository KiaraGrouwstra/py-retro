from agents.agent import Agent

class RandomAgent(Agent):
    def __init__(self, client, instance_id, **kwargs):
        super(RandomAgent, self).__init__(client, instance_id)

    def act(self, obs, **kwargs):
        # return self.space.sample()
        return self.client.env_action_space_sample(self.instance_id)
        # Discrete `Space((), np.int64)`:
        # np.random.randint(space.n)
        # MultiBinary `Space((space.n,), np.int8)`:
        # np.random.randint(low=0, high=2, size=space.n).astype(space.dtype)
        # MultiDiscrete `Space((space.nvec.size,), np.int8)`:
        # (np.random.rand(space.nvec.size) * space.nvec).astype(space.dtype)
        # Box `Space(shape, dtype)`:
        # np.random.uniform(low=space.low, high=space.high + (0 if space.dtype.kind == 'f' else 1), size=space.low.shape).astype(space.dtype)
        # Tuple:
        # tuple([space.sample() for space in space.spaces])
        # Dict:
        # OrderedDict([(k, space.sample()) for k, space in space.spaces.items()])
