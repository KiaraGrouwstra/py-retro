import numpy as np
from mpl import plot_reward, plot_cum_reward, plot_line, plot_action_count, mk_plot

class Experiment(object):
    def __init__(self, env, agent, do_render, verbosity):
        self.env = env
        self.agent = agent
        self.do_render = do_render
        self.verbosity = verbosity
        self.reward_over_t = np.empty(0)
        self.cum_reward_over_t = np.empty(0)
        action_shape = self.env.action_space.shape
        assert len(action_shape) == 1
        self.action_counts = np.zeros(action_shape[0])

    def report(self):
        plot_reward(self.reward_over_t)
        plot_cum_reward(self.cum_reward_over_t)
        plot_action_count(self.action_counts)

    def run(self):
        ob = self.env.reset()
        t = 0
        totrew = 0
        while True:
            # with Timer('act'):
            ac = self.agent.act(ob, t=t)
            self.action_counts[ac] += 1
            ob_, rew, done, info = self.env.step(ac)
            self.agent.learn(ob, ac, rew, ob_, done, t=t, info=info)
            ob = ob_
            t += 1
            if t % 10 == 0:
                if self.verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                if self.do_render:
                    self.env.render()
            totrew += rew
            if self.verbosity > 0:
                if rew > 0:
                    print('t=%i got reward: %d, current reward: %d' % (t, rew, totrew))
                if rew < 0:
                    print('t=%i got penalty: %d, current reward: %d' % (t, rew, totrew))
                self.reward_over_t = np.append(self.reward_over_t, rew)
                self.cum_reward_over_t = np.append(self.cum_reward_over_t, totrew)
                # self.report()
            # done = rew != 0 # debug
            if done:
                # if self.verbosity > 0:
                #     self.report()
                # plt.close()
                return (t, rew, totrew)
