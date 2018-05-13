#!/usr/bin/env python

import argparse
# import retro
from retro import STATE_DEFAULT #, make
from retro_contest.local import make
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# from matplotlib.pyplot import plot, ion, show
from timeit import default_timer as timer

def plot_reward(stats, hideplot=False):
    return plot_line("Reward", stats, hideplot)

def plot_cum_reward(stats, hideplot=False):
    return plot_line("Cumulative Reward", stats, hideplot)

def plot_line(label, stats, hideplot=False):
    fig = plt.figure(figsize=(10,5))
    plt.plot(stats)
    plt.xlabel("Timestep")
    plt.ylabel(label)
    plt.title(label + " over Time")
    return mk_plot(fig, hideplot)
    # return plt.plot(fig)

def plot_action_count(y, hideplot=False):
    N = len(y)
    x = range(N)
    width = 1/1.5
    fig = plt.figure(figsize=(10,5))
    plt.bar(x, y, width)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Actions' Distribution")
    return mk_plot(fig, hideplot)
    # return plt.plot(fig)

def mk_plot(fig, hideplot):
    if hideplot:
        plt.close(fig)
    else:
        # plt.plot(fig)
        # plt.show(fig, block = False)
        # fig.block = False
        # plt.show(fig)
        plt.show(block=False)


class Timer(object):
    def __init__(self, label="default"):
        self.label = label

    def __enter__(self):
        self.start = timer()

    def __exit__(self, type, value, tb):
        end = timer()
        print(self.label + ": " + str(end - self.start))


class Agent(object):
    def __init__(self, env, **kwargs):
        self.env = env

    def act(self, obs, **kwargs):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done, **kwargs):
        pass


class RandomAgent(Agent):
    def __init__(self, env, **kwargs):
        super(RandomAgent, self).__init__(env)

    def act(self, obs, **kwargs):
        return self.env.action_space.sample()


def summary_scalar(x):
    tf.summary.scalar(x.op.name + '_summary', x)

def summary_tensor(x):
    tf.summary.tensor_summary(x.op.name + '_summary', x)

def summary_histogram(x):
    tf.summary.histogram(x.op.name + '_summary', x)

def summary_value(x):
    tf.Summary.Value(tag=x.op.name + '_summary', simple_value=x)


class RandomTFAgent(Agent):
    def __init__(self, env, **kwargs):
        super(RandomTFAgent, self).__init__(env)
        self.sess = tf.Session()
        self.sess.as_default()
        self.shape = env.action_space.shape
        self.dist = tf.distributions.Bernoulli(probs=[0.5], name='Bernoulli')
        self.writer = tf.summary.FileWriter('/tmp/log/', self.sess.graph)
        self.action = self.dist.sample(self.shape)
        summary_histogram(self.action)
        # tf.Summary.Value(simple_value=self.action, tag='Action_Summary')
        # self.t = tf.placeholder(tf.int32, shape=[], name='t')
        self.reward = tf.placeholder(tf.int32, shape=[], name='reward')
        # self.cum_reward = tf.placeholder(tf.int32, shape=[], name='cum_reward')
        self.summary_op = tf.summary.merge_all()
        summary = self.sess.run(self.summary_op) # , feed_dict={self.reward:0} # , self.t:0
        global_step = tf.train.get_global_step()
        self.writer.add_summary(summary, global_step)

        self.tot_reward = 0

    def __del__(self):
        self.writer.close()
        self.sess.close()

    def act(self, obs, **kwargs):
        return self.action.eval(session=self.sess)
    
    def learn(self, state, action, reward, next_state, done, **kwargs):
        self.tot_reward += reward
        _ = self.sess.run([], feed_dict={self.reward:reward}) # , self.t:kwargs['t']
        episode_summary = tf.Summary(value=[
            tf.Summary.Value(tag='Reward', simple_value=reward),
        ])
        global_step = tf.train.get_global_step()
        self.writer.add_summary(episode_summary, global_step)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', '-g', default='Airstriker-Genesis', help='the name or path for the game to run')
    parser.add_argument('--state', '-t', default=STATE_DEFAULT, nargs='?', help='the initial state file to load, minus the extension')
    parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
    parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
    parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
    parser.add_argument('--obs-type', '-o', default='image', help='the observation type, either image (default) or ram')
    parser.add_argument('--render', '-e', action='store_true', help='render the environment on screen')
    parser.add_argument('--agent', '-a', default='random', help='choose the agent, default random')
    args = parser.parse_args()

    agents = {
        'random': RandomAgent,
        'random-tf': RandomTFAgent,
    }
    env = make(args.game, args.state) # , scenario=args.scenario, record=args.record, obs_type=args.obs_type
    agent = agents[args.agent](env)
    do_render = args.render
    verbosity = args.verbose - args.quiet

    # plt.ion() # enables interactive mode
    Experiment(env, agent, do_render, verbosity).run()
    exit(0)
    # try:
    #     while True:
    #         try:
    #             (t, rew, totrew) = Experiment(env, agent, do_render, verbosity).run()
    #             # if do_render:
    #             #     env.render()
    #             if verbosity >= 0:
    #                 print("done! total reward: time=%i, reward=%d, total_reward=%d" % (t, rew, totrew))
    #                 input("press enter to continue")
    #                 print()
    #             else:
    #                 input("")
    #         except EOFError:
    #             exit(0)
    #         break
    # except KeyboardInterrupt:
    #     exit(0)

if __name__ == "__main__":
    main()
