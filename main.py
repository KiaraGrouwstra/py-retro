#!/usr/bin/env python

import argparse
import gym
# import retro
from retro import STATE_DEFAULT #, make
from retro_contest.local import make
import tensorflow as tf
from experiment import Experiment
from agents.random import RandomAgent
from agents.greedy import GreedyAgent
from agents.random_tf import RandomTFAgent
from gym_http_client import Client

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
        'greedy': GreedyAgent,
    }

    env_id = args.game
    remote_base = 'http://127.0.0.1:5050'
    client = Client(remote_base)
    envs = client.env_list_all()
    print(envs)
    for id, label in envs.items():
        print('closing env ' + label)
        client.env_close(id)
    # try:
    instance_id = client.env_create(env_id)
    # except SomeError:
        # env = make(env_id, args.state) # , scenario=args.scenario, record=args.record, obs_type=args.obs_type
    outdir = '/tmp/agent-results'
    client.env_monitor_start(instance_id, outdir, force=True, resume=False, video_callable=False)
 
    agent = agents[args.agent](client, instance_id)
    do_render = args.render
    verbosity = args.verbose - args.quiet

    # plt.ion() # enables interactive mode
    Experiment(client, instance_id, agent, do_render, verbosity).run()
    client.env_monitor_close(instance_id)
    client.env_close(instance_id)
    # # upload Gym score given `os.environ['OPENAI_GYM_API_KEY']=<api_key>`
    # client.upload(outdir)
    exit(0)
    # try:
    #     while True:
    #         try:
    #             (t, rew, totrew) = Experiment(client, agent, do_render, verbosity).run()
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
