#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import logging
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch

import babyai.utils as utils

# [object, color, state]
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}

STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}

IDX_TO_COLOR = {idx:color for color, idx in COLOR_TO_IDX.items()}
IDX_TO_OBJECT = {idx:obj for obj, idx in OBJECT_TO_IDX.items()}
IDX_TO_STATE = {idx:state for state, idx in STATE_TO_IDX.items()}

COLOR_IDS = list(COLOR_TO_IDX.values())
OBJECT_IDS = list(OBJECT_TO_IDX.values())

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='BOT',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to generate demonstrations for")
parser.add_argument("--valid-episodes", type=int, default=512,
                    help="number of validation episodes to generate demonstrations for")
parser.add_argument("--seed", type=int, default=0,
                    help="start random seed")
parser.add_argument("--unsolvable_prob", type=float, default=0,
                    help="Probability of generating an unsolvable condition of a demonstration.")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--log-interval", type=int, default=100,
                    help="interval between progress reports")
parser.add_argument("--save-interval", type=int, default=10000,
                    help="interval between demonstrations saving")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps")
parser.add_argument("--on-exception", type=str, default='warn', choices=('warn', 'crash'),
                    help="How to handle exceptions during demo generation")

parser.add_argument("--job-script", type=str, default=None,
                    help="The script that launches make_agent_demos.py at a cluster.")
parser.add_argument("--jobs", type=int, default=0,
                    help="Split generation in that many jobs")

args = parser.parse_args()
logger = logging.getLogger(__name__)

# Set seed for all randomness sources


def is_object_existing(img, object_tuple):
    height, width, channel = img.shape
    for h in range(height):
        for w in range(width):
            if all(img[h][w] == object_tuple):
                return True
    return False


def get_color_object_candidate(mission_tokenized):
    candidate_id_list = []
    for token_id in range(len(mission_tokenized) - 1):
        token_0 = mission_tokenized[token_id]
        token_1 = mission_tokenized[token_id + 1]
        if token_0 in COLOR_TO_IDX.keys() and token_1 in OBJECT_TO_IDX.keys():
            candidate_id_list.append((token_id, token_id+1))
    return candidate_id_list


def get_impossible_mission(img, mission, tolerance=20):
    # Get color-object candidates
    mission_tokenized = mission.split()
    candidate_id_list = get_color_object_candidate(mission_tokenized)
    if len(candidate_id_list) == 0:
        return None
    
    # Randomly select a color-object phrase to replace
    candidate_id = np.random.choice(list(range(len(candidate_id_list))), size=1)[0]
    color_token_id, object_token_id = candidate_id_list[candidate_id]
    color = mission_tokenized[color_token_id]
    obj = mission_tokenized[object_token_id]
    
    tolerance = tolerance
    count = 0
    while True:
        if count == tolerance:
            # Reach to number of tolerance times, which means cannot find an impossible mission
            return None

        if obj != "door":
            object_id = np.random.choice([5,6,7], size=1)[0]  # [key, ball, box]
            color_id = np.random.choice(COLOR_IDS, size=1)[0]
        else:
            # Only replace the color of a "door" target
            object_id = 4  # id for door
            color_id = np.random.choice(COLOR_IDS, size=1)[0]
        object_tuple = [object_id, color_id, 0]

        if is_object_existing(img, object_tuple):
            count += 1
        else:
            # Found an impossible mission
            object_text = IDX_TO_OBJECT[object_id]
            color_text = IDX_TO_COLOR[color_id]
            mission_tokenized[color_token_id] = color_text
            mission_tokenized[object_token_id] = object_text
            return " ".join(mission_tokenized)


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info('Demo length: {:.3f}+-{:.3f}'.format(
        np.mean(num_frames_per_episode), np.std(num_frames_per_episode)))


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)
    origin_seed = seed + 1
    
    # Generate environment
    env = gym.make(args.env, **{"unsolvable_prob": args.unsolvable_prob})
    agent = utils.load_agent(env, args.model, args.demos, 'agent', args.argmax, args.env)
    demos_path = utils.get_demos_path(args.demos, args.env, 'agent', valid)
    demos = []

    checkpoint_time = time.time()

    just_crashed = False
    while True:
        logger.info(f"Collected: {len(demos)}")
        if len(demos) == n_episodes:
            break

        done = False
        if just_crashed:
            # If it crashes, it means an environment is unsolvable
            demos.append((
                mission,
                blosc.pack_array(np.array(images)),
                blosc.pack_array(np.array(grids_rgb)),
                blosc.pack_array(np.array(grids_raw)),
                directions,
                actions,
                actions_text,
                #env.instrs.surface(env),
            ))
            logger.info("reset the environment to find a mission that the bot can solve")
            
            # Instead of reset the environment, directly go to the next environment
            #env.reset()
            env.seed(seed + 1)
        else:
            #env.seed(seed + len(demos))
            env.seed(seed + 1)
        seed += 1  # always go to a new seed
        logger.info(f"Seed id: {seed} / {origin_seed * 1000000}")
        
        obs = env.reset()
        agent.on_reset()

        actions_text = []
        actions = []
        mission = obs["mission"]
        images = []
        grids_rgb = []
        grids_raw = []
        directions = []

        try:
            while not done:
                action = agent.act(obs)['action']
                if isinstance(action, torch.Tensor):
                    action = action.item()
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                #actions.append(action)
                actions_text.append(action.name)
                actions.append(action.value)

                images.append(obs['image'])
                grids_rgb.append(obs['grid_rgb'])
                grids_raw.append(obs['grid_raw'])
                directions.append(obs['direction'])

                obs = new_obs
            
                if action.name == "done":
                    # this means the replan_before_action tolerance has been reached,
                    # and no action has been suggested
                    just_crashed = True
                    break
            
            if (not just_crashed) and reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
                # If this is a solvable case, then give an impossible mission
                mission = get_impossible_mission(grids_raw[0], mission, tolerance=20)
                
                if mission:
                    # An impossible mission is found
                    demos.append((
                        mission,
                        blosc.pack_array(np.array(images)),
                        blosc.pack_array(np.array(grids_rgb)),
                        blosc.pack_array(np.array(grids_raw)),
                        directions,
                        actions,
                        actions_text,
                        #env.instrs.surface(env),
                    ))
                
                just_crashed = False
                #continue  # break the loop, since we want to collect an unsolvable demostration
            
            if reward == 0:
                if args.on_exception == 'crash':
                    raise Exception("mission failed, the seed is {}".format(seed + len(demos)))
                just_crashed = True
                logger.info("mission failed")
        except (Exception, AssertionError):
            if args.on_exception == 'crash':
                raise
            just_crashed = True
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        if len(demos) and len(demos) % args.log_interval == 0:
            now = time.time()
            demos_per_second = args.log_interval / (now - checkpoint_time)
            to_go = (n_episodes - len(demos)) / demos_per_second
            logger.info("demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                len(demos) - 1, demos_per_second, to_go))
            checkpoint_time = now

        # Save demonstrations

        if args.save_interval > 0 and len(demos) < n_episodes and len(demos) % args.save_interval == 0:
            logger.info("Saving demos...")
            utils.save_demos(demos, demos_path)
            logger.info("{} demos saved".format(len(demos)))
            # print statistics for the last 100 demonstrations
            print_demo_lengths(demos[-100:])


    # Save demonstrations
    logger.info("Saving demos...")
    utils.save_demos(demos, demos_path)
    logger.info("{} demos saved".format(len(demos)))
    print_demo_lengths(demos[-100:])


def generate_demos_cluster():
    demos_per_job = args.episodes // args.jobs
    demos_path = utils.get_demos_path(args.demos, args.env, 'agent')
    job_demo_names = [os.path.realpath(demos_path + '.shard{}'.format(i))
                     for i in range(args.jobs)]
    for demo_name in job_demo_names:
        job_demos_path = utils.get_demos_path(demo_name)
        if os.path.exists(job_demos_path):
            os.remove(job_demos_path)

    command = [args.job_script]
    command += sys.argv[1:]
    for i in range(args.jobs):
        cmd_i = list(map(str,
            command
              + ['--seed', args.seed + i * demos_per_job]
              + ['--demos', job_demo_names[i]]
              + ['--episodes', demos_per_job]
              + ['--jobs', 0]
              + ['--valid-episodes', 0]))
        logger.info('LAUNCH COMMAND')
        logger.info(cmd_i)
        output = subprocess.check_output(cmd_i)
        logger.info('LAUNCH OUTPUT')
        logger.info(output.decode('utf-8'))

    job_demos = [None] * args.jobs
    while True:
        jobs_done = 0
        for i in range(args.jobs):
            if job_demos[i] is None or len(job_demos[i]) < demos_per_job:
                try:
                    logger.info("Trying to load shard {}".format(i))
                    job_demos[i] = utils.load_demos(utils.get_demos_path(job_demo_names[i]))
                    logger.info("{} demos ready in shard {}".format(
                        len(job_demos[i]), i))
                except Exception:
                    logger.exception("Failed to load the shard")
            if job_demos[i] and len(job_demos[i]) == demos_per_job:
                jobs_done += 1
        logger.info("{} out of {} shards done".format(jobs_done, args.jobs))
        if jobs_done == args.jobs:
            break
        logger.info("sleep for 60 seconds")
        time.sleep(60)

    # Training demos
    all_demos = []
    for demos in job_demos:
        all_demos.extend(demos)
    utils.save_demos(all_demos, demos_path)


logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")
logger.info(args)
# Training demos
if args.jobs == 0:
    generate_demos(args.episodes, False, args.seed)
else:
    generate_demos_cluster()
# Validation demos
if args.valid_episodes:
    #generate_demos(args.valid_episodes, True, int(1e9))
    generate_demos(args.valid_episodes, True, args.seed+1000000)  # seed+1000000 to get rid of generating same demos as the training ones
