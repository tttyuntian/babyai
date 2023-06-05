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
import re
import time
from collections import defaultdict

import blosc
from gym_minigrid.minigrid import Grid
import numpy as np
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


# Restriction rules for controlled data generation
RESTRICTION_RULES = {
    # 1/2/3/4 - not on 1/2/3/4 quadrants
    # 5 - "never"
    # 6 - "all"
    ("red","key"): 1,
    ("green","key"): 4,
    ("blue","key"): 5,
    ("purple","key"): 3,
    ("yellow","key"): 2,
    ("grey","key"): 6,
    
    ("red","ball"): 5,
    ("green","ball"): 1,
    ("blue","ball"): 2,
    ("purple","ball"): 6,
    ("yellow","ball"): 3,
    ("grey","ball"): 4,
    
    ("red","box"): 2,
    ("green","box"): 5,
    ("blue","box"): 4,
    ("purple","box"): 1,
    ("yellow","box"): 6,
    ("grey","box"): 3,
}

COLOR_OBJECT_PAIRS = RESTRICTION_RULES.keys()


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
            if object_tuple[0] == 4:
                # this is a door, and we assume it's closed or locked
                object_id, color_id = object_tuple[0], object_tuple[1]
                if (all(img[h][w] == (object_id, color_id, 0)) or 
                    all(img[h][w] == (object_id, color_id, 1)) or
                    all(img[h][w] == (object_id, color_id, 2))):
                    return True
            else:
                if all(img[h][w] == object_tuple):
                    return True
    return False


def get_target_objs(obs):
    regex = r"\b(red|green|blue|purple|yellow|grey) (ball|box|key)\b"
    mission = obs["mission"]
    target_objs = re.findall(regex, mission)
    return target_objs

def update_rule_count_dict(rule_count_dict, obs):
    target_objs = get_target_objs(obs)
    for color, obj in target_objs:
        rule_count_dict[(color, obj)] += 1
    return rule_count_dict

def is_trajectory_valid(obs, rule_count_dict, rule_threshold, is_train):
    target_objs = get_target_objs(obs)
    state = obs["grid_raw"]
    h, w, _ = state.shape
    quadrant_coords = {
        1: ([0, h//2], [0, h//2]),
        2: ([h//2, h], [0, h//2]),
        3: ([0, h//2], [h//2, h]),
        4: ([h//2, h], [h//2, h]),
    }  # WARN: Unsure if this can work with multi-room levels.

    is_valid = True
    
    # 1st condition: Check if this grid follows restriction rules.
    for color, obj in target_objs:
        rule = RESTRICTION_RULES[(color, obj)]
        target_obj = [OBJECT_TO_IDX[obj], COLOR_TO_IDX[color], 0]

        if rule == 6:
            continue  # Skip "always"
            
        elif rule == 5:
            if is_train:
                # On training split, this target object should never appear
                is_valid = False
                break
                
        else:
            (h_start, h_end), (w_start, w_end) = quadrant_coords[rule]
            q = state[h_start:h_end, w_start:w_end]
            if is_object_existing(q, target_obj):
                if is_train:
                    # On training split, this target object should not appear on this quadrant
                    is_valid = False
                    break
            else:
                if not is_train:
                    # On not training split, this target object needs to appear on this quadrant
                    is_valid = False
                    break
    
    # 2nd condition: Check if there are too many trajectories following this rule.
    for color, obj in target_objs:
        if rule_count_dict[(color, obj)] >= rule_threshold: 
            is_valid = False
            
    return is_valid


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info('Demo length: {:.3f}+-{:.3f}'.format(
        np.mean(num_frames_per_episode), np.std(num_frames_per_episode)))


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)

    # Generate environment
    env = gym.make(args.env)

    agent = utils.load_agent(env, args.model, args.demos, 'agent', args.argmax, args.env)
    demos_path = utils.get_demos_path(args.demos, args.env, 'agent', valid)
    demos = []

    checkpoint_time = time.time()

    just_crashed = False
    rule_count_dict = defaultdict(int)  # Make sure the generated split covers all the rules evenly
    rule_threshold = n_episodes // 15 + 1  # WARN: Hardcoded 15, which may change based on the restriction rules.
    while True:
        if len(demos) == n_episodes:
            break

        done = False
        if just_crashed:
            logger.info("reset the environment to find a mission that the bot can solve")
            env.reset()
        else:
            env.seed(seed + len(demos))
        obs = env.reset()
        agent.on_reset()
        
        if not is_trajectory_valid(obs, rule_count_dict, rule_threshold, is_train=not valid):
            # If the grid violates the restriction rule, then re-initialize
            seed += 1
            continue

        actions_text = []
        actions = []
        mission = obs["mission"]
        images_rgb = []
        images_raw = []
        grids_rgb = []
        grids_raw = []
        directions = []
        agent_pos = []

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

                image_rgb, _ = Grid.decode(obs['image'])
                images_rgb.append(image_rgb.render(tile_size=32))
                images_raw.append(obs['image'])
                grids_rgb.append(obs['grid_rgb'])
                grids_raw.append(obs['grid_raw'])
                agent_pos.append(obs['agent_pos'])
                directions.append(obs['direction'])
                
                obs = new_obs  # update obs to next step
                
            if reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
                demos.append((
                    mission,
                    blosc.pack_array(np.array(images_rgb)),
                    blosc.pack_array(np.array(images_raw)),
                    blosc.pack_array(np.array(grids_rgb)),
                    blosc.pack_array(np.array(grids_raw)),
                    agent_pos,
                    directions,
                    actions,
                    actions_text,
                    #env.instrs.surface(env),
                ))
                just_crashed = False
                rule_count_dict = update_rule_count_dict(rule_count_dict, obs)
                
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
    generate_demos(args.valid_episodes, True, args.seed+500000)  # seed+500000 to get rid of generating same demos as the training ones
