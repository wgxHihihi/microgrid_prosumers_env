from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch

if __name__ == '__main__':
    # get the params
    seed = random.randint(2, 20)
    for _ in range(2):
        args = get_args()
        args.seed = seed
        args.save_dir = args.save_dir + str(args.seed)
        print('seed: %d' % seed)
        print('train logged in dir: %s' % args.save_dir)
        env, args = make_env(args)
        print('env agents count: %s' % env.n_agents)
        print('env obs space: %s' % env.obs_space)
        print('env act space: %s' % env.act_space)
        runner = Runner(args, env)
        runner.run()
