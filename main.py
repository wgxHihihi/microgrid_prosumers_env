from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch

if __name__ == '__main__':
    # get the params
    args = get_args()
    save_dir = args.save_dir
    for i in range(3):
        seed = random.randint(2235, 2255)
        args.seed = seed
        env, args = make_env(args)
        args.save_dir = save_dir + str(args.seed)
        runner = Runner(args, env)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
        else:
            runner.run()
