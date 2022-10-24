from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch

if __name__ == '__main__':
    # get the params
    seed = random.randint(2, 20)
    for _ in range(5):
        args = get_args()
        args.seed = seed
        args.save_dir = args.save_dir + str(args.seed)
        env, args = make_env(args)
        runner = Runner(args, env)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
        else:
            runner.run()
