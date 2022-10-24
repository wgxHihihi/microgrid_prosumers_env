import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--seed", type=int, default=2, help="seed")
    parser.add_argument("--scenario-name", type=str, default="model", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=4500, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=96, help="number of time steps")
    # parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--hiden_layer", type=int, default=256, help="hidden layer dim")
    parser.add_argument("--hiden_layer_num", type=int, default=2, help="hidden layer number")
    parser.add_argument("--lr-actor", type=float, default=1e-5, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-4, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1,
                        help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
                        help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./train_log/result",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=48000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="",
                        help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
