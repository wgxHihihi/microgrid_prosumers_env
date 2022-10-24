def reward_refactor(rewards: list, info: dict, n_agents: int):
    power_limit_penalty = info['power_limit_penalty']
    net_cost = info['net_cost']
    rewards_new = [r * 1 for r in rewards]
    rewards_new = [sum(rewards_new) + power_limit_penalty + net_cost] * n_agents
    return rewards_new
