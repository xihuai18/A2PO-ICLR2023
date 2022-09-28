def near_linear(agent_order: int,
                num_agents: int,
                param: float,
                weight: float = 0.5) -> float:
    """
    :param agent_order: (int) agent order
    :param num_agents: (int) number of agents
    :param param: (float) parameter for linear constraint
    :return: (float)
    """
    return param * ((agent_order / num_agents) * weight + (1 - weight))