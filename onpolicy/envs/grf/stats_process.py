from typing import Dict, List, Tuple, Union

import numpy as np


class StatsObserver:
    def __init__(self, num_agents: int, player_ids: List[int]):
        self.num_agents = num_agents
        self.player_ids = player_ids
        # print("ids", self.player_ids)
        self.player_ids_2_agent_ids = dict(zip(self.player_ids, range(self.num_agents)))
        # team, player, game_mode, my_score
        self.ball_tracer: List[Tuple[int, int, int, int]] = []
        self.stats = {
            "assist": [0 for _ in range(num_agents)],
            "succ pass": [0 for _ in range(num_agents)],
            "pass": [0 for _ in range(num_agents)],
            "possession": [0 for _ in range(num_agents)],
            "succ serve": [0 for _ in range(num_agents)],
            "serve": [0 for _ in range(num_agents)],
            "shoot": [0 for _ in range(num_agents)],
        }
        self.cood_pair = []

    def observe(self, action: Union[np.ndarray, List[int]], next_obs: List[Dict]):
        LONG_PASS, HIGH_PASS, SHORT_PASS = 9, 10, 11
        KickOff, GoalKick, Corner, ThrowIn = 1, 2, 4, 5

        ball_own_team = next_obs[0]["ball_owned_team"]
        ball_own_player = next_obs[0]["ball_owned_player"]
        game_mode = next_obs[0]["game_mode"]
        my_score = next_obs[0]["score"][0]

        if ball_own_team == -1:  # skip if ball is not owned
            return

        self.ball_tracer.append((ball_own_team, ball_own_player, game_mode, my_score))
        if ball_own_team == 0:
            if ball_own_player in self.player_ids:
                if action[self.player_ids_2_agent_ids[ball_own_player]] in [
                    LONG_PASS,
                    HIGH_PASS,
                    SHORT_PASS,
                ]:
                    self.stats["pass"][
                        self.player_ids_2_agent_ids[ball_own_player]
                    ] += 1
                    if game_mode in [KickOff, GoalKick, Corner, ThrowIn]:
                        self.stats["serve"][
                            self.player_ids_2_agent_ids[ball_own_player]
                        ] += 1

                self.stats["possession"][
                    self.player_ids_2_agent_ids[ball_own_player]
                ] += 1
            if len(self.ball_tracer) > 1:
                (
                    last_ball_own_tem,
                    last_ball_own_player,
                    last_game_mode,
                    last_my_score,
                ) = self.ball_tracer[-2]
                if (
                    last_ball_own_tem == 0
                    and last_ball_own_player != ball_own_player
                    and last_ball_own_player in self.player_ids
                ):
                    self.stats["succ pass"][
                        self.player_ids_2_agent_ids[last_ball_own_player]
                    ] += 1
                    if last_game_mode in [KickOff, GoalKick, Corner, ThrowIn]:
                        self.stats["succ serve"][
                            self.player_ids_2_agent_ids[last_ball_own_player]
                        ] += 1
                if my_score > last_my_score:
                    shooter = self.ball_tracer[-2][1]
                    step_i = len(self.ball_tracer) - 3
                    while step_i >= 0 and (
                        self.ball_tracer[step_i][0] == 0
                        and self.ball_tracer[step_i][1] == shooter
                    ):
                        step_i -= 1
                    if (
                        step_i >= 0
                        and self.ball_tracer[step_i][0] == 0
                        and self.ball_tracer[step_i][1] in self.player_ids
                    ):
                        self.stats["assist"][
                            self.player_ids_2_agent_ids[self.ball_tracer[step_i][1]]
                        ] += 1
                        
                        try:
                            self.stats["shoot"][self.player_ids_2_agent_ids[shooter]] += 1
                            self.cood_pair.append((self.player_ids_2_agent_ids[shooter], self.player_ids_2_agent_ids[self.ball_tracer[step_i][1]]))
                            print(self.cood_pair)
                        except:
                            pass

    def get_stats(self) -> Dict[str, List]:
        ret_stats = {}
        for k, v in self.stats.items():
            ret_stats[k] = sum(v)
        ret_stats["succ pass rate"] = (
            ret_stats["succ pass"] / ret_stats["pass"] if ret_stats["pass"] > 0 else 0
        )
        ret_stats["succ serve rate"] = (
            ret_stats["succ serve"] / ret_stats["serve"]
            if ret_stats["serve"] > 0
            else 0
        )
        if len(self.cood_pair) > 0:
            ret_stats["cood_pair"] = self.cood_pair.copy()
        return ret_stats

    def reset(self):
        self.stats = {
            "assist": [0 for _ in range(self.num_agents)],
            "succ pass": [0 for _ in range(self.num_agents)],
            "pass": [0 for _ in range(self.num_agents)],
            "possession": [0 for _ in range(self.num_agents)],
            "succ serve": [0 for _ in range(self.num_agents)],
            "serve": [0 for _ in range(self.num_agents)],
            "shoot": [0 for _ in range(self.num_agents)],
        }
        self.cood_pair = []
