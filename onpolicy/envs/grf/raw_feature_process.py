import numpy as np
from gym.spaces import Box
from typing import List, Dict


class FeatureEncoder:
    def __init__(
        self,
        num_left_agents,
        num_right_agents,
        action_n,
        avail_in_feature: bool = False,
        obs_agent_id: bool = False,
        obs_last_action: bool = False,
        obs_match: bool = False,
        left_steps_interval: int = 500,
        game_length: int = 3000,
    ):
        self.active = -1
        self.left_steps_interval = left_steps_interval
        self.game_length = game_length
        self.left_steps_n = int(np.ceil(self.game_length / self.left_steps_interval))
        self.action_n = action_n
        self.num_left_agents = num_left_agents
        self.num_right_agents = num_right_agents
        self.avail_in_feature = avail_in_feature
        self.obs_agent_id = obs_agent_id
        self.obs_last_action = obs_last_action
        self.obs_match = obs_match

    def get_feature_dims(self):
        dims = {
            "player": 19 if not self.obs_agent_id else 19 + self.num_left_agents,
            "ball": 18,
            "left_team": 7,
            "left_team_closest": 7,
            "right_team": 7,
            "right_team_closest": 7,
            "match": 9 + self.left_steps_n,
        }
        return dims

    def encode(self, states: List[Dict]):
        feats = []
        avails = []
        for state in states:
            feat, avail = self.encode_each(state)
            feats.append(feat)
            avails.append(avail)
        return feats, avails

    @property
    def observation_space(self):
        feature_dims = self.get_feature_dims()
        size = (
            feature_dims["player"]
            + feature_dims["ball"]
            + feature_dims["left_team"] * (self.num_left_agents - 1)
            + feature_dims["left_team_closest"]
            + feature_dims["right_team"] * self.num_right_agents
            + feature_dims["right_team_closest"]
        )
        if self.avail_in_feature:
            size += self.action_n
        if self.obs_last_action:
            size += self.action_n
        if self.obs_match:
            size += feature_dims["match"]

        # print("size", size)
        return Box(low=float("-inf"), high=float("inf"), shape=(size,))

    def encode_each(self, obs: Dict):
        player_num = obs["active"]

        player_pos_x, player_pos_y = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs["left_team_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs["left_team_tired_factor"][player_num]
        is_dribbling = obs["sticky_actions"][9]
        is_sprinting = obs["sticky_actions"][8]

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self.get_available_actions(obs, ball_distance)

        # size 2+2+1+10+4=19
        player_state = np.concatenate(
            (
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        if self.obs_agent_id:
            agent_id = np.eye(self.num_left_agents)[player_num]
            player_state = np.concatenate((player_state, agent_id))

        # size 3+6+2+3+4=18
        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array(obs["ball_direction"]) * 20,
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )

        obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
        obs_left_team_direction = np.delete(
            obs["left_team_direction"], player_num, axis=0
        )
        left_team_relative = obs_left_team
        left_team_distance = np.linalg.norm(
            left_team_relative - obs["left_team"][player_num], axis=1, keepdims=True
        )
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(
            obs["left_team_tired_factor"], player_num, axis=0
        ).reshape(-1, 1)
        # (self.num_left_agents - 1) * (2+2+1+1+1)
        left_team_state = np.concatenate(
            (
                left_team_relative * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance * 2,
                left_team_tired,
            ),
            axis=1,
        )
        left_closest_idx = np.argmin(left_team_distance)
        # 2+2+1+1+1 = 7
        left_closest_state = left_team_state[left_closest_idx]

        obs_right_team = np.array(obs["right_team"])
        obs_right_team_direction = np.array(obs["right_team_direction"])
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["left_team"][player_num], axis=1, keepdims=True
        )
        right_team_speed = np.linalg.norm(
            obs_right_team_direction, axis=1, keepdims=True
        )
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        right_team_state = np.concatenate(
            (
                obs_right_team * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance * 2,
                right_team_tired,
            ),
            axis=1,
        )
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        match_state = np.concatenate(
            [
                obs["score"],
                self._encode_left_steps_onehost(obs["steps_left"]),
                np.eye(7)[obs["game_mode"]],
            ]
        ).astype(np.float)

        state_dict = {
            "player": player_state,
            "ball": ball_state,
            "left_team": left_team_state,
            "left_closest": left_closest_state,
            "right_team": right_team_state,
            "right_closest": right_closest_state,
        }
        # for k, v in state_dict.items():
        #     print(k, v.shape)
        if self.avail_in_feature:
            state_dict.update({"player": np.concatenate((player_state, avail))})

        if self.obs_match:
            state_dict.update({"match": match_state})
            # print(match_state)

        feats = np.hstack(
            [
                np.array(state_dict[k], dtype=np.float32).flatten()
                for k in sorted(state_dict)
            ]
        )

        return feats, avail

    def get_available_actions(self, obs, ball_distance):
        # todo further restrict it by automata
        avail = self._get_avail(obs, ball_distance, self.action_n)
        return avail

    def _get_avail(self, obs, ball_distance, action_n):
        assert action_n == 19 or action_n == 20  # we don't support full action set
        avail = [1] * action_n

        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        if action_n == 20:
            BUILTIN_AI = 19

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
            if ball_distance > 0.03:
                avail[SLIDE] = 0
        elif (
            obs["ball_owned_team"] == -1
            and ball_distance > 0.03
            and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
                avail[SLIDE],
            ) = (0, 0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0
            if ball_distance > 0.03:
                (
                    avail[LONG_PASS],
                    avail[HIGH_PASS],
                    avail[SHORT_PASS],
                    avail[SHOT],
                    avail[DRIBBLE],
                ) = (0, 0, 0, 0, 0)

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (
            -0.27 <= ball_y and ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1] + [0] * (action_n - 1)
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1] + [0] * (action_n - 1)
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1] + [0] * (action_n - 1)
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def _encode_role_onehot(self, role_num):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[role_num] = 1.0
        return np.array(result)

    def _encode_left_steps_onehost(self, left_steps):
        result = [0 for _ in range(self.left_steps_n)]
        for i in reversed(range(self.left_steps_n)):
            if left_steps >= i * self.left_steps_interval:
                result[i] = 1.0
                break
        return result
