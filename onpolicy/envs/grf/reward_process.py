from typing import Dict
import numpy as np
import torch


class Rewarder:
    def __init__(self, reward_config: Dict = None) -> None:
        self.reward_config = reward_config

    def calc_reward(self, rew, prev_obs, obs):
        reward = 0
        if self.reward_config:
            for sub_reward, weight in self.reward_config.items():
                if sub_reward == "score":
                    reward += weight * rew
                else:
                    reward += weight * eval("_" + sub_reward)(prev_obs, obs)
        else:
            reward = rew

        return reward

def _goal_reward(prev_obs, obs):
    [my_score_prev, opponent_score_prev] = prev_obs["score"]
    [my_score_after, opponent_score_after] = obs["score"]
    rew = 0
    if my_score_after > my_score_prev:
        rew += 1
    if opponent_score_after > opponent_score_prev:
        rew -= 1
    return rew


def _ball_status_reward(prev_obs, obs):
    r = 0
    if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
        r -= 0.1
    elif prev_obs["ball_owned_team"] == 1 and obs["ball_owned_team"] == 0:
        r += 0.1
    return r


def _yellow_reward(prev_obs, obs):
    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow
    return yellow_r


def _ball_min_dist_reward(prev_obs, obs):
    if obs["ball_owned_team"] != 0:
        ball_position = np.array(obs["ball"][:2])
        left_team_position = obs["left_team"][1:]
        left_team_dist2ball = np.linalg.norm(left_team_position - ball_position, axis=1)
        min_dist2ball = np.min(left_team_dist2ball)
    else:
        min_dist2ball = 0.0
    return -min_dist2ball


def _win_reward(prev_obs, obs):
    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = my_score - opponent_score
    return win_reward


def _loss_penalty(prev_obs, obs):
    loss_penalty = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score <= opponent_score:
            loss_penalty = -1.0
    return loss_penalty


def _call_ball_position_reward(obs):
    ball_x, ball_y, _ = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):  # in our penalty area
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):  #
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    return ball_position_r


def _ball_position_reward(prev_obs, obs):
    prev_ball_pos_r = _call_ball_position_reward(prev_obs)
    ball_pos_r = _call_ball_position_reward(obs)
    return ball_pos_r - prev_ball_pos_r


def _ball_position_reward2(prev_obs, obs):
    ball_pos_r = _call_ball_position_reward(obs)
    return ball_pos_r


def _ball_dist_to_goal(prev_obs, obs):
    ball_position = np.array(obs["ball"][:2])
    ball_dist_to_goal = np.linalg.norm(np.array([-1, 0]) - ball_position)
    ball_dist_to_goal -= np.linalg.norm(np.array([1, 0]) - ball_position)
    return ball_dist_to_goal
