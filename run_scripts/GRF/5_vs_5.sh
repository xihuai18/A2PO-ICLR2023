#!/bin/sh

env="grf"
scenario="5_vs_5_easy"
algo="rmappo"
gpu=0
n_t=24
ep_len=3000
exp="test"
seed_begin=1
seed_end=1

if [ $# -ge 1 ]
then
    gpu=$1
fi

# if [ $# -ge 2 ]
# then
#     n_t=$2
# fi

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_end}"
for seed in `seq ${seed_begin} ${seed_end}`;
do
    echo "seed is ${seed} use gpu ${gpu}"
    set -x
    CUDA_VISIBLE_DEVICES=${gpu} python3 onpolicy/scripts/train/train_grf.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --seed ${seed} --n_rollout_threads ${n_t} --episode_length ${ep_len} --use_eval True --n_run 1 --use_aim True --aim_repo .aim_grf \
    --action_set "default" \
    --reward_shaping True --rewards "scoring" \
    --ppo_epoch 10 --use_sequential True --loop_order agent --seq_strategy semi_greedy --adv gae_trace --gae_lambda 0.95 --clip_param_tuner True --use_two_stage True --clip_others True --use_agent_block False --share_policy False --use_cum_sequence False --log_agent_order True \
    # --ppo_epoch 12 --use_sequential True --loop_order ppo --seq_strategy random --clip_others False --clip_before_prod True --share_policy True --use_cum_sequence True --use_agent_block True --block_num 6 \
    # --ppo_epoch 10 --share_policy True \
    # --ppo_epoch 10 --use_MA_ratio True --clip_others True --share_policy True \
    # --ppo_epoch 10 --share_policy False \
    # --ppo_epoch 10 --use_sequential True --loop_order agent --seq_strategy semi_greedy --adv gae_trace --gae_lambda 0.95 --clip_param_tuner True --use_two_stage True --clip_others True --use_agent_block False --share_policy False --use_cum_sequence False \
    # --ppo_epoch 10 --use_sequential True --loop_order agent --seq_strategy random --clip_others False --use_agent_block False --clip_before_prod True --share_policy False --use_cum_sequence False \
    # --ppo_epoch 10 --use_MA_ratio True --clip_others True --share_policy False \
    # --obs_match True \
    # --avail_in_feature True --obs_last_action True \
done

# share policy

# mappo
# --ppo_epoch 10 --share_policy True

# CoPPO
# --ppo_epoch 10 --use_MA_ratio True --clip_others True --share_policy True

# HAPPO
# --ppo_epoch 10 --use_sequential True --loop_order ppo --seq_strategy random --clip_others False --clip_before_prod True --share_policy True --use_cum_sequence True --use_agent_block True --block_num 5

# A2PO
# --ppo_epoch 10 --use_sequential True --loop_order ppo --seq_strategy semi_greedy --adv gae_trace --gae_lambda 0.95 --clip_param_tuner True --clip_param_weight_rp True --use_two_stage True --clip_others True --share_policy True --use_cum_sequence True --use_agent_block True --block_num 5

# individual policy

# mappo
# --ppo_epoch 10 --share_policy False

# CoPPO
# --ppo_epoch 10 --use_MA_ratio True --clip_others True --share_policy False

# HAPPO
# --ppo_epoch 10 --use_sequential True --loop_order agent --seq_strategy random --clip_others False --use_agent_block False --clip_before_prod True --share_policy False --use_cum_sequence False

# A2PO
# --ppo_epoch 10 --use_sequential True --loop_order agent --seq_strategy semi_greedy --adv gae_trace --gae_lambda 0.95 --clip_param_tuner True --use_two_stage True --clip_others True --use_agent_block False --share_policy False --use_cum_sequence False