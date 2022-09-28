#!/bin/sh
env="PettingZoo"
scenario="simple_reference"
algo="rmappo"
exp="test"
seed_begin=1
seed_end=1

gpu=0
n_t=32

if [ $# -ge 1 ]
then
    gpu=$1
fi

if [ $# -ge 2 ]
then
    n_t=$2
fi

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_end}"
for seed in `seq ${seed_begin} ${seed_end}`;
do
    echo "seed is ${seed} use gpu ${gpu}"
    set -x
    CUDA_VISIBLE_DEVICES=${gpu} python onpolicy/scripts/train/train_pettingzoo.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --seed ${seed} --n_rollout_threads ${n_t} --num_agents 2 --episode_length $((800/$n_t)) --use_eval True --n_run 1 --use_aim True --aim_repo .aim_pettingzoo --ppo_epoch 10 --use_sequential True --loop_order agent --adv gae_trace --gae_lambda 0.95 --use_two_stage True --clip_param_tuner True
done

# mappo
# --ppo_epoch 10

# CoPPO
# --ppo_epoch 10 --use_MA_ratio True --clip_others True

# HAPPO
# --ppo_epoch 10 --use_sequential True --loop_order agent --seq_strategy random --clip_others False

# A2PO
# --ppo_epoch 10 --use_sequential True --loop_order agent --adv gae_trace --gae_lambda 0.95 --use_two_stage True --clip_param_tuner True