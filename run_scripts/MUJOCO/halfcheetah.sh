#!/bin/sh

export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="2x3"
agent_obsk=-1
algo="rmappo"
exp="test"
seed_begin=1
seed_end=1

gpu=0
n_t=16

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
    CUDA_VISIBLE_DEVICES=${gpu} python onpolicy/scripts/train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --seed ${seed} --n_rollout_threads ${n_t} --episode_length $((4000 / $n_t)) --use_eval True --n_run 1 --use_aim True --aim_repo .aim_mujoco --ppo_epoch 5 --use_sequential True --loop_order agent --seq_strategy random --clip_others False --use_two_stage True
done

# mappo
# --ppo_epoch 5

# CoPPO
# --ppo_epoch 5 --use_MA_ratio True --clip_others True

# HAPPO
# --ppo_epoch 5 --use_sequential True --loop_order agent --seq_strategy random --clip_others False

# A2PO
# --ppo_epoch 5 --use_sequential True --loop_order agent --seq_strategy semi_greedy --adv gae_trace --gae_lambda 0.93 --clip_param_tuner True --use_two_stage True --clip_others True