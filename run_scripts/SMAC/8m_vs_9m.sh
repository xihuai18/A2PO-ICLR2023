#!/bin/sh
env="StarCraft2"
map="8m_vs_9m"
algo="rmappo"
exp="test"
seed_begin=1
seed_end=1

gpu=0
n_t=10

if [ $# -ge 1 ]
then
    gpu=$1
fi

if [ $# -ge 2 ]
then
    n_t=$2
fi

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed in ${seeds[@]}"
# for seed in `seq ${seed_begin} ${seed_end}`;
for seed in ${seeds[@]};
do
    echo "seed is ${seed} use gpu ${gpu}"
    set -x
    CUDA_VISIBLE_DEVICES=${gpu} python onpolicy/scripts/train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_rollout_threads ${n_t} --episode_length $((3200 / $n_t)) --use_eval True --use_aim True --aim_repo .aim_smac --n_run 1 \
    --ppo_epoch 15 --use_sequential True --loop_order ppo --seq_strategy random --clip_others False \
    --clip_param_tuner True --clip_param_weight_rp True 
done
# mappo
# --ppo_epoch 15

# CoPPO
# --ppo_epoch 15 --use_MA_ratio True --clip_others True

# HAPPO
# --ppo_epoch 15 --use_sequential True --loop_order ppo --seq_strategy random --clip_others False --clip_before_prod True

# A2PO
# --ppo_epoch 15 --use_sequential True --adv gae_trace --gae_lambda 0.95 --loop_order ppo --seq_strategy semi_greedy --use_two_stage True --clip_param_tuner True --clip_param_weight_rp True --clip_others True 