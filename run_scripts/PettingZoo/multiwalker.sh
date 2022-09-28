#!/bin/sh
env="PettingZoo"
scenario="multiwalker"
num_agents=3
algo="rmappo"
exp="test"
seed_begin=1
seed_end=1

gpu=0
n_t=20

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
    CUDA_VISIBLE_DEVICES=${gpu} python onpolicy/scripts/train/train_pettingzoo.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --scenario_name ${scenario} --seed ${seed} --n_rollout_threads ${n_t} --episode_length $((5000/$n_t)) --use_eval True --n_run 1 --use_aim True --aim_repo .aim_pettingzoo  --ppo_epoch 10 --use_sequential True --adv gae_trace --loop_order ppo --use_two_stage True --clip_param_tuner True --clip_param_weight_rp True --seq_strategy semi_greedy --gae_lambda 0.95 --share_policy True --use_agent_block True --block_num 3 --use_cum_sequence True --clip_others True --lr 1.0e-4 --critic_lr 1.0e-4 --clip_param 0.1 --data_chunk_length 10
done
# --ppo_epoch 8 --use_sequential True --adv gae_trace --loop_order agent --use_two_stage True --clip_param_tuner True --gae_lambda 0.93 --use_agent_block False --clip_others True
# --ppo_epoch 10


# shared policy
# --ppo_epoch 10 --use_sequential True --adv gae_trace --loop_order ppo --use_two_stage True --clip_param_tuner True --clip_param_weight_rp True --seq_strategy semi_greedy --gae_lambda 0.95 --share_policy True --use_agent_block True --block_num 3 --use_cum_sequence True --clip_others True

# --ppo_epoch 10 --share_policy True --lr 1.0e-4 --critic_lr 1.0e-4 --clip_param 0.1