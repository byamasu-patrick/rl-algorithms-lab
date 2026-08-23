#!/bin/bash
set -Eeuo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <num_instances> <current_instance_index>"
  exit 1
fi

num_instances=$1
cur_instance=$2

all_commands=()
for seed in {1..10}; do
    for env in CartPole-v1 Acrobot-v1 MountainCarContinuous-v0 HalfCheetah-v4 Humanoid-v4; do
        #### D1: Baselines

        # Standard PPO
        all_commands+=("ppo__${env}__${seed} --env-id ${env} --seed ${seed}")

        for num_steps in 0 128; do
            # PPO with no baselines / REINFORCE with Off Policy Correction
            all_commands+=("ppo_nobaseline-h${num_steps}__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn")
            all_commands+=("ppo_nobaseline-h${num_steps}-y1__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --gamma 1")

            # PPO + group stats random baseline
            all_commands+=("ppo_rstats-h${num_steps}__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type stats")
            all_commands+=("ppo_rstats-h${num_steps}-y1__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type stats --gamma 1")

            # PPO + mean of iteration returns baseline
            all_commands+=("ppo_batchmean-h${num_steps}__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type batch_mean")
            all_commands+=("ppo_batchmean-h${num_steps}-y1__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type batch_mean --gamma 1")

            # PPO + constant baseline and scaling / normalization (GRPO with batch group)
            all_commands+=("grpo_batch-h${num_steps}_g8_y0.99__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type batch_mean --scale-adv-batch")

            # PPO + EMA of episodic returns
            all_commands+=("ppo_ema-h${num_steps}__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type ema")
            all_commands+=("ppo_ema-h${num_steps}-y1__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type ema --gamma 1")
        done

        #### D2: Gamma / Horizon
        for num_steps in 0 128; do
            for gamma in 0 0.1 0.5 0.9 0.95 0.99 1; do
                # PPO
                all_commands+=("ppo-h${num_steps}_y${gamma}__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --gamma ${gamma}")
                # GRPO
                all_commands+=("grpo_batch-h${num_steps}_g8_y${gamma}__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type batch_mean --scale-adv-batch --gamma ${gamma}")
            done
        done

        #### D3: Group Sampling
        for num_steps in 0 128; do
            for num_envs in 8 16 32 64 128; do
                # GRPO
                all_commands+=("grpo_batch-h${num_steps}_g${num_envs}_y0.99__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type batch_mean --scale-adv-batch --num-envs ${num_envs}")
                all_commands+=("grpo_batch-h${num_steps}_g${num_envs}_y1__${env}__${seed} --env-id ${env} --seed ${seed} --num-steps ${num_steps} --return-type mc --no-use-value-fn --baseline-type batch_mean --scale-adv-batch --num-envs ${num_envs} --gamma 1")
            done
        done
    done
done

# Pre-filter experiments based on instance assignment.
logdirs=()
commands=()
for idx in "${!all_commands[@]}"; do
  cmd="${all_commands[$idx]}"
  # Extract logdir from the command string
  logdir="runs/$(echo "$cmd" | cut -d' ' -f1)"
  if (( idx % num_instances == cur_instance )); then
    if [ -d "$logdir" ]; then
      if [ -f "$logdir/config.yaml" ]; then
        echo "Skipping $logdir (already exists)"
        continue
      fi
    fi
    logdirs+=("$logdir")
    commands+=("uv run python algorithm.py --run-name ${cmd}")
  fi
done

echo "About to run ${#logdirs[@]}/${#all_commands[@]} experiments."

# Iterate over the filtered experiments.
for idx in "${!logdirs[@]}"; do
  logdir="${logdirs[$idx]}"
  cmd="${commands[$idx]}"
  echo "Running experiment in $logdir"
  echo "Command: $cmd"
  mkdir -p "$logdir"
  # Launch and log output
  "$SHELL" -li -c "$cmd 2>&1 | tee \"$logdir/run.log\""
  sleep 1
done