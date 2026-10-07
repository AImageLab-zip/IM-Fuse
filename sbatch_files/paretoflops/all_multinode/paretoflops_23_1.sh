#!/bin/bash
#SBATCH --job-name=PARETOFLOPS23_FOLD_1_MN
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoflops_23_1_mn.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoflops_23_1_mn.out
#SBATCH --gres=gpu:2
#SBATCH --account=grana_neuro
#SBATCH --constraint=gpu_RTXPro6000B_96G

# Multi-node variant of ../all/paretoflops_23_1.sh: 2 nodes x 2 GPUs = 4 ranks.
# --nodes=2 is ONE allocation, so Slurm starts both nodes together -- there is no
# race between separately submitted jobs. --ntasks-per-node=1 puts exactly one
# launcher on each node (an unqualified --ntasks=2 could stack both on one node);
# each launcher then relaunches itself under torchrun for its 2 local GPUs.

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Rendezvous on the first node of the allocation. The port is derived from the
# job id so two concurrent jobs sharing a node cannot collide on it.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# Re-run of Phase 4 with the budget denominated in MACs instead of parameters,
# over a 229x range (8.341 GMACs .. 1.9099 TMACs) with sandwich sampling.
# See src/mimose/data/configs/pareto_flops_23.yaml for why.
#
# NOTE: --batch-size is per rank, so the effective batch here is 4x the value in
# the config (vs 2x in the single-node script). Adjust the LR accordingly.
#
# WARNING: --run-suffix fold1 is the same as ../all/paretoflops_23_1.sh, so both
# scripts write to the same art_dir and --try-resume will pick up whichever
# checkpoint is there. Run one or the other, or change the suffix here -- the
# two differ in effective batch size and are not the same experiment.
srun mimose train --config pareto_flops_23.yaml --fold 1 --run-suffix fold1 --seed 0 --try-resume --distributed --nnodes 2

# Testing is single-process and runs on the first node only -- no srun.
#
# As in pareto_23_1.sh: this evaluates the supernet at its DEFAULT budget only,
# i.e. the widest subnetwork (1.9099 TMACs). One operating point, not the family.
# The dense sweep over the range is the flops_budgets / test_budgets pair, which
# must be regenerated for this run -- its budgets are MACs now, not parameters.
mimose test --config pareto_flops_23.yaml --fold 1 --run-suffix fold1 --precision bf16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from the singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
