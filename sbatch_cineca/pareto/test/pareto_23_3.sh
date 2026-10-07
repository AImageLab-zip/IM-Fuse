#!/bin/bash
#SBATCH --job-name=PARETO23_TEST_BUDGETS_FOLD_3
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/pareto_23_3_test_budgets.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/pareto_23_3_test_budgets.out
#SBATCH --gres=gpu:1
#SBATCH --account=IscrB_MEDSYN

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

# Evaluates the pareto_23 supernet at a fixed set of parameter budgets, one at
# a time (--budget selects the subnetwork; see ElasticUnet.set_budget). Each
# budget writes into its own "pareto_b<TAG>_18" results directory -- matching
# the "<model>_<dsnum>" layout build_reproduction_comparison.py's discover()
# scans for -- so the recap/summary scripts pick up each budget as its own
# row, the same way they already do for the mimosa_[size] sweep.
#
# Range mirrors the supernet's stated span for (64,128,256,512,1024):
# ~1.43M .. ~91.4M params (pareto_23.yaml). Edit BUDGETS to add/remove points.
BUDGETS=(1430000 5000000 10000000 20000000 45000000 91400000)

FOLD=3
CONFIG=pareto_23.yaml

for BUDGET in "${BUDGETS[@]}"; do
    TAG=$(python3 -c "print(f'{${BUDGET}/1e6:.1f}M')")
    mimose test \
        --config "$CONFIG" \
        --fold "$FOLD" \
        --run-suffix "fold${FOLD}" \
        --budget "$BUDGET" \
        --output-path "/work/phd_mimose/results/pareto_b${TAG}_23/results.txt" \
        --precision fp16
done
