#!/usr/bin/env bash
#SBATCH --job-name=kvret-calib
#SBATCH --output=hpc_calib_%j.out
#SBATCH --error=hpc_calib_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
##SBATCH --partition=FILL_ME_IN
##SBATCH --account=FILL_ME_IN

# Slurm wrapper. All the work is in hpc/run_calibration.sh, which knows nothing about a
# scheduler; a PBS or LSF site replaces this header and calls the same body.
#
# --time is a ceiling, not a booking. The job is billed on the wall clock it actually
# uses, and run_calibration.sh kills vLLM from an EXIT trap on every path, so asking for
# three hours costs nothing if the work takes two and a half. Asking for too LITTLE is the
# expensive mistake: a job killed mid-validation has spent the money and produced nothing.
#
# The two commented ##SBATCH lines above are commented on purpose. Uncomment and fill them
# after checking `sinfo` and `sacctmgr show assoc user=$USER` -- guessing a partition name
# gets the job rejected at submit time, which is harmless, but guessing an account can
# charge the wrong budget, which is not.
#
#   sbatch --export=ALL,PROJECT=$HOME/agent-kv-retention,VENV=$HOME/venv-vllm,\
# EXPECT_CAPABILITY=7.5,RM_PER_HOUR=3.06 hpc/calibrate.sbatch
#
# EXPECT_CAPABILITY must match the node the partition actually hands out: 7.5 for T4
# (Turing), 8.9 for L40S/L4 (Ada). It is asserted, not assumed -- a mismatch aborts before
# any measurement, because a constant fitted on an unexpected architecture is worse than no
# constant at all.

set -uo pipefail

# Site module setup. Comment out or replace: on a cluster with no module system these are
# errors, and on one with modules the names are local.
if command -v module > /dev/null 2>&1; then
  module purge
  # module load cuda/12.8
  # module load python/3.12
  :
fi

echo "node        : $(hostname)"
echo "job id      : ${SLURM_JOB_ID:-none}"
echo "started     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo

bash "${PROJECT:?set PROJECT}/hpc/run_calibration.sh"
