#!/bin/bash
#SBATCH --job-name=wan-chai-kv-diagnostics-step2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=wan-chai-kv-diagnostics-step2-%j.out

set -euo pipefail
exec bash "$HOME/cp4101/sglang/slurm/week12/experiment2/wan_chai_kv_diagnostics.sh" step2
