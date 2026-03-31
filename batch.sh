#!/bin/bash
#SBATCH -p dev
#SBATCH -c 64
#SBATCH --mem=375G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=AXA230262@utdallas.edu
#SBATCH -J example_job
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

source .venv/bin/activate
python src/example.py
