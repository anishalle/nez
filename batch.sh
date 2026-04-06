#!/bin/bash
#SBATCH -p a30-4.6gb 
#SBATCH -c 216
#SBATCH --mem=68G
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --mail-user=AXA230262@utdallas.edu
#SBATCH -J nez
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

  #srun -p h100 -N 1 -c 32 --mem=244G --gpus=1 --pty bash
set -euo pipefail

#a30-4.6gb
#CfgTRES=cpu=256,mem=1000G,billing=256,gres/gpu=8,gres/gpu:nvidia_a30_1g.6gb=8
   #AllocTRES=cpu=40,mem=932G,gres/gpu=1,gres/gpu:nvidia_a30_1g.6gb=1
cd '/home/axa230262/work/000 me/nez'
source .venv/bin/activate
python src/ssl_bot.py
