#!/bin/bash
#SBATCH --account=p31841 ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=gengpu-long  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --time=48:00:00 ## how long does this need to run (remember different partitions have restrictions on this para
#SBATCH --mem=120G ## how much RAM do you need per CPU (this effects your FairShare$
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nikolay.markov@northwestern.edu
#SBATCH --gres=gpu:a100:1

source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate /projects/b1196/envs/serniczek

cd /projects/b1196/ewa_group/serniczek/01_ehrformer
python EHRFormer_pretrain.py
