#!/bin/bash
#SBATCH --job-name pixsfm

#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH --mail-user gkiavash@gmail.com
#SBATCH --mail-type ALL

#SBATCH --time 12:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups

#SBATCH --mem 128G
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:rtx

cd $WORKING_DIR
#your working directory

srun singularity exec --writable --nv Master-Thesis-Structure-from-Motion/sif_files/pixsfm_1_0_4.sif python3 Master-Thesis-Structure-from-Motion/experiments/sfm_pixsfm/run.py /home/ghamsariki/Master-Thesis-Structure-from-Motion/pixsfm_project
