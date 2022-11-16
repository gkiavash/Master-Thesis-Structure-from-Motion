#!/bin/bash
#SBATCH --job-name pixsfm

#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH --mail-user gkiavash@gmail.com
#SBATCH --mail-type ALL

#SBATCH --time 12:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups

#SBATCH --mem 32G
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1

cd $WORKING_DIR
#your working directory

srun singularity exec --writable --nv Master-Thesis-Structure-from-Motion/sif_files/pixsfm_all.sif /bin/bash Master-Thesis-Structure-from-Motion/scripts/colmap_pixsfm.sh Master-Thesis-Structure-from-Motion/colmap_project Master-Thesis-Structure-from-Motion/scripts/pixsfm_existing_db.py
