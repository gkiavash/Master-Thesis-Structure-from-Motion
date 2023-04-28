# Setup environments

Since singularity provides the environments, and it uses docker images, all environments 
are implemented and tested in docker files in `/docker` directory


# Singularity
### build sif:

Build from .dif file:

`sudo singularity build colmap_local.sif colmap_test.dif`

Build directly from a docker image:

`sudo singularity build --sandbox pixsfm_raw.sif docker://gkiavash/pixsfm:1.0.3`

Convert to sandbox

`sudo singularity build --sandbox output_sif/ input.sif`


Run container interactively:

`singularity shell pixsfm_raw.sif`

### inside container

script pixsfm:

`python3 Master-Thesis-Structure-from-Motion/scripts/pixsfm_existing_db.py  $DATASET_PATH t1 kkk kkk`


# GPMF-parser

`docker run -v "/home/gkiavash/Downloads/sfm_street_1/GH010024.MP4:/input.mp4" runsascoded/gpmf-parser --entrypoint "./run.sh -h"`

`ffmpeg -y -i GH010024.MP4 -codec copy -map 0:m:handler_name:" GoPro MET" -f rawvideo GH010024.bin`

# Cluster DEI

`ssh ghamsariki@login.dei.unipd.it`

Run job file: 

`sbatch /to/job/path`

`sbatch Master-Thesis-Structure-from-Motion/docker/pixsfm_job.sh `

Check job status: `squeue`


# Others

In order to limit the resources for specific linux command:

`systemd-run --scope -p MemoryLimit=4096M -p CPUQuota=60% make`


# Calibration

`docker run -it -v /home/gkiavash/Downloads/sfm_projects/datasets/cakibration_2:/input gkiavash/calib:0.0.1 ../bin/cam_calib -f /input -c /input/out.yaml --bw 8 --bh 6 -q 0.4 -k -s 4 -u --opencv_format`

in  singularity 

`/cv_ext/bin/cam_calib -f /home/gkiavash/Downloads/sfm_projects/datasets/cakibration_2/ -c /home/gkiavash/Downloads/sfm_projects/datasets/cakibration_2/out.yaml --bw 8 --bh 6 -q 0.4 -k -s 4 -u --opencv_format`



# Steps to create a new singularity in ClusterDei:

1. Create docker file and pushed to Docker Hub
2. In Ubuntu VM, build the singularity file WITHOUT sandbox: `sudo singularity build pixsfm_raw_no_sandbox.sif docker://gkiavash/pixsfm:1.0.4`
3. ssh copy the file to ClusterDei: 
   `
   sudo scp pixsfm_raw_.sif ghamsariki@login.dei.unipd.it:/home/ghamsariki/Master-Thesis-Structure-from-Motion/sif_files/
   `
4. Convert it to sandbox sif file by using sbatch file
