# Singularity
### build sif:

Build from .dif file:

`sudo singularity build colmap_local.sif colmap_test.dif`

Build directly from a docker image:

`sudo singularity build colmap_local.sif docker-daemon://gkiavash/colmap:0.0.1`

`sudo singularity build --sandbox pixsfm_raw.sif docker://gkiavash/pixsfm:1.0.3`

Convert to sandbox

`singularity build --sandbox lolcow_sandbox/ lolcow.sif`


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


Run sbatch

`sbatch Master-Thesis-Structure-from-Motion/docker/pixsfm_job.sh `


# Pixel-Perfect SfM

To download weights for cnn
`
from pixsfm.features.models import s2dnet
s2dnet.S2DNet(conf={})
`


# Others

`systemd-run --scope -p MemoryLimit=4096M -p CPUQuota=60% make`

/bin/bash Master-Thesis-Structure-from-Motion/scripts/colmap_pixsfm.sh Master-Thesis-Structure-from-Motion/colmap_project Master-Thesis-Structure-from-Motion/scripts/pixsfm_existing_db.py