# Singularity
### build sif:

Build from .dif file

`sudo singularity build colmap_local.sif colmap_test.dif`

Build directly from a docker image

`sudo singularity build colmap_local.sif docker-daemon://gkiavash/colmap:0.0.1`


# GPMF-parser

`docker run -v "/home/gkiavash/Downloads/sfm_street_1/GH010024.MP4:/input.mp4" runsascoded/gpmf-parser --entrypoint "./run.sh -h"`

`ffmpeg -y -i GH010024.MP4 -codec copy -map 0:m:handler_name:" GoPro MET" -f rawvideo GH010024.bin`

# Cluster DEI

`ssh ghamsariki@login.dei.unipd.it`


# Others

`systemd-run --scope -p MemoryLimit=4096M -p CPUQuota=60% make`
