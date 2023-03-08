DATASET_PATH=$1
is_distorted=$2
max_num_features=$3


# init
slam() {
  max_num_features=2000
  sh /home/ghamsariki/Master-Thesis-Structure-from-Motion/scripts/colmap_sparse.sh $DATASET_PATH $is_distorted $max_num_features
}

slam_to_sfm() {
  colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0 \
    --output_type TXT

  rm $DATASET_PATH/sparse/0/*.bin
  echo $DATASET_PATH/sparse/0/points3D.txt
  python3 /home/ghamsariki/Master-Thesis-Structure-from-Motion/scripts/utils/colmap_pose.py $DATASET_PATH/sparse/0/images.txt $DATASET_PATH/sparse/0/images.txt
}

sfm() {
  echo "going to sfm"
}

slam
slam_to_sfm
sfm
