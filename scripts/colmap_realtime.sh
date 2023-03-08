DATASET_PATH=$1
is_distorted=$2
max_num_features=$3


POSE_MODEL_PATH=$DATASET_PATH/sparse/model

# init
slam() {
  max_num_features=2000
  sh /home/ghamsariki/Master-Thesis-Structure-from-Motion/scripts/colmap_sparse.sh $DATASET_PATH $is_distorted $max_num_features 0
}

slam_to_sfm() {
  colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $POSE_MODEL_PATH \
    --output_type TXT

  echo $POSE_MODEL_PATH/points3D.txt
  python3 /home/ghamsariki/Master-Thesis-Structure-from-Motion/scripts/utils/colmap_pose.py $POSE_MODEL_PATH/images.txt $POSE_MODEL_PATH/images.txt
  rm -f $DATASET_PATH/database.db
}

sfm() {
  echo "going to sfm"
  max_num_features=32000
  sh /home/ghamsariki/Master-Thesis-Structure-from-Motion/scripts/colmap_sparse.sh $DATASET_PATH $is_distorted $max_num_features 1
}

FILE=$DATASET_PATH/sparse/0/images.bin

if [ ! -f "$FILE" ]; then
  slam
fi

slam_to_sfm
sfm
