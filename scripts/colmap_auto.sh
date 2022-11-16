DATASET_PATH=$1

colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images \
    --data_type video \
    --quality extreme \
    --single_camera 1
