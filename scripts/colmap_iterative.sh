DATASET_PATH=$1


colmap_sparse() {
  colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --SiftExtraction.max_image_size 10000 \
    --SiftExtraction.max_num_features 50000 \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model FULL_OPENCV \
    --ImageReader.camera_params $CALIB_PARAMS_YAML_4 \
    --image_list_path $DATASET_PATH/image-list.txt

  colmap sequential_matcher \
    --database_path $DATASET_PATH/database.db \
    --SiftMatching.max_num_matches 32000

  colmap image_registrator \
      --database_path $DATASET_PATH/database.db \
      --input_path $DATASET_PATH/sparse \
      --output_path $DATASET_PATH/sparse \
      --Mapper.ba_refine_principal_point 1

  colmap bundle_adjuster \
      --input_path $DATASET_PATH/sparse \
      --output_path $DATASET_PATH/sparse
}

# 1) start with first images
sh /home/ghamsariki/Master-Thesis-Structure-from-Motion/scripts/colmap_sparse.sh $DATASET_PATH


# 2) add new images iteratively
for FILE in $DATASET_PATH/incoming/*; do
  cp $FILE $DATASET_PATH/images/

  echo "$(basename $FILE)"
  echo "$(basename $FILE)" >> $DATASET_PATH/image-list.txt

  colmap_sparse

done
