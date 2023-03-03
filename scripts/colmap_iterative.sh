DATASET_PATH=$1


colmap_sparse_() {
  colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/incoming \
    --SiftExtraction.max_image_size 10000 \
    --SiftExtraction.max_num_features 50000 \
    --ImageReader.existing_camera_id 1 \
    --image_list_path $DATASET_PATH/image-list.txt

  colmap sequential_matcher \
    --database_path $DATASET_PATH/database.db \
    --SiftMatching.max_num_matches 32000

  colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/incoming \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0
}

# 1) start with first images
sh /home/ghamsariki/Master-Thesis-Structure-from-Motion/scripts/colmap_sparse.sh $DATASET_PATH


# 2) add new images iteratively
for FILE in $DATASET_PATH/incoming/*; do
  cp $FILE $DATASET_PATH/images/

  echo "$(basename $FILE)"
  echo "$(basename $FILE)" >> $DATASET_PATH/image-list.txt

  colmap_sparse_

done
