DATASET_PATH=$1


colmap point_triangulator \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/model \
    --output_path $DATASET_PATH/sparse/model
