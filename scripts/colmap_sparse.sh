DATASET_PATH=$1

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --SiftExtraction.max_image_size 10000 \
   --SiftExtraction.max_num_features 50000 \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model OPENCV \
   --ImageReader.camera_params "2303.14339,2312.64462,2562.53024,1925.10736,-0.25332645,0.09693553,-0.00118667,-0.00139854"

# Sequential Match
colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.max_num_matches 50000

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --Mapper.ba_refine_principal_point 1
