DATASET_PATH=$1

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --SiftExtraction.max_image_size 10000 \
   --SiftExtraction.max_num_features 50000 \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model OPENCV \
   --ImageReader.camera_params "2288.0100739220425,2284.529372201004,2571.616999477249,1920.5706165878657,-0.2643419699833927,0.09957147141738609,-0.0002416031486266408,-0.00022267220647390027"

# Sequential Match
colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.max_num_matches 50000

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
