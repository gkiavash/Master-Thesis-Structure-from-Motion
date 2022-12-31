DATASET_PATH=$1

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --SiftExtraction.max_image_size 10000 \
   --SiftExtraction.max_num_features 32768 \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model OPENCV_FISHEYE \
   --ImageReader.camera_params 2.2880100739220425e+03,2.2845293722010042e+03,2.5716169994772490e+03,1.9205706165878657e+03,-2.6434196998339271e-01,9.9571471417386093e-02,-2.4160314862664079e-04,-2.2267220647390027e-04,-1.9631169477800196e-02,0.,0.,0.

# Sequential Match
colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.max_num_matches 32768

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
