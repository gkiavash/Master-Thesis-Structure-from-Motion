# Change DATASET_PATH
DATASET_PATH=/home/gkiavash/Downloads/sfm_street_1

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --SiftExtraction.max_num_features 8192 \
   --ImageReader.single_camera 1

# Sequential Match
colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.max_num_matches 8192 \
   --SiftMatching.min_num_inliers 100
   
# Normal Match
#colmap exhaustive_matcher \
#   --database_path $DATASET_PATH/database.db \
#   --SiftMatching.max_num_matches 8192 \
#   --SiftMatching.min_num_inliers 100

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

sudo mkdir $DATASET_PATH/dense

sudo colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP


sudo colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.num_iterations 2

sudo colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

#sudo colmap poisson_mesher \
#    --input_path $DATASET_PATH/dense/fused.ply \
#    --output_path $DATASET_PATH/dense/meshed-poisson.ply
#
#sudo colmap delaunay_mesher \
#    --input_path $DATASET_PATH/dense \
#    --output_path $DATASET_PATH/dense/meshed-delaunay.ply
