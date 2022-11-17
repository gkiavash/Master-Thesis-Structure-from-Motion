WORKPLACE_PATH=$1

colmap image_undistorter \
    --image_path $WORKPLACE_PATH/images \
    --input_path $WORKPLACE_PATH/refined/sfm_t1/colmap \
    --output_path $WORKPLACE_PATH/refined/sfm_t1/dense \
    --output_type COLMAP

colmap patch_match_stereo \
    --workspace_path $WORKPLACE_PATH/refined/sfm_t1/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.gpu_index 0,1,2

colmap stereo_fusion \
    --workspace_path $WORKPLACE_PATH/refined/sfm_t1/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $WORKPLACE_PATH/refined/sfm_t1/dense/fused.ply
