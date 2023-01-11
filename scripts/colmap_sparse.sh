DATASET_PATH=$1

CALIB_PARAMS_OPENCV="2303.14339,2312.64462,1352,769,-0.25332645,0.09693553,-0.00118667,-0.00139854,-0.02359868,0,0,0"
CALIB_PARAMS_YAML="2288.0100739220425,2284.529372201004,1352,769,-0.2643419699833927,0.09957147141738609,-0.0002416031486266408,-0.00022267220647390027,-0.019631169477800196,0,0,0"

# The calib params below are obtained from the checkerboard dataset taken by camera with video settings
CALIB_PARAMS_YAML_3="1267.305126032294,1161.14123676566,1370.030430369817,719.7204648074699,-0.22329932675609362,0.25083021808893313,-0.00243685822518905,0.0028180643499969675,-0.15365831433891083,0,0,0"
CALIB_PARAMS_YAML_4="1333.0333496269382,1152.7953767273623,1362.2385792503271,705.0287442757885,-0.1697821786723204,0.08551140923416184,-0.0019370125115926041,-0.005935762788516092,-0.02775070400965532,0,0,0"

#// Full OpenCV camera model.
#// fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --SiftExtraction.max_image_size 10000 \
   --SiftExtraction.max_num_features 50000 \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model FULL_OPENCV \
   --ImageReader.camera_params $CALIB_PARAMS_YAML_4

# Sequential Match
colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.max_num_matches 32000

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --Mapper.ba_refine_principal_point 1
