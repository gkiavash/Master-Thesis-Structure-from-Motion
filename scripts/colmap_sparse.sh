DATASET_PATH=$1
is_distorted=$2
max_num_features=$3


# The calib params below are obtained from the checkerboard dataset taken by camera with video settings
CALIB_PARAMS_YAML_4="1333.0333496269382,1152.7953767273623,1362.2385792503271,705.0287442757885,-0.1697821786723204,0.08551140923416184,-0.0019370125115926041,-0.005935762788516092,-0.02775070400965532,0,0,0"

#// Full OpenCV camera model.
#// fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

if [[ $is_distorted -eq 1 ]]
then
    colmap feature_extractor \
       --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images \
       --SiftExtraction.max_image_size 10000 \
       --SiftExtraction.max_num_features $max_num_features \
       --ImageReader.single_camera 1 \
       --ImageReader.camera_model FULL_OPENCV \
       --ImageReader.camera_params $CALIB_PARAMS_YAML_4
else
    colmap feature_extractor \
       --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images \
       --SiftExtraction.max_image_size 10000 \
       --SiftExtraction.max_num_features $max_num_features \
       --ImageReader.single_camera 1
fi


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
