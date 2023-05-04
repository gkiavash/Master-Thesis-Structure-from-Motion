import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


ply_source = "D:/sfm_dense/street_1_0_real_time/target.ply"
# ply_target = "D:/sfm_dense/street_1_0_query/dense/0/fused.ply"
ply_target = "D:/sfm_dense/street_1_0_query/query.sparse.ply"

source = o3d.io.read_point_cloud(ply_source)
target = o3d.io.read_point_cloud(ply_target)
threshold = 0.05
# trans_init = np.asarray([
#     [0.862, 0.011, -0.507, 11.9],
#     [-0.139, 0.967, -0.215, 0.1],
#     [0.487, 0.255, 0.835, -1.4],
#     [0.0, 0.0, 0.0, 1.0]
# ])
trans_init = np.asarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
draw_registration_result(source, target, trans_init)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
print(evaluation)
print()


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size
):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


voxel_size = 0.1  # means 5cm for the dataset
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# result_fast = execute_fast_global_registration(
result_fast = execute_global_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size
)
print("Global", result_fast)
draw_registration_result(source_down, target_down, result_fast.transformation)
evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, result_fast.transformation)
print("Global", evaluation)
print("Global Transformation",)
print(result_fast.transformation)
print()

threshold = 1
print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target,
    threshold, result_fast.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
print("")
draw_registration_result(source, target, reg_p2p.transformation)

print("Apply point-to-plane ICP")
reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target,
    threshold, result_fast.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
print("")
draw_registration_result(source, target, reg_p2l.transformation)

