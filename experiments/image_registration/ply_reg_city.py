import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt


source = o3d.io.read_point_cloud("E:/University/Thesis/color_pd30_8bit_small_filtered.ply")
source_np = np.asarray(source.points)
for i in range(3):
    print(source_np[:, i].min(),
          source_np[:, i].max(),
          source_np[:, i].min() - source_np[:, i].max()
          )

plt.plot(source_np[:, 0], source_np[:, 1])
plt.show()
o3d.visualization.draw_geometries([target])

source_new = source.select_by_index(np.where(source_np[:, 2] < 70)[0])
source_new.translate([0, 0, 0], relative=False)
source_new.scale(1, center=source_new.get_center())

o3d.io.write_point_cloud("E:/University/Thesis/color_pd30_8bit_small_filtered.ply", source_new)


target = o3d.io.read_point_cloud("D:/sfm_dense/street_2_0/dense/0/fused_filtered_down_aligned.ply")
target_np = np.asarray(target.points)
for i in range(3):
    print(target_np[:, i].min(), target_np[:, i].max(), target_np[:, i].min() - target_np[:, i].max())

target.translate([0, 0, 0], relative=False)
target.scale(20, center=target.get_center())
o3d.io.write_point_cloud("D:/sfm_dense/street_2_0/dense/0/fused_filtered.ply", target)

target.sample_points_uniformly(number_of_points=1000)
target.uniform_down_sample(100)
target_down = target.voxel_down_sample(voxel_size=5)


o3d.visualization.draw_geometries([target])




def get_plane(ply):
    plane_model, inliers = ply.segment_plane(
        distance_threshold=5,
        ransac_n=1000,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = ply.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 1.0, 0])

    outlier_cloud = ply.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([1.0, 0, 0])

    o3d.visualization.draw_geometries([inlier_cloud])
    o3d.visualization.draw_geometries([outlier_cloud])

get_plane(source_scaled_uniformed)

def align_ply_to_z(ply):
    plane_model, inliers = ply.segment_plane(
        distance_threshold=5,
        ransac_n=1000,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    plane_n = np.array([a, b, c])
    plane_z = np.array([0, 0, -1])

    vec1 = np.reshape(plane_n, (1, -1))
    vec2 = np.reshape(plane_z, (1, -1))

    from scipy.spatial.transform import Rotation as R
    r = R.align_vectors(vec2, vec1)
    rot = r[0].as_matrix()

    ply_aligned = copy.deepcopy(ply)
    ply_aligned.rotate(rot, center=ply_aligned.get_center())
    o3d.visualization.draw_geometries([ply, ply_aligned])
    return ply_aligned

source_scaled_uniformed_aligned = align_ply_to_z(source_scaled_uniformed)



# street_4_0
import copy
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt


source = "D:/sfm_dense/street_4_0_1/dense/0/fused.ply"
target = "D:/sfm_dense/color_pd30_8bit_street_4_0.ply"

source = o3d.io.read_point_cloud(source)
target = o3d.io.read_point_cloud(target)

source.translate([0, 0, 0], relative=False)
target.translate([0, 0, 0], relative=False)

def ply_show_range_coords(ply):
    ply_np = np.asarray(ply.points)
    print()
    for i in range(3):
        print(
            ply_np[:, i].min(),
            ply_np[:, i].max(),
            "range:",
            ply_np[:, i].max() - ply_np[:, i].min()
        )

ply_show_range_coords(source)
ply_show_range_coords(target)

# target
target_sliced = target.select_by_index(np.where(np.asarray(target.points)[:, 2] < 0)[0])
o3d.visualization.draw_geometries([target_sliced])

target_sliced_uniformed = target_sliced.voxel_down_sample(voxel_size=1)
o3d.visualization.draw_geometries([target_sliced_uniformed])


# source
o3d.visualization.draw_geometries([source])

source_scaled = source.scale(20, center=source.get_center())

source_scaled_uniformed = source_scaled.voxel_down_sample(voxel_size=1)
o3d.visualization.draw_geometries([source_scaled_uniformed])


# scaling
def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])


ply_show_range_coords(source_scaled_uniformed)
ply_show_range_coords(target_sliced_uniformed)
draw_registration_result(source_scaled_uniformed_aligned, target_sliced_uniformed)



# save

o3d.io.write_point_cloud("D:/sfm_dense/street_4_0_1/dense/0/fused_scaled_uniformed.ply", source_scaled_uniformed)
o3d.io.write_point_cloud("D:/sfm_dense/color_pd30_8bit_street_4_0_sliced_uniformed.ply", target_sliced_uniformed)


# Test
# This function is only used to make the keypoints look better on the rendering
def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres

source = "D:/sfm_dense/street_4_0_1/dense/0/fused.ply"
target = "D:/sfm_dense/color_pd30_8bit_street_4_0.ply"

source = o3d.io.read_point_cloud(source)
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(source,
                                                        salient_radius=0.5,
                                                        non_max_radius=1.0,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)

source.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), source])
