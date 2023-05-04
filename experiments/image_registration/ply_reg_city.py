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
