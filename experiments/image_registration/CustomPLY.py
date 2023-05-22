import copy
from functools import cached_property
import os
from pathlib import Path

import open3d as o3d
import numpy as np


class CustomPLY:
    def __init__(self, ply_path):
        self.ply_path = ply_path
        self.ply = o3d.io.read_point_cloud(self.ply_path)

    @cached_property
    def radius(self):
        distances = self.ply.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        print("radius:", radius)
        return radius

    def down_sampling(self, overwrite=True, preview=True):
        ply_uniformed = self.ply.voxel_down_sample(voxel_size=self.radius)

        if overwrite:
            self.ply = ply_uniformed

        if preview:
            o3d.visualization.draw_geometries([ply_uniformed])

        return ply_uniformed

    def move_center(self):
        self.ply.translate([0, 0, 0], relative=False)

    def scale(self, scale):
        self.ply.scale(scale, center=self.ply.get_center())

    def show_range_coords(self):
        ply_np = np.asarray(self.ply.points)
        print()
        for i in range(3):
            print(
                ply_np[:, i].min(),
                ply_np[:, i].max(),
                "range:",
                ply_np[:, i].max() - ply_np[:, i].min()
            )

    def align_to_z(self, z_=-1, preview=True):
        plane_model, inliers = self.ply.segment_plane(
            distance_threshold=self.radius,
            ransac_n=1000,
            num_iterations=1000
        )
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        plane_n = np.array([a, b, c])
        plane_z = np.array([0, 0, z_])

        vec1 = np.reshape(plane_n, (1, -1))
        vec2 = np.reshape(plane_z, (1, -1))

        from scipy.spatial.transform import Rotation as R
        r = R.align_vectors(vec2, vec1)
        rot = r[0].as_matrix()

        ply_aligned = copy.deepcopy(self.ply)
        ply_aligned.rotate(rot, center=ply_aligned.get_center())

        if preview:
            o3d.visualization.draw_geometries([self.ply, ply_aligned])

        self.ply = ply_aligned
        return ply_aligned

    def elevation_map(self, overwrite=False, preview=False):
        ply_np = np.asarray(self.ply.points)

        x_range = ply_np[:, 0].max() - ply_np[:, 0].min()
        y_range = ply_np[:, 1].max() - ply_np[:, 1].min()

        z_min = ply_np[:, 2].min()
        z_max = ply_np[:, 2].max()
        z_range = z_max - z_min

        if not min(x_range, y_range, z_range) == z_range:
            raise Exception("not aligned with z axis")

        new_colors = []
        for point in ply_np:
            new_colors.append([0, (point[2] - z_min) / z_range, 0])
            # new_colors.append([0.5, 0, 0])

        ply.colors = o3d.utility.Vector3dVector(
            np.array(new_colors)
        )
        ply_new = o3d.geometry.PointCloud()

        ply_new.points = o3d.utility.Vector3dVector(ply_np)
        ply_new.colors = o3d.utility.Vector3dVector(np.array(new_colors))

        if preview:
            o3d.visualization.draw_geometries([ply_new])
        if overwrite:
            self.ply = ply_new

        return ply_new

    def save(self, name):
        output_path = Path(os.path.dirname(os.path.abspath(self.ply_path))) / name
        print("writing to:", output_path)
        o3d.io.write_point_cloud(str(output_path), self.ply)


if __name__ == "__main__":
    ply_path = "D:/sfm_dense/street_4_3_3/dense/0/fused.ply"
    ply_path = "D:/sfm_dense/street_4_3_3/dense/0/fused_uniformed_aligned.ply"
    ply_path = "D:/sfm_dense/street_4_3_3/color_pd30_8bit_small_street_4_3_3.ply"

    ply = CustomPLY(ply_path)
    # ply.down_sampling(overwrite=True, preview=False)
    ply.show_range_coords()
    ply.move_center()
    ply.show_range_coords()
    ply.align_to_z(z_=1, preview=False)  # city = 1, sfm = -1
    # ply.show_range_coords()
    # ply.save("street_4_3_3_fused_uniformed_aligned.ply")
    ply.elevation_map(preview=True)
