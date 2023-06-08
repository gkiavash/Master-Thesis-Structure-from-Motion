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
        self.ply.estimate_normals()

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1,
            origin=[0, 0, 0]
        )

    def radius(self):
        distances = self.ply.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        # radius = 3 * avg_dist
        # print("radius:", radius)
        return avg_dist

    def down_sampling(self, coef=1., overwrite=True, preview=True):
        ply_uniformed = self.ply.voxel_down_sample(voxel_size=self.radius() * coef)

        if overwrite:
            self.ply = ply_uniformed

        if preview:
            o3d.visualization.draw_geometries([ply_uniformed, self.mesh_frame])

        return ply_uniformed

    def squeeze_to_z(self, overwrite=True, preview=True):
        for p in self.ply.points:
            p[2] = 0

    def move_center(self):
        self.ply.translate([0, 0, 0], relative=False)

    def scale(self, scale):
        self.ply.scale(scale, center=self.ply.get_center())

    def sort(self):
        # ply.ply.points = a[a[:, 0].argsort()]
        pass

    def slice(self, threshold, overwrite=True, preview=False):
        """
        :param threshold: percentage from z_min
        """
        assert 0 < threshold < 1
        ply_np = np.asarray(self.ply.points)
        z_min = ply_np[:, 2].min()
        z_max = ply_np[:, 2].max()
        z_threshold = (z_max - z_min) * threshold

        ply_sliced = self.ply.select_by_index(np.where(ply_np[:, 2] < z_min + z_threshold)[0])

        print(self.ply)
        print(ply_sliced)
        print(1, z_min + z_threshold)

        if preview:
            o3d.visualization.draw_geometries([ply_sliced, self.mesh_frame])
        if overwrite:
            self.ply = ply_sliced

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

    def align_to_z(self, z_=-1, overwrite=False, preview=False):
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
            o3d.visualization.draw_geometries([self.ply, ply_aligned, self.mesh_frame])

        if overwrite:
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
            o3d.visualization.draw_geometries([ply_new, self.mesh_frame])
        if overwrite:
            self.ply = ply_new

        return ply_new

    def erosion(self, overwrite=False, preview=False):
        print("Apply erosion")
        ply_np = np.asarray(self.ply.points)

        radius = self.radius() * 2
        filtered_indexes = []
        for ind, point in enumerate(ply_np):

            distances = np.linalg.norm(ply_np - point, axis=1)
            indices = np.where(distances <= radius)
            # print(np.mean(distances))
            points_around = ply_np[indices]
            # print(points_around.shape, len(indices))
            points_around_centroid = np.mean(points_around, axis=0)

            distance_with_centroid = np.linalg.norm(point - points_around_centroid)
            if distance_with_centroid <= radius * .1:
                filtered_indexes.append(ind)

        points_filtered = ply_np[filtered_indexes]

        ply_new = o3d.geometry.PointCloud()
        ply_new.points = o3d.utility.Vector3dVector(points_filtered)

        if preview:
            o3d.visualization.draw_geometries([ply_new])

        if overwrite:
            self.ply = ply_new

    def detect_crossroads(self, overwrite=False, preview=False):
        ply_np = np.asarray(self.ply.points)
        r_ = self.radius()

        filtered_indexes = []
        for ind, point in enumerate(ply_np):

            distances = np.linalg.norm(ply_np - point, axis=1)
            indices = np.where(distances <= r_ * 10)
            points_around = ply_np[indices]
            points_around_centroid = np.mean(points_around, axis=0)
            distance_with_centroid = np.linalg.norm(point - points_around_centroid)
            print(points_around.shape, distance_with_centroid)
            if distance_with_centroid > r_ * 3:
                filtered_indexes.append(ind)

        points_filtered = ply_np[filtered_indexes]
        ply_new = o3d.geometry.PointCloud()
        ply_new.points = o3d.utility.Vector3dVector(points_filtered)

        if preview:
            o3d.visualization.draw_geometries([ply_new])

        if overwrite:
            self.ply = ply_new

    def iter_points_around(self, radius):
        ply_np = np.asarray(self.ply.points)
        for ind, point in enumerate(ply_np):
            if ind != 4511:
                continue
            distances = np.linalg.norm(ply_np - point, axis=1)
            indices = np.where(distances <= radius)
            points_around = ply_np[indices]
            yield ind, point, points_around

    def dataset(self):
        self.ply.translate([
            # ply_np[:, 0].max(),
            # ply_np[:, 1].max(),
            # ply_np[:, 2].max()
            100, 100, 100
        ], relative=False)

        self.show_range_coords()
        r_ = self.radius() * 20
        import matplotlib.pyplot as plt


        for ind, point, points_around in self.iter_points_around(r_):
            z = (
                int(points_around[:, 0].max() - points_around[:, 0].min()),
                int(points_around[:, 1].max() - points_around[:, 1].min()),
            )
            print(z)
            image = np.zeros(z)
            points_around_x_min = int(points_around[:, 0].min())
            points_around_y_min = int(points_around[:, 1].min())

            for point_around in points_around:
                print(int(point_around[0]), int(point_around[1]))
                image[
                    int(point_around[0]) - (points_around_x_min + 2),
                    int(point_around[1]) - (points_around_y_min + 2)
                ] = 1

            plt.imshow(image)
            plt.show()
            print()

            print(points_around.shape, points_around[:, 2].max())

    def detect_crossroads_2(self, overwrite=False, preview=False):
        def get_boundries(coef, r):
            area_all = 3.14 * ((r*coef) ** 2)
            area_each = 3.14 * ((r*2) ** 2)
            density = area_all / area_each
            print(area_all, area_each, density)
            pass

        r_ = self.radius() * 20
        print("r_", r_)
        filtered_indexes = []
        points_sum = {}
        points_num = 0
        for ind, point, points_around in self.iter_points_around(r_):
            print(points_around.shape)
            points_sum.update({ind: len(points_around)})
            points_num += 1
            # distances_around = np.linalg.norm(points_around - point, axis=1)
            # print(distances_around)
            # print()
        points_sum_ = 0
        for i in points_sum:
            points_sum_ += points_sum[i]
        ave_ = points_sum_/points_num

        num_ = 0
        for i in points_sum:
            if ave_ * 0.95 < points_sum[i] < ave_ * 1.1:
                filtered_indexes.append(i)
                num_ += 1
                print(points_sum[i])
        print(123)

        points_filtered = ply_np[filtered_indexes]
        ply_new = o3d.geometry.PointCloud()
        ply_new.points = o3d.utility.Vector3dVector(points_filtered)

        if preview:
            o3d.visualization.draw_geometries([ply_new])
        return ply_new

    def save(self, name):
        output_path = Path(os.path.dirname(os.path.abspath(self.ply_path))) / name
        print("writing to:", output_path)
        o3d.io.write_point_cloud(str(output_path), self.ply)


if __name__ == "__main__":
    # street_4_0_1:
    # ply_path = "D:\sfm_dense\street_4_0_1\color_pd30_8bit_street_4_0_sliced_uniformed.ply"

    # street_4_1_3:
    # ply_path = "D:\sfm_dense\street_4_1_3\street_4_1_3_color_pd30_8bit_small.ply"
    # ply_path = "D:\sfm_dense\street_4_1_3\street_4_1_3_color_pd30_8bit_small_ele_uniformed_sliced.ply"
    # ply_path = "D:\sfm_dense\street_4_1_3\street_4_1_3_color_pd30_8bit_small_ele_uniformed_sliced_erosion.ply"
    # ply_path = "D:/sfm_dense/street_4_1_3/dense/0/fused.ply"
    # ply_path = "D:/sfm_dense/street_4_1_3/dense/0/fused_ele_uniformed_aligned_sliced.ply"

    # street_4_2_1
    # ply_path = "D:\sfm_dense\street_4_2_1\color_pd30_8bit_small.ply"
    # ply_path = "D:/sfm_dense/street_4_2_1/dense/0/fused.ply"

    # street_4_3_2
    # ply_path = "D:\sfm_dense\street_4_3_2\street_4_3_2_color_pd30_8bit_small.ply"
    # ply_path = "D:/sfm_dense/street_4_3_2/dense/0/fused.ply"
    # ply_path = "D:/sfm_dense/street_4_3_2/dense/0/street_4_3_2_fused_ele_uniformed_aligned.ply"

    # street_4_3_3
    # ply_path = "D:/sfm_dense/street_4_3_3/dense/0/fused.ply"
    # ply_path = "D:/sfm_dense/street_4_3_3/dense/0/fused_uniformed_aligned_sliced.ply"
    ply_path = "D:/sfm_dense/street_4_3_3/dense/0/fused_uniformed_aligned_sliced_erosion.ply"
    # ply_path = "D:/sfm_dense/street_4_3_3/color_pd30_8bit_small_street_4_3_3.ply"
    # ply_path = "D:\sfm_dense\street_4_3_3\color_pd30_8bit_small_street_4_3_3_ele_uniformed_sliced.ply"

    # good for crossroads:
    # ply_path = "D:\sfm_dense\street_4_3_3\color_pd30_8bit_small_street_4_3_3_ele_uniformed_sliced_erosion.ply"

    ply = CustomPLY(ply_path)
    # ply.move_center()
    # ply.align_to_z(z_=-1, overwrite=True, preview=False)  # city = 1, sfm = -1
    ply.show_range_coords()
    # ply.dataset()
    # ply.elevation_map(overwrite=True, preview=False)
    # ply.slice(0.2, overwrite=True, preview=True)
    # ply.squeeze_to_z()
    # ply.down_sampling(coef=10, overwrite=True, preview=True)  # city = 0.5, sfm = 0.8

    # ply.erosion(overwrite=True, preview=True)
    # ply_ero = copy.deepcopy(ply.ply)
    # ply_ero.paint_uniform_color([0, 0.651, 0.929])
    print()
    ply.detect_crossroads(preview=True)

    o3d.visualization.draw_geometries(geometry_list=[
        ply.ply,
        ply_ero,
        ply.mesh_frame
    ],
        point_show_normal=True
    )

    # ply.down_sampling(coef=0.8, overwrite=True, preview=True)  # only for sfm
    # ply.show_range_coords()
    # ply.move_center()
    # ply.show_range_coords()
    # ply.show_range_coords()
    # ply.save("fused_ele_uniformed_aligned_squeezed.ply")
