import copy
import glob
from functools import cached_property
import os
from pathlib import Path

import open3d as o3d
import numpy as np
from utils import preprocess_point_cloud, radius


class RegisterCorr:
    def __init__(self, source_path, target_path, **kwargs):
        self.source_path = source_path  # sfm
        self.target_path = target_path  # city

        self.source = o3d.io.read_point_cloud(self.source_path)
        self.target = o3d.io.read_point_cloud(self.target_path)
        self.corres = kwargs.get("corres", None)
        self.scale_factor = kwargs.get("scale_factor", None)

    def preprocess(self):
        self.scale(self.scale_factor if self.scale_factor is not None else self.get_scale_factor())
        self.density_normalize(coef=1.5, source=False)

        self.source_down, self.source_fpfh = preprocess_point_cloud(self.source)
        self.target_down, self.target_fpfh = preprocess_point_cloud(self.target)
        self.source_down = self.source_down.translate([100, 100, 0], relative=False)

    def draw_registration_result(self, transformation=None):
        if transformation is None:
            transformation = np.asarray([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        source_temp = copy.deepcopy(self.source)
        target_temp = copy.deepcopy(self.target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def global_registration(self, distance_threshold=None):
        print("Register using features")
        if distance_threshold is None:
            distance_threshold = radius(self.source) * 1.5

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=self.source_down,
            target=self.target_down,
            source_feature=self.source_fpfh,
            target_feature=self.target_fpfh,
            mutual_filter=False,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.99999)
        )
        self.result = result
        return self.result

    def global_registration_corr(self, distance_threshold=None):
        if distance_threshold is None:
            distance_threshold = radius(self.source) * 10
        print("Registering using correspondence", "distance_threshold", distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            self.source_down,
            self.target_down,
            self.corres,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.99999)
        )
        self.result = result
        return self.result

    def register(self, **kwargs):
        assert self.source_down is not None
        if self.corres:
            result = self.global_registration_corr()
        else:
            result = self.global_registration()

        print("Global", result)
        self.draw_registration_result(result.transformation)

        evaluation = o3d.pipelines.registration.evaluate_registration(
            self.source, self.target,
            radius(self.source) * 1.5,
            result.transformation
        )
        print("Global", evaluation)
        print("Global Transformation", )
        print(result.transformation)
        print()
        return result

    def get_scale_factor(self):
        print("Calc Scale Factor")
        source_np = np.asarray(self.source.points)
        target_np = np.asarray(self.target.points)
        corres = self.corres

        t_d_0 = np.linalg.norm(target_np[corres[0][1]] - target_np[corres[1][1]])
        t_d_1 = np.linalg.norm(target_np[corres[1][1]] - target_np[corres[2][1]])

        s_d_0 = np.linalg.norm(source_np[corres[0][0]] - source_np[corres[1][0]])
        s_d_1 = np.linalg.norm(source_np[corres[1][0]] - source_np[corres[2][0]])
        # print(t_d_0, s_d_0, t_d_0/s_d_0)
        # print(t_d_1, s_d_1, t_d_1/s_d_1)
        scale_ratio_0 = float(t_d_0/s_d_0)
        scale_ratio_1 = float(t_d_1/s_d_1)

        scale_factor = sum([scale_ratio_0, scale_ratio_1])/2
        print("scale_ratio_0:", scale_ratio_0)
        print("scale_ratio_1:", scale_ratio_1)
        print("scale_factor calculated:", scale_factor)
        return scale_factor

    def scale(self, factor):
        self.source.scale(factor, center=self.source.get_center())

    def density_normalize(self, coef=1., source=True):
        if source:
            distances = self.target.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances) * coef
            self.source = r.source.voxel_down_sample(voxel_size=avg_dist)
        else:
            distances = self.source.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances) * coef
            self.target = r.target.voxel_down_sample(voxel_size=avg_dist)

    def draw_result_corres(self, source, target, result):
        lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            source, target, result.correspondence_set
        )
        o3d.visualization.draw_geometries([source, target, lines])


if __name__ == "__main__":
    # street_4_3_3
    # path_target = "D:\sfm_dense\street_4_3_3\color_pd30_8bit_small_street_4_3_3_ele_uniformed_sliced.ply"
    # path_source = "D:/sfm_dense/street_4_3_3/dense/0/fused_extra_uniformed_aligned_sliced.ply"
    # target_indexes = [
    #     5572,
    #     4510,
    #     2124,
    #     2624
    # ]
    # source_indexes = [
    #     1811,
    #     1984,
    #     9395,
    #     566
    # ]
    # path_target = "D:\sfm_dense\street_4_3_3\color_pd30_8bit_small_street_4_3_3_ele_uniformed_sliced_erosion.ply"
    # path_source = "D:/sfm_dense/street_4_3_3/dense/0/fused_uniformed_aligned_sliced_erosion.ply"
    # scale_factor = 11.23

    # street_4_1_3
    path_source = "D:/sfm_dense/street_4_1_3/dense/0/fused_ele_uniformed_aligned_squeezed.ply"
    path_target = "D:\sfm_dense\street_4_1_3\street_4_1_3_color_pd30_8bit_small_ele_uniformed_sliced_erosion.ply"
    scale_factor = 9.97
    target_indexes = [
        7408,
        5957,
        7751,
    ]
    source_indexes = [
        2430,
        472,
        201,
    ]

    corres = o3d.utility.Vector2iVector(list(zip(source_indexes, target_indexes)))
    r = RegisterCorr(
        source_path=path_source,
        target_path=path_target,
        corres=corres
        # scale_factor=scale_factor
    )
    r.preprocess()

    r.draw_registration_result()

    res = r.register(corres=corres)
    r.draw_result_corres(r.source, r.target, res)

    print("hi")

    # to visualize features
    t_ = r.target_fpfh
    s_ = r.source_fpfh

    from matplotlib import pyplot as plt
    plt.imshow(t_.data[:, :200], interpolation='nearest')
    plt.show()

    plt.imshow(s_.data[:, :200], interpolation='nearest')
    plt.show()

