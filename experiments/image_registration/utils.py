import copy
import glob
from functools import cached_property
import os
from pathlib import Path

import open3d as o3d
import numpy as np


def radius(ply):
    distances = ply.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    # print("radius:", radius)
    return radius


def preprocess_point_cloud(pcd):
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = pcd

    radius_ = radius(pcd)
    radius_normal = radius_ * 10
    radius_feature = radius_ * 2
    print("r", radius_feature)
    max_nn = 10
    # radius_feature = voxel_size * 5
    # radius_normal = voxel_size * 2

    # pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
    pcd_down.estimate_normals()
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn)
    )
    for row_ in pcd_fpfh.data:
        for col_ in row_:
            if col_ < 10:
                col_ *= 20
    return pcd_down, pcd_fpfh


def show_range_coords(ply):
    ply_np = np.asarray(ply.points)
    print()
    for i in range(3):
        print(
            ply_np[:, i].min(),
            ply_np[:, i].max(),
            "range:",
            ply_np[:, i].max() - ply_np[:, i].min()
        )
