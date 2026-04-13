import os
import numpy as np


def gaussian_generator(cam_infos, cam_intrinsics, num_points, image_depth_scale_map):
    """
    Note: Not guaranteed to return exactly num_points
    Usually, number of returned points is 33% lower than num_points
    """

    num_images = len(cam_infos)
    num_points_per_image = num_points // num_images

    assert num_points >= num_images, "Generate more points than number of images"

    xyzs, rgbs, normals = [], [], []
    for i, cam_info in enumerate(cam_infos):
        fx, fy, cx, cy = cam_intrinsics[cam_info.uid].params

        # Sample from 2D gaussian with 50% samples coming from central portion of image bounded by ([-w/4, w/4], [-h/4, h/4])
        xs = np.random.normal(loc=cx, scale=cam_info.width / 2.696, size=num_points_per_image).astype(int)
        ys = np.random.normal(loc=cy, scale=cam_info.height / 2.696, size=num_points_per_image).astype(int)

        valid_mask = (xs >= 0) & (xs < cam_info.width) & (ys >= 0) & (ys < cam_info.height)

        xs = xs[valid_mask]
        ys = ys[valid_mask]

        image = cam_info.image # RGBA image
        image = np.array(image)

        rgb = image[ys, xs, :3]
        normal = cam_info.world_normal_map[ys, xs, :]
        depth = cam_info.depth_map[ys, xs]

        depth = depth * image_depth_scale_map[cam_info.image_id]["estimated_scale"]

        camera_coords = np.stack([(xs - cx) / fx * depth, (ys - cy) / fy * depth, depth], axis=0)

        # Camera to world transform
        world_coords = cam_info.R @ (camera_coords - cam_info.T[:, None])
        world_coords = world_coords.T

        xyzs.append(world_coords)
        rgbs.append(rgb)
        normals.append(normal)

    xyzs = np.vstack(xyzs)
    rgbs = np.vstack(rgbs)
    normals = np.vstack(normals)

    return xyzs, rgbs, normals